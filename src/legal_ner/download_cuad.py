"""Fetch and validate the CUAD dataset (spec section 2).

Source: the canonical Hugging Face dataset `theatticusproject/cuad-qa` (CC BY 4.0),
loaded at runtime via `datasets` (preferred — see load_cuad_hf). A legacy zip-based
fetch (download_cuad) is kept as a fallback. We never re-host or bundle the raw data.

The HF dataset is FLAT (one row per question/context); hf_rows_to_cuad adapts it into
the nested SQuAD shape that convert_cuad consumes. The adaptation, validation and stats
logic are pure functions, unit-tested without the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

from . import config

# Preferred source: the canonical Hugging Face dataset (CC BY 4.0). This is more robust
# than scraping a raw GitHub URL (which moves) and needs no auth. See load_cuad_hf().
HF_DATASET_ID = "theatticusproject/cuad-qa"

# Legacy fallback: the repository historically distributed CUAD_v1.json inside a zip.
# NOTE: the exact raw path has changed over time; prefer the HF loader above.
CUAD_ZIP_URL = (
    "https://github.com/TheAtticusProject/cuad/raw/main/CUAD_v1.zip"
)
# Member name of the JSON inside the zip (CUAD ships it under a top-level folder).
_ZIP_MEMBER_SUFFIX = "CUAD_v1.json"


@dataclass
class CuadStats:
    contracts: int
    qa_pairs: int
    impossible: int
    categories: int


def validate_cuad(cuad: dict) -> None:
    """Raise ValueError unless ``cuad`` is a well-formed CUAD object.

    Checks the top-level "data" key exists and is a non-empty list, and that at least
    the first entry has the expected paragraph/qas shape.
    """
    if not isinstance(cuad, dict) or "data" not in cuad:
        raise ValueError("CUAD JSON missing top-level 'data' key")
    data = cuad["data"]
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("CUAD 'data' must be a non-empty list")
    first = data[0]
    if "paragraphs" not in first or not first["paragraphs"]:
        raise ValueError("CUAD entry missing 'paragraphs'")
    if "qas" not in first["paragraphs"][0]:
        raise ValueError("CUAD paragraph missing 'qas'")


def cuad_stats(cuad: dict) -> CuadStats:
    """Compute basic corpus statistics (section 2 logging requirement)."""
    contracts = 0
    qa_pairs = 0
    impossible = 0
    categories: set[str] = set()
    # Import here to avoid a hard dependency cycle at module import time.
    from .cuad_to_bio import extract_category

    for entry in cuad["data"]:
        contracts += 1
        for para in entry["paragraphs"]:
            for qa in para.get("qas", []):
                qa_pairs += 1
                if qa.get("is_impossible"):
                    impossible += 1
                cat = extract_category(qa.get("question", ""))
                if cat is not None:
                    categories.add(cat)
    return CuadStats(contracts, qa_pairs, impossible, len(categories))


def load_cuad_json(path: str) -> dict:
    """Load and validate a CUAD_v1.json from disk."""
    with open(path, "r", encoding="utf-8") as f:
        cuad = json.load(f)
    validate_cuad(cuad)
    return cuad


def _extract_json_from_zip(zip_path: str, dest_json_path: str) -> str:
    """Extract the CUAD_v1.json member from a downloaded zip to dest_json_path."""
    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith(_ZIP_MEMBER_SUFFIX)), None
        )
        if member is None:
            raise ValueError(f"{_ZIP_MEMBER_SUFFIX} not found inside {zip_path}")
        with zf.open(member) as src, open(dest_json_path, "wb") as dst:
            dst.write(src.read())
    return dest_json_path


def download_cuad(
    dest_dir: str | None = None,
    url: str = CUAD_ZIP_URL,
    force: bool = False,
    limit: int | None = None,
) -> tuple[dict, CuadStats]:
    """Download CUAD_v1.json (via its zip) into ``dest_dir`` and return (cuad, stats).

    If the JSON already exists and ``force`` is False, the download is skipped. ``limit``
    truncates the returned ``data`` list for fast local development (the on-disk file is
    left complete). Stats are computed on the (possibly truncated) returned object.
    """
    dest_dir = dest_dir or config.PROJECT_DIR
    os.makedirs(dest_dir, exist_ok=True)
    json_path = os.path.join(dest_dir, config.CUAD_JSON_FILENAME)

    if force or not os.path.exists(json_path):
        zip_path = os.path.join(dest_dir, "CUAD_v1.zip")
        urllib.request.urlretrieve(url, zip_path)  # noqa: S310 (trusted project URL)
        _extract_json_from_zip(zip_path, json_path)
        try:
            os.remove(zip_path)
        except OSError:
            pass

    cuad = load_cuad_json(json_path)
    if limit is not None:
        cuad = {**cuad, "data": cuad["data"][:limit]}
    stats = cuad_stats(cuad)
    return cuad, stats


def format_stats(stats: CuadStats) -> str:
    """Human-readable one-liner for logging after download (section 2)."""
    return (
        f"CUAD: {stats.contracts} contracts, {stats.qa_pairs} QA pairs, "
        f"{stats.impossible} impossible, {stats.categories} categories detected "
        f"(expected 41)."
    )


# --------------------------------------------------------------------------- #
# Hugging Face source (preferred)
# --------------------------------------------------------------------------- #
# The HF dataset is FLAT: one row per (question, context) with columns
# id, title, context, question, answers{text[], answer_start[]}. There is no
# is_impossible field — an impossible question simply has an empty answers.text list.
# convert_cuad() expects the nested SQuAD shape data[].paragraphs[].qas[], so we adapt.

def hf_rows_to_cuad(rows) -> dict:
    """Adapt flat HF CUAD rows into the nested {"data": [...]} shape convert_cuad expects.

    Rows are grouped by ``title`` (one contract == one paragraph). ``is_impossible`` is
    synthesized as True when a row's answers list is empty. ``rows`` is any iterable of
    dicts with keys: title, context, question, answers{text, answer_start}.
    """
    by_title: dict[str, dict] = {}
    order: list[str] = []
    for r in rows:
        title = r["title"]
        if title not in by_title:
            by_title[title] = {"context": r["context"], "qas": []}
            order.append(title)

        ans = r.get("answers", {}) or {}
        texts = list(ans.get("text", []) or [])
        starts = list(ans.get("answer_start", []) or [])
        answers = [{"text": t, "answer_start": s} for t, s in zip(texts, starts)]

        by_title[title]["qas"].append(
            {
                "question": r["question"],
                "id": r.get("id", ""),
                "answers": answers,
                "is_impossible": len(answers) == 0,
            }
        )

    data = [
        {"title": t, "paragraphs": [{"context": by_title[t]["context"], "qas": by_title[t]["qas"]}]}
        for t in order
    ]
    return {"data": data}


def load_cuad_hf(
    dataset_id: str = HF_DATASET_ID,
    split: str = "train+test",
    limit: int | None = None,
) -> tuple[dict, CuadStats]:
    """Load CUAD from Hugging Face and return (cuad_nested, stats).

    Uses `datasets.load_dataset`. By default concatenates train+test so the pipeline can
    do its own contract-level split (section 4.4). ``limit`` caps the number of CONTRACTS
    (titles) kept, for fast smoke runs.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    cuad = hf_rows_to_cuad(ds)
    if limit is not None:
        cuad = {**cuad, "data": cuad["data"][:limit]}
    validate_cuad(cuad)
    stats = cuad_stats(cuad)
    return cuad, stats
