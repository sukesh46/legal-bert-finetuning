"""Fetch and validate the CUAD dataset (spec section 2).

Source: https://github.com/TheAtticusProject/cuad (CC BY 4.0). The canonical file is
CUAD_v1.json, SQuAD 2.0-formatted. We always fetch at runtime and never re-host or
bundle the raw file, per CUAD's distribution terms.

The download itself is thin; the validation and stats logic is factored into pure
functions (validate_cuad / cuad_stats) so it can be unit-tested without the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass

from . import config

# The repository distributes CUAD_v1.json inside CUAD_v1.zip at the raw GitHub path.
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
