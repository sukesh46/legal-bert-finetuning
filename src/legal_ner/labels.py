"""BIO label-schema construction and persistence (section 4.1).

The label schema is: "O" plus B-/I- for each of the 41 CUAD categories -> 83 labels.
label2id / id2label are persisted to labels.json so inference never has to guess the
label ordering; this file must travel with every checkpoint.
"""

from __future__ import annotations

import json
import os

from . import config


def build_bio_labels(categories: list[str] | None = None) -> list[str]:
    """Return the ordered BIO label list: ["O", "B-<cat>", "I-<cat>", ...].

    Ordering is deterministic: "O" first, then for each category in its canonical
    order a B- tag immediately followed by its I- tag. This yields 2*N + 1 labels.
    """
    categories = categories if categories is not None else config.CUAD_CATEGORIES
    labels = ["O"]
    for cat in categories:
        labels.append(f"B-{cat}")
        labels.append(f"I-{cat}")
    return labels


def build_label_maps(
    categories: list[str] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return (label2id, id2label) for the BIO schema."""
    labels = build_bio_labels(categories)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def num_labels(categories: list[str] | None = None) -> int:
    """Total number of BIO labels (2 * num_categories + 1)."""
    categories = categories if categories is not None else config.CUAD_CATEGORIES
    return 2 * len(categories) + 1


def save_labels(path: str, categories: list[str] | None = None) -> str:
    """Persist the label schema to a JSON file at ``path``.

    The payload stores both the raw category list and the derived maps so that a
    reader can reconstruct everything and cross-check consistency. Returns ``path``.
    """
    categories = categories if categories is not None else config.CUAD_CATEGORIES
    label2id, id2label = build_label_maps(categories)
    payload = {
        "categories": categories,
        "labels": build_bio_labels(categories),
        "label2id": label2id,
        # JSON object keys are strings; keep id2label keys as strings on disk.
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_labels(path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load (label2id, id2label) from a labels.json written by ``save_labels``.

    id2label keys are coerced back to ints. The loaded maps are validated for
    mutual consistency before being returned.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    label2id = {str(k): int(v) for k, v in payload["label2id"].items()}
    id2label = {int(k): str(v) for k, v in payload["id2label"].items()}

    if len(label2id) != len(id2label):
        raise ValueError("label2id and id2label have mismatched sizes in labels.json")
    for label, idx in label2id.items():
        if id2label.get(idx) != label:
            raise ValueError(
                f"Inconsistent label maps in labels.json: {label!r} <-> id {idx}"
            )
    return label2id, id2label
