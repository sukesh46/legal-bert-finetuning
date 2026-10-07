"""Entity-level evaluation via seqeval (spec section 8).

Produces aggregate precision/recall/F1 plus a per-category breakdown sorted worst-first,
and flags categories with too little support to be statistically meaningful.

ML/seqeval imports are done lazily inside functions so importing this module (e.g. to
reuse the pure helpers) does not require the heavy stack to be installed.
"""

from __future__ import annotations

from collections import Counter

from . import config


def decode_predictions(predictions, label_ids, id2label):
    """Convert raw logits/pred ids + gold label ids into seqeval string sequences.

    Drops positions whose gold label is the ignore index (-100), i.e. continuation
    subwords and special tokens. Returns (true_labels, pred_labels) as lists of lists.

    ``predictions`` may be logits (…, num_labels); we argmax over the last axis.
    """
    import numpy as np

    predictions = np.asarray(predictions)
    label_ids = np.asarray(label_ids)
    # predictions are logits with a trailing num_labels axis when they have one more
    # dimension than the gold labels; collapse that axis to class ids.
    if predictions.ndim == label_ids.ndim + 1:
        predictions = predictions.argmax(axis=-1)

    true_labels, pred_labels = [], []
    for pred_row, gold_row in zip(predictions, label_ids):
        t, p = [], []
        for pred_id, gold_id in zip(pred_row, gold_row):
            if gold_id == -100:
                continue
            t.append(id2label[int(gold_id)])
            p.append(id2label[int(pred_id)])
        true_labels.append(t)
        pred_labels.append(p)
    return true_labels, pred_labels


def make_compute_metrics(id2label):
    """Return a Trainer-compatible compute_metrics(eval_pred) closure.

    Reports aggregate precision/recall/f1/accuracy via seqeval.
    """
    from seqeval.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred):
        predictions, label_ids = eval_pred
        true_labels, pred_labels = decode_predictions(predictions, label_ids, id2label)
        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

    return compute_metrics


def training_support(train_dataset, id2label) -> Counter:
    """Count B-<category> occurrences per category in the training split.

    Used to flag low-support categories (section 8). Counts entity *starts* (B- tags),
    which approximates the number of training spans per category.
    """
    support: Counter = Counter()
    for labels_row in train_dataset["labels"]:
        for lab in labels_row:
            if lab == -100:
                continue
            name = id2label[int(lab)]
            if name.startswith("B-"):
                support[name[2:]] += 1
    return support


def per_category_report(
    true_labels,
    pred_labels,
    support: Counter | None = None,
    min_support: int = config.MIN_CATEGORY_SUPPORT,
) -> str:
    """Build a per-category seqeval report sorted worst-F1 first (section 8).

    Categories whose training support is below ``min_support`` are annotated as
    statistically unreliable rather than silently appearing as 0.0/1.0 outliers.
    """
    from seqeval.metrics import classification_report

    report = classification_report(
        true_labels, pred_labels, output_dict=True, zero_division=0
    )

    rows = []
    for category, metrics in report.items():
        if category in {"micro avg", "macro avg", "weighted avg"}:
            continue
        f1 = metrics.get("f1-score", 0.0)
        sup = support.get(category, 0) if support is not None else metrics.get("support", 0)
        low = support is not None and sup < min_support
        rows.append((f1, category, metrics, sup, low))

    rows.sort(key=lambda r: r[0])  # worst F1 first

    lines = [f"{'category':<40} {'P':>6} {'R':>6} {'F1':>6} {'support':>8}  note"]
    for f1, category, metrics, sup, low in rows:
        note = "LOW SUPPORT — not statistically meaningful" if low else ""
        lines.append(
            f"{category:<40} "
            f"{metrics.get('precision', 0.0):>6.3f} "
            f"{metrics.get('recall', 0.0):>6.3f} "
            f"{f1:>6.3f} "
            f"{sup:>8}  {note}"
        )
    return "\n".join(lines)
