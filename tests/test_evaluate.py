"""Tests for the model-free parts of evaluate (spec section 8).

decode_predictions needs numpy (installed in the dev env); training_support is pure
Python. seqeval-backed functions (make_compute_metrics, per_category_report) are
exercised only when seqeval is importable, and skipped otherwise.
"""

import pytest

from legal_ner import evaluate as ev
from legal_ner import labels as lbl

np = pytest.importorskip("numpy")


def test_decode_predictions_argmaxes_logits_and_drops_ignore():
    label2id, id2label = lbl.build_label_maps()
    o = label2id["O"]
    b_gl = label2id["B-Governing Law"]

    # Batch of 1, seq len 4, with a -100 at position 2 (continuation subword).
    # logits shaped (1, 4, num_labels): make the argmax land on the intended ids.
    num = lbl.num_labels()
    logits = np.full((1, 4, num), -5.0)
    logits[0, 0, o] = 10.0
    logits[0, 1, b_gl] = 10.0
    logits[0, 2, o] = 10.0     # will be dropped (gold is -100)
    logits[0, 3, o] = 10.0
    gold = np.array([[o, b_gl, -100, o]])

    true_labels, pred_labels = ev.decode_predictions(logits, gold, id2label)
    assert true_labels == [["O", "B-Governing Law", "O"]]
    assert pred_labels == [["O", "B-Governing Law", "O"]]


def test_decode_predictions_accepts_precomputed_ids():
    label2id, id2label = lbl.build_label_maps()
    o = label2id["O"]
    b_p = label2id["B-Parties"]
    # Already argmaxed ids (ndim == gold ndim) -> no argmax applied.
    preds = np.array([[o, b_p]])
    gold = np.array([[o, b_p]])
    true_labels, pred_labels = ev.decode_predictions(preds, gold, id2label)
    assert true_labels == [["O", "B-Parties"]]
    assert pred_labels == [["O", "B-Parties"]]


def test_training_support_counts_b_tags():
    label2id, id2label = lbl.build_label_maps()
    b_gl = label2id["B-Governing Law"]
    i_gl = label2id["I-Governing Law"]
    b_p = label2id["B-Parties"]
    o = label2id["O"]

    train_ds = {
        "labels": [
            [o, b_gl, i_gl, o, b_p],   # 1 Governing Law start, 1 Parties start
            [b_gl, -100, o],           # 1 more Governing Law start, ignore index skipped
        ]
    }
    support = ev.training_support(train_ds, id2label)
    assert support["Governing Law"] == 2
    assert support["Parties"] == 1


def test_per_category_report_flags_low_support():
    seqeval = pytest.importorskip("seqeval")  # noqa: F841
    from collections import Counter

    true_labels = [["B-Governing Law", "I-Governing Law"], ["B-Parties"]]
    pred_labels = [["B-Governing Law", "I-Governing Law"], ["O"]]
    support = Counter({"Governing Law": 100, "Parties": 2})
    report = ev.per_category_report(true_labels, pred_labels, support=support, min_support=20)
    assert "LOW SUPPORT" in report
    # Worst-F1 category (Parties, missed) should be listed before Governing Law.
    assert report.index("Parties") < report.index("Governing Law")
