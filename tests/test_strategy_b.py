"""Tests for the deterministic Strategy B (multi-label) pieces.

Covers: multi-hot label encoding, per-category metric counts, and multi-label span
reconstruction with overlaps preserved. Model forward/training is NOT tested here (needs
torch + GPU); those are verified on Colab. numpy-backed tests importorskip numpy.
"""

import pytest

from legal_ner import config
from legal_ner import tokenize_align as ta
from legal_ner import inference as inf
from legal_ner import cuad_to_bio as c2b

CATS = ["Governing Law", "Parties", "License Grant"]


# --------------------------------------------------------------------------- #
# Multi-hot label encoding
# --------------------------------------------------------------------------- #
def test_presence_vector_o_is_all_zero():
    assert ta.category_presence_vector("O", CATS) == [0.0, 0.0, 0.0]


def test_presence_vector_marks_correct_category():
    assert ta.category_presence_vector("B-Parties", CATS) == [0.0, 1.0, 0.0]
    assert ta.category_presence_vector("I-License Grant", CATS) == [0.0, 0.0, 1.0]


def test_align_multilabels_masks_specials_and_continuations():
    # [CLS] w0 w0 w1 [SEP]  -> word0 "B-Governing Law", word1 "O"
    word_ids = [None, 0, 0, 1, None]
    tags = ["B-Governing Law", "O"]
    out = ta.align_multilabels_for_example(word_ids, tags, CATS)
    assert out[0] == [-100.0, -100.0, -100.0]      # [CLS]
    assert out[1] == [1.0, 0.0, 0.0]               # first subword of word0
    assert out[2] == [-100.0, -100.0, -100.0]      # continuation subword
    assert out[3] == [0.0, 0.0, 0.0]               # word1 "O"
    assert out[4] == [-100.0, -100.0, -100.0]      # [SEP]


def test_align_multilabels_row_width_matches_categories():
    word_ids = [None, 0, None]
    out = ta.align_multilabels_for_example(word_ids, ["B-Parties"], CATS)
    assert all(len(row) == len(CATS) for row in out)


# --------------------------------------------------------------------------- #
# Per-category metric counts (numpy)
# --------------------------------------------------------------------------- #
def test_multilabel_counts_tp_fp_fn():
    np = pytest.importorskip("numpy")
    from legal_ner import evaluate as ev

    # 3 tokens, 2 categories. Use large +/- logits so threshold 0.5 is unambiguous.
    BIG = 10.0
    logits = np.array([
        [BIG, -BIG],    # pred cat0
        [-BIG, BIG],    # pred cat1
        [-BIG, -BIG],   # ignored row below
    ])
    labels = np.array([
        [1.0, 0.0],         # gold cat0  -> TP cat0
        [1.0, 1.0],         # gold both  -> cat1 TP, cat0 FN
        [-100.0, -100.0],   # ignored
    ])
    tp, fp, fn = ev.multilabel_counts(logits, labels, threshold=0.5)
    assert list(tp) == [1.0, 1.0]
    assert list(fp) == [0.0, 0.0]
    assert list(fn) == [1.0, 0.0]  # cat0 missed on token 1


def test_multilabel_micro_f1_closure():
    pytest.importorskip("numpy")
    import numpy as np
    from legal_ner import evaluate as ev

    cm = ev.make_compute_metrics_multilabel(["a", "b"], threshold=0.5)
    BIG = 10.0
    logits = np.array([[[BIG, -BIG], [-BIG, BIG]]])      # (1, 2, 2)
    labels = np.array([[[1.0, 0.0], [0.0, 1.0]]])        # perfect
    m = cm((logits, labels))
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


# --------------------------------------------------------------------------- #
# Multi-label span reconstruction (overlaps preserved)
# --------------------------------------------------------------------------- #
def test_overlapping_spans_both_recovered():
    text = "Governed Delaware license grant"
    words = c2b.tokenize_with_offsets(text)
    # word0 belongs to BOTH Governing Law and License Grant (the whole point of B).
    word_cats = [
        {"Governing Law", "License Grant"},
        {"Governing Law"},
        {"License Grant"},
        {"License Grant"},
    ]
    spans = inf.spans_from_word_category_sets(words, word_cats)
    by_cat = {}
    for s in spans:
        by_cat.setdefault(s.category, []).append(s)

    # Governing Law spans words 0-1; License Grant spans word 0 then words 2-3.
    gl = by_cat["Governing Law"]
    assert len(gl) == 1 and gl[0].text == "Governed Delaware"
    assert text[gl[0].start_char:gl[0].end_char] == "Governed Delaware"

    lg = sorted(by_cat["License Grant"], key=lambda s: s.start_char)
    assert [s.text for s in lg] == ["Governed", "license grant"]


def test_no_categories_no_spans():
    text = "nothing here"
    words = c2b.tokenize_with_offsets(text)
    assert inf.spans_from_word_category_sets(words, [set(), set()]) == []


def test_merge_window_category_sets_unions():
    specs = [
        {"start": 0, "cat_sets": [{"A"}, {"B"}]},
        {"start": 1, "cat_sets": [{"C"}, set()]},  # word1 gets B (win0) + C (win1)
    ]
    merged = inf.merge_window_category_sets(specs, num_words=3)
    assert merged[0] == {"A"}
    assert merged[1] == {"B", "C"}
    assert merged[2] == set()


def test_config_strategy_b_defaults():
    assert config.STRATEGY in {"A", "B"}
    assert config.NUM_TRAIN_EPOCHS == 3
    assert 0.0 < config.STRATEGY_B_THRESHOLD < 1.0
