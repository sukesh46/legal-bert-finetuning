"""Tests for the pure (model-free) parts of inference (spec section 9)."""

from legal_ner import cuad_to_bio as c2b
from legal_ner import inference as inf


def test_subword_preds_first_subword_rule():
    # [CLS] w0 w0 w0 w1 [SEP]
    word_ids = [None, 0, 0, 0, 1, None]
    preds = ["O", "B-Governing Law", "X", "Y", "B-Parties", "O"]
    out = inf._subword_preds_to_word_tags(word_ids, preds)
    assert out == {0: "B-Governing Law", 1: "B-Parties"}


def test_merge_prefers_non_o_and_earliest():
    # word 2 is "O" in window 0 but tagged in window 1 -> keep the tag.
    specs = [
        {"start": 0, "tags": ["B-X", "I-X", "O"]},
        {"start": 2, "tags": ["B-Y", "O"]},
    ]
    merged = inf.merge_window_word_predictions(specs, num_words=4)
    assert merged == ["B-X", "I-X", "B-Y", "O"]


def test_merge_earliest_non_o_wins():
    # Both windows tag word 1; the earliest non-O assignment sticks.
    specs = [
        {"start": 0, "tags": ["O", "B-X"]},
        {"start": 1, "tags": ["B-Y", "O"]},
    ]
    merged = inf.merge_window_word_predictions(specs, num_words=3)
    assert merged[1] == "B-X"


def test_spans_carry_original_char_offsets():
    text = "This Agreement is governed by Delaware law today."
    words = c2b.tokenize_with_offsets(text)
    # Tag "governed by Delaware" as Governing Law.
    tags = ["O"] * len(words)
    # Indices: This(0) Agreement(1) is(2) governed(3) by(4) Delaware(5) law(6) today.(7)
    tags[3] = "B-Governing Law"
    tags[4] = "I-Governing Law"
    tags[5] = "I-Governing Law"

    spans = inf.spans_from_word_tags(words, tags)
    assert len(spans) == 1
    span = spans[0]
    assert span.category == "Governing Law"
    assert span.text == "governed by Delaware"
    # The offsets must slice the ORIGINAL text back to the span surface form.
    assert text[span.start_char:span.end_char] == "governed by Delaware"


def test_spans_split_on_category_change_without_o():
    text = "Acme Delaware"
    words = c2b.tokenize_with_offsets(text)
    tags = ["B-Parties", "B-Governing Law"]
    spans = inf.spans_from_word_tags(words, tags)
    assert [(s.category, s.text) for s in spans] == [
        ("Parties", "Acme"),
        ("Governing Law", "Delaware"),
    ]


def test_no_spans_when_all_o():
    text = "nothing to see here"
    words = c2b.tokenize_with_offsets(text)
    tags = ["O"] * len(words)
    assert inf.spans_from_word_tags(words, tags) == []
