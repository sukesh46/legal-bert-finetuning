"""Tests for subword tokenization + label alignment (spec section 6).

The alignment logic is isolated into align_labels_for_example(word_ids, ...) so it is
testable without the real tokenizer. tokenize_and_align_labels is exercised with a tiny
fake tokenizer that mimics the Hugging Face interface (word_ids + is_split_into_words),
so these tests need no ML dependency and run anywhere.
"""

from legal_ner import tokenize_align as ta
from legal_ner import labels as lbl

IGNORE = ta.IGNORE_INDEX


def test_special_tokens_masked():
    label2id = {"O": 0, "B-X": 1, "I-X": 2}
    # [CLS] w0 [SEP]
    word_ids = [None, 0, None]
    tags = ["B-X"]
    out = ta.align_labels_for_example(word_ids, tags, label2id)
    assert out == [IGNORE, 1, IGNORE]


def test_only_first_subword_labeled():
    label2id = {"O": 0, "B-X": 1, "I-X": 2}
    # word 0 split into 3 subwords, word 1 into 1 subword
    word_ids = [None, 0, 0, 0, 1, None]
    tags = ["B-X", "I-X"]
    out = ta.align_labels_for_example(word_ids, tags, label2id)
    #        [CLS]  B-X   cont  cont  I-X   [SEP]
    assert out == [IGNORE, 1, IGNORE, IGNORE, 2, IGNORE]


def test_alignment_uses_real_label_schema():
    label2id, _ = lbl.build_label_maps()
    tags = ["O", "B-Governing Law", "I-Governing Law"]
    word_ids = [None, 0, 1, 1, 2, None]
    out = ta.align_labels_for_example(word_ids, tags, label2id)
    assert out[0] == IGNORE and out[-1] == IGNORE
    assert out[1] == label2id["O"]
    assert out[2] == label2id["B-Governing Law"]
    assert out[3] == IGNORE  # continuation subword of word 1
    assert out[4] == label2id["I-Governing Law"]


class _FakeEncoding(dict):
    """Minimal stand-in for a BatchEncoding: a dict plus word_ids(batch_index)."""

    def __init__(self, input_ids, word_ids_per_example):
        super().__init__(input_ids=input_ids)
        self._word_ids = word_ids_per_example

    def word_ids(self, batch_index=0):
        return self._word_ids[batch_index]


class _FakeTokenizer:
    """Splits each word into len(word) chars as 'subwords', wraps with None specials."""

    def __call__(self, batch_words, is_split_into_words=True, truncation=True, max_length=384):
        assert is_split_into_words
        input_ids = []
        word_ids_per_example = []
        for words in batch_words:
            wids = [None]
            ids = [101]  # fake [CLS]
            for wi, word in enumerate(words):
                for _ in range(max(1, len(word))):
                    wids.append(wi)
                    ids.append(1000 + wi)
            wids.append(None)
            ids.append(102)  # fake [SEP]
            # Apply truncation like the real tokenizer would.
            ids = ids[:max_length]
            wids = wids[:max_length]
            input_ids.append(ids)
            word_ids_per_example.append(wids)
        return _FakeEncoding(input_ids, word_ids_per_example)


def test_tokenize_and_align_batch():
    label2id, _ = lbl.build_label_maps()
    examples = {
        "words": [["This", "Agreement"], ["Delaware"]],
        "ner_tags": [["O", "B-Governing Law"], ["B-Parties"]],
    }
    out = ta.tokenize_and_align_labels(examples, _FakeTokenizer(), label2id, max_length=50)
    assert "labels" in out
    assert len(out["labels"]) == 2

    # Example 0: [CLS] This(4 subwords) Agreement(9 subwords) [SEP]
    labels0 = out["labels"][0]
    assert labels0[0] == IGNORE                      # [CLS]
    assert labels0[1] == label2id["O"]               # first subword of "This"
    assert all(x == IGNORE for x in labels0[2:5])    # continuation of "This"
    assert labels0[5] == label2id["B-Governing Law"] # first subword of "Agreement"
    assert labels0[-1] == IGNORE                      # [SEP]


def test_truncation_respected():
    label2id, _ = lbl.build_label_maps()
    examples = {
        "words": [["word"] * 100],
        "ner_tags": [["O"] * 100],
    }
    out = ta.tokenize_and_align_labels(examples, _FakeTokenizer(), label2id, max_length=10)
    assert len(out["labels"][0]) == 10
