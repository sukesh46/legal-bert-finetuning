"""Subword tokenization + BIO label alignment (spec section 6).

Generalises the standard CoNLL alignment convention to the 83-label CUAD schema:
only the FIRST subword of each word receives the word's label; continuation subwords
and special tokens receive -100 (the ignore index for the loss).

max_length is bound to config.WINDOW_SIZE so chunking (section 5) and tokenization stay
in sync from a single constant.

This module imports no ML libraries at module load; the tokenizer is passed in by the
caller, keeping the alignment logic itself unit-testable with a lightweight fake.
"""

from __future__ import annotations

from . import config

IGNORE_INDEX = -100


def align_labels_for_example(
    word_ids: list[int | None],
    tags: list[str],
    label2id: dict[str, int],
) -> list[int]:
    """Align one example's word-level BIO tags to subword tokens.

    Rules:
      * special tokens (word_id is None) -> IGNORE_INDEX
      * first subword of a word          -> label2id[tag]
      * continuation subwords            -> IGNORE_INDEX
    """
    label_ids: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            label_ids.append(label2id[tags[word_id]])
        else:
            label_ids.append(IGNORE_INDEX)
        previous_word_id = word_id
    return label_ids


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length: int | None = None):
    """Batch tokenize pre-split words and align labels (Hugging Face map-compatible).

    ``examples`` is a dict of columns with "words" and "ner_tags" lists. Returns the
    tokenizer output augmented with a "labels" column of aligned label ids.
    """
    max_length = max_length if max_length is not None else config.WINDOW_SIZE
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        all_labels.append(align_labels_for_example(word_ids, tags, label2id))
    tokenized["labels"] = all_labels
    return tokenized


# --------------------------------------------------------------------------- #
# Strategy B: multi-label (presence per category) alignment
# --------------------------------------------------------------------------- #
def category_presence_vector(tag: str, categories: list[str]) -> list[float]:
    """One-hot-ish presence vector for a single word's BIO tag under Strategy B.

    A word's ner_tag is still a single BIO string (produced upstream per category pass),
    but Strategy B represents it as a length-``len(categories)`` 0/1 vector marking which
    category the word belongs to. "O" -> all zeros. B-/I- -> a 1 at that category.

    Note: this encodes presence of ONE category per word (the row shape the Strategy A
    converter emits). Genuine multi-category overlap is realised across the dataset: the
    model learns independent per-category heads, and at train time the BCE target for a
    token is the union of category presences the data provides for it.
    """
    vec = [0.0] * len(categories)
    if tag and tag != "O":
        cat = tag[2:] if tag[:2] in ("B-", "I-") else tag
        try:
            vec[categories.index(cat)] = 1.0
        except ValueError:
            pass
    return vec


def align_multilabels_for_example(
    word_ids: list[int | None],
    tags: list[str],
    categories: list[str],
) -> list[list[float]]:
    """Align word-level tags to subword tokens as per-category presence vectors.

    Mirrors align_labels_for_example's masking rules, but emits a float vector per token:
      * special tokens (word_id None) and continuation subwords -> all -100.0 (ignored
        by the masked BCE loss)
      * first subword of a word -> that word's category-presence vector
    """
    n = len(categories)
    ignore_row = [IGNORE_INDEX * 1.0] * n
    out: list[list[float]] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            out.append(list(ignore_row))
        else:
            out.append(category_presence_vector(tags[word_id], categories))
        previous_word_id = word_id
    return out


def tokenize_and_align_multilabels(
    examples, tokenizer, categories, max_length: int | None = None
):
    """Strategy B batch tokenize + multi-hot align (Hugging Face map-compatible).

    Returns tokenizer output plus a "labels" column of shape [seq_len, len(categories)]
    per example, where ignored positions are rows of -100.0.
    """
    max_length = max_length if max_length is not None else config.WINDOW_SIZE
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        all_labels.append(align_multilabels_for_example(word_ids, tags, categories))
    tokenized["labels"] = all_labels
    return tokenized
