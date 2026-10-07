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
