"""Inference on raw contract text (spec section 9).

Pipeline:
    raw text
      -> offset-aware whitespace tokenization (reuse cuad_to_bio.tokenize_with_offsets)
      -> word-level chunking (reuse chunking)
      -> per-chunk subword tokenize + predict
      -> map subword predictions back to words (first-subword rule)
      -> merge overlapping-window predictions (dedup rule, section 5)
      -> reconstruct contiguous entity spans
      -> return [{category, text, start_char, end_char}] with offsets into the ORIGINAL
         document (section 9 requirement for downstream citation grounding).

The merge and span-reconstruction logic is factored into pure functions so it is unit
testable without a model. Heavy imports are lazy.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import chunking, config
from . import cuad_to_bio as c2b
from . import labels as lbl


@dataclass
class EntitySpan:
    category: str
    text: str
    start_char: int
    end_char: int


def merge_window_word_predictions(
    chunk_specs: list[dict],
    num_words: int,
) -> list[str]:
    """Merge per-chunk word-level BIO predictions into one tag per original word.

    ``chunk_specs`` is a list of {"start": int, "tags": [str, ...]} where ``start`` is
    the index (into the original word list) of the chunk's first word and ``tags`` are
    the predicted word-level tags for that chunk.

    Dedup rule (section 5): a word covered by multiple overlapping windows prefers a
    non-"O" prediction over "O"; among windows the earliest non-"O" prediction wins
    (windows are processed in order, and once a word has a non-"O" tag a later window
    does not override it). This favours the window where the entity's span is present
    rather than one where it was clipped at the boundary into background.
    """
    merged: list[str] = ["O"] * num_words
    assigned: list[bool] = [False] * num_words

    for spec in chunk_specs:
        start = spec["start"]
        for offset, tag in enumerate(spec["tags"]):
            wi = start + offset
            if wi >= num_words:
                break
            if assigned[wi]:
                continue
            if tag != "O":
                merged[wi] = tag
                assigned[wi] = True
    return merged


def spans_from_word_tags(words, tags) -> list[EntitySpan]:
    """Reconstruct entity spans with ORIGINAL-document char offsets.

    ``words`` are cuad_to_bio.Word objects (carrying start/end char offsets into the
    original text). Contiguous B-/I- runs of the same category become one EntitySpan
    whose offsets span from the first word's start to the last word's end, and whose
    text is sliced semantics (joined surface form).
    """
    spans: list[EntitySpan] = []
    cur_cat: str | None = None
    cur_start_char = 0
    cur_end_char = 0
    cur_words: list[str] = []

    def flush():
        nonlocal cur_cat, cur_words
        if cur_cat is not None and cur_words:
            spans.append(
                EntitySpan(cur_cat, " ".join(cur_words), cur_start_char, cur_end_char)
            )
        cur_cat, cur_words = None, []

    for word, tag in zip(words, tags):
        if tag == "O":
            flush()
            continue
        prefix, _, cat = tag.partition("-")
        if prefix == "B" or cat != cur_cat:
            flush()
            cur_cat = cat
            cur_start_char = word.start
            cur_end_char = word.end
            cur_words = [word.text]
        else:  # I- continuing the same category
            cur_end_char = word.end
            cur_words.append(word.text)
    flush()
    return spans


def _subword_preds_to_word_tags(word_ids, pred_label_names) -> dict[int, str]:
    """Map subword predictions to word-level tags using the first-subword rule."""
    word_tag: dict[int, str] = {}
    previous = None
    for wid, name in zip(word_ids, pred_label_names):
        if wid is None:
            previous = wid
            continue
        if wid != previous:
            word_tag[wid] = name
        previous = wid
    return word_tag


def predict(
    text: str,
    model,
    tokenizer,
    id2label=None,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> list[EntitySpan]:
    """Run the full inference pipeline on a raw contract string.

    ``model``/``tokenizer`` are a loaded token-classification pair. ``id2label`` defaults
    to the canonical schema but should be the one loaded from the checkpoint's
    labels.json in practice.
    """
    import torch

    if id2label is None:
        _, id2label = lbl.build_label_maps()

    words = c2b.tokenize_with_offsets(text)
    word_strs = [w.text for w in words]

    # Chunk at the word level, tracking each chunk's starting word index.
    row = {"contract_id": "inference", "words": word_strs, "ner_tags": ["O"] * len(word_strs)}
    chunks = chunking.chunk_row(row, window_size=window_size, stride=stride)

    step = chunking.word_stride(stride)
    chunk_specs: list[dict] = []
    for chunk in chunks:
        start = chunk.chunk_index * step
        if not chunk.words:
            continue
        enc = tokenizer(
            [chunk.words],
            is_split_into_words=True,
            truncation=True,
            max_length=window_size,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**{k: v for k, v in enc.items()}).logits
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        pred_names = [id2label[int(i)] for i in pred_ids]
        word_ids = enc.word_ids(batch_index=0)
        word_tag = _subword_preds_to_word_tags(word_ids, pred_names)
        tags = [word_tag.get(i, "O") for i in range(len(chunk.words))]
        chunk_specs.append({"start": start, "tags": tags})

    merged_tags = merge_window_word_predictions(chunk_specs, len(words))
    return spans_from_word_tags(words, merged_tags)


def load_for_inference(model_dir: str):
    """Load (model, tokenizer, id2label) from a saved checkpoint directory."""
    import os

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    labels_path = os.path.join(model_dir, config.LABELS_FILENAME)
    if os.path.exists(labels_path):
        _, id2label = lbl.load_labels(labels_path)
    else:
        _, id2label = lbl.build_label_maps()
    model.eval()
    return model, tokenizer, id2label
