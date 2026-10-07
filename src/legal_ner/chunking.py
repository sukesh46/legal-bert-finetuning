"""Sliding-window chunking of long contracts (spec section 5).

CUAD contracts routinely exceed Legal-BERT's 512-token limit, so each contract row is
split into overlapping windows BEFORE subword tokenization, with BIO tags travelling
with their words into every window.

Window units: the spec specifies window size 384 / stride 128 in TOKEN units. Exact
token counts require the subword tokenizer, which we deliberately keep out of this
module so chunking stays pure and testable. Instead we chunk at the WORD level using a
word budget derived from WINDOW_SIZE, discounted by an estimated subword-per-word
inflation factor so a window's words are very unlikely to exceed the token budget after
subword tokenization. The downstream tokenizer still applies truncation at
max_length=WINDOW_SIZE as a hard safety net (section 6), so a rare under-estimate
cannot produce an over-length sequence; it would only cost a little overlap.

Each emitted chunk carries (contract_id, chunk_index) for traceability and inference-time
merge (section 5 dedup rule, implemented in inference.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config

# Legal/English subword inflation: BERT WordPiece averages well under 1.6 subwords per
# whitespace word on legal text, plus 2 special tokens ([CLS]/[SEP]). We budget
# conservatively so a window rarely overflows WINDOW_SIZE tokens after tokenization.
SUBWORD_PER_WORD = 1.6
SPECIAL_TOKENS = 2


def word_budget(window_size: int = config.WINDOW_SIZE) -> int:
    """Max words per window such that subword tokenization stays within window_size."""
    return max(1, int((window_size - SPECIAL_TOKENS) / SUBWORD_PER_WORD))


def word_stride(stride: int = config.STRIDE) -> int:
    """Window step in words, derived from the token stride by the same discount."""
    return max(1, int(stride / SUBWORD_PER_WORD))


@dataclass
class Chunk:
    contract_id: str
    chunk_index: int
    words: list[str]
    ner_tags: list[str]


def chunk_row(
    row: dict,
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> list[Chunk]:
    """Split a single {contract_id, words, ner_tags} row into overlapping word windows.

    - Windows never split a word (we operate on whole words).
    - BIO tags travel with their words.
    - Consecutive windows overlap by (word_budget - word_stride) words.
    - A contract shorter than one window yields exactly one chunk.
    """
    words = row["words"]
    tags = row["ner_tags"]
    assert len(words) == len(tags), "words and ner_tags must be the same length"

    size = word_budget(window_size)
    step = word_stride(stride)
    assert 0 < step <= size, "word stride must be positive and not exceed the window"

    contract_id = row["contract_id"]
    n = len(words)
    chunks: list[Chunk] = []

    if n == 0:
        return [Chunk(contract_id, 0, [], [])]

    idx = 0
    start = 0
    while start < n:
        end = min(start + size, n)
        chunks.append(Chunk(contract_id, idx, words[start:end], tags[start:end]))
        idx += 1
        if end == n:
            break
        start += step
    return chunks


def chunk_rows(
    rows: list[dict],
    window_size: int = config.WINDOW_SIZE,
    stride: int = config.STRIDE,
) -> list[dict]:
    """Chunk many rows, returning plain dicts ready for a datasets.Dataset.

    Output rows: {contract_id, chunk_index, words, ner_tags}.
    """
    out: list[dict] = []
    for row in rows:
        for ch in chunk_row(row, window_size, stride):
            out.append(
                {
                    "contract_id": ch.contract_id,
                    "chunk_index": ch.chunk_index,
                    "words": ch.words,
                    "ner_tags": ch.ner_tags,
                }
            )
    return out
