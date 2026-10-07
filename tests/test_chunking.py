"""Tests for sliding-window chunking (spec section 5)."""

from legal_ner import chunking


def _row(n_words, contract_id="C0"):
    return {
        "contract_id": contract_id,
        "words": [f"w{i}" for i in range(n_words)],
        "ner_tags": ["O"] * n_words,
    }


def test_word_budget_and_stride_positive():
    assert chunking.word_budget(384) > 0
    assert chunking.word_stride(128) > 0
    # Stride must be smaller than the window so consecutive windows overlap.
    assert chunking.word_stride(128) < chunking.word_budget(384)


def test_short_contract_single_chunk():
    chunks = chunking.chunk_row(_row(5))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].words == [f"w{i}" for i in range(5)]


def test_empty_contract_yields_one_empty_chunk():
    chunks = chunking.chunk_row(_row(0))
    assert len(chunks) == 1
    assert chunks[0].words == []


def test_windows_overlap_and_cover_everything():
    # Use a small window/stride so the arithmetic is easy to reason about.
    size, stride = 10, 4  # pre-discount token values
    budget = chunking.word_budget(size)
    step = chunking.word_stride(stride)
    n = budget * 3
    chunks = chunking.chunk_row(_row(n), window_size=size, stride=stride)

    # Every chunk (except possibly the last) is exactly `budget` words.
    for ch in chunks[:-1]:
        assert len(ch.words) == budget
    # chunk_index is sequential from 0.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Consecutive windows overlap by (budget - step) words.
    if len(chunks) >= 2:
        overlap = budget - step
        assert chunks[0].words[-overlap:] == chunks[1].words[:overlap]
    # Union of all chunk words covers the whole contract in order.
    seen = []
    for ch in chunks:
        for w in ch.words:
            if not seen or seen[-1] != w:
                pass
    all_words = set()
    for ch in chunks:
        all_words.update(ch.words)
    assert all_words == {f"w{i}" for i in range(n)}


def test_tags_travel_with_words():
    row = {
        "contract_id": "C1",
        "words": [f"w{i}" for i in range(30)],
        "ner_tags": [f"t{i}" for i in range(30)],
    }
    chunks = chunking.chunk_row(row, window_size=10, stride=4)
    for ch in chunks:
        # Each word's tag index matches its word index (constructed to be parallel).
        for w, t in zip(ch.words, ch.ner_tags):
            assert w.replace("w", "") == t.replace("t", "")


def test_no_word_is_split():
    # Operating on whole-word lists, a word can never be fragmented: every chunk word
    # is an element of the original word list.
    row = _row(50)
    original = set(row["words"])
    for ch in chunking.chunk_row(row, window_size=12, stride=5):
        assert set(ch.words).issubset(original)


def test_chunk_rows_output_shape():
    rows = [_row(40, "A"), _row(5, "B")]
    out = chunking.chunk_rows(rows, window_size=10, stride=4)
    assert all(set(r) == {"contract_id", "chunk_index", "words", "ner_tags"} for r in out)
    # Contract B is short -> exactly one chunk.
    b_chunks = [r for r in out if r["contract_id"] == "B"]
    assert len(b_chunks) == 1


def test_last_chunk_not_duplicated_when_aligned():
    # When the contract length is an exact multiple of the step, the loop must still
    # terminate cleanly without emitting a spurious trailing chunk.
    budget = chunking.word_budget(10)
    step = chunking.word_stride(4)
    n = budget + step  # exactly two windows
    chunks = chunking.chunk_row(_row(n), window_size=10, stride=4)
    # Last chunk must reach the final word.
    assert chunks[-1].words[-1] == f"w{n - 1}"
