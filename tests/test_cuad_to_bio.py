"""Tests for the CUAD -> BIO conversion core (spec section 4).

The span-reconstruction test (section 4.3.5) is the single most important correctness
check in the pipeline, so it gets the most attention here. All tests run against a tiny
synthetic fixture whose answer_start offsets are verified to slice to their text.
"""

import json
import os

import pytest

from legal_ner import config, cuad_to_bio as c2b

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_cuad.json")


@pytest.fixture
def cuad():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Offset-aware tokenizer
# --------------------------------------------------------------------------- #
def test_tokenize_offsets_slice_back_exactly():
    ctx = "This Agreement  shall be\tgoverned."
    words = c2b.tokenize_with_offsets(ctx)
    for w in words:
        assert ctx[w.start:w.end] == w.text
    # Collapses all whitespace kinds, keeps punctuation attached to the word.
    assert [w.text for w in words] == ["This", "Agreement", "shall", "be", "governed."]


# --------------------------------------------------------------------------- #
# Category extraction
# --------------------------------------------------------------------------- #
def test_extract_category_from_quotes():
    q = 'Highlight the parts related to "Governing Law" that should be reviewed.'
    assert c2b.extract_category(q) == "Governing Law"


def test_extract_category_longest_match_not_shadowed():
    q = 'related to "Non-Transferable License" here'
    assert c2b.extract_category(q) == "Non-Transferable License"


def test_extract_category_unknown_returns_none():
    assert c2b.extract_category('related to "Teleportation Clause"') is None


# --------------------------------------------------------------------------- #
# Tagging + the all-important reconstruction
# --------------------------------------------------------------------------- #
def test_tag_contract_basic_bio_shape(cuad):
    para = cuad["data"][0]["paragraphs"][0]
    words, tags, _ = c2b.tag_contract(para["context"], para["qas"])
    assert len(words) == len(tags)
    # First tagged token of a span is B-, continuation is I-.
    gl_tags = [t for t in tags if t.endswith("Governing Law")]
    assert gl_tags[0] == "B-Governing Law"
    assert all(t == "I-Governing Law" for t in gl_tags[1:])


def test_impossible_qas_is_skipped(cuad):
    para = cuad["data"][0]["paragraphs"][0]
    _, tags, _ = c2b.tag_contract(para["context"], para["qas"])
    # Non-Compete is is_impossible -> must not appear anywhere.
    assert not any("Non-Compete" in t for t in tags)


def test_reconstruction_matches_answers_exactly(cuad):
    """Section 4.3.5: tagged spans must reconstruct to the original answer text."""
    rows, _ = c2b.convert_cuad(cuad)
    mismatches = c2b.validate_reconstruction(cuad, rows, sample_size=len(rows))
    assert mismatches == [], f"Span reconstruction mismatches: {mismatches}"


def test_reconstruct_spans_roundtrip():
    words = ["This", "Agreement", "is", "governed", "by", "Delaware"]
    tags = ["O", "O", "O", "B-Governing Law", "I-Governing Law", "I-Governing Law"]
    spans = c2b.reconstruct_spans(words, tags)
    assert spans == [("Governing Law", "governed by Delaware")]


# --------------------------------------------------------------------------- #
# Strategy A overlap resolution
# --------------------------------------------------------------------------- #
def test_overlap_priority_higher_rank_wins():
    # "Document Name" (rank 40, lowest priority) vs "Governing Law" (higher priority)
    # overlapping the same token -> Governing Law must win that token.
    context = "Governed Delaware"
    qas = [
        {"question": 'related to "Document Name"', "is_impossible": False,
         "answers": [{"text": "Governed Delaware", "answer_start": 0}]},
        {"question": 'related to "Governing Law"', "is_impossible": False,
         "answers": [{"text": "Governed", "answer_start": 0}]},
    ]
    _, tags, stats = c2b.tag_contract(context, qas)
    assert tags[0] == "B-Governing Law"
    # An override was recorded for audit.
    assert any(k.startswith("Governing Law|") for k in stats)


def test_overlap_lower_priority_does_not_override():
    context = "Governed Delaware"
    qas = [
        {"question": 'related to "Governing Law"', "is_impossible": False,
         "answers": [{"text": "Governed", "answer_start": 0}]},
        {"question": 'related to "Document Name"', "is_impossible": False,
         "answers": [{"text": "Governed", "answer_start": 0}]},
    ]
    _, tags, stats = c2b.tag_contract(context, qas)
    # Governing Law was there first and outranks Document Name -> stays.
    assert tags[0] == "B-Governing Law"
    assert stats == {}


# --------------------------------------------------------------------------- #
# Contract-level split (section 4.4) — no leakage
# --------------------------------------------------------------------------- #
def test_contract_level_split_no_leakage():
    # Build many synthetic rows across several contracts.
    rows = []
    for cid in range(10):
        for chunk in range(3):
            rows.append({"contract_id": f"C{cid}", "words": ["x"], "ner_tags": ["O"]})
    splits = c2b.contract_level_split(rows, seed=0, ratios=(0.6, 0.2, 0.2))

    def ids(key):
        return {r["contract_id"] for r in splits[key]}

    train, val, test = ids("train"), ids("val"), ids("test")
    # No contract id appears in more than one split.
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # Every contract landed somewhere.
    assert train | val | test == {f"C{i}" for i in range(10)}


def test_split_is_deterministic_under_seed():
    rows = [{"contract_id": f"C{i}", "words": ["x"], "ner_tags": ["O"]} for i in range(20)]
    a = c2b.contract_level_split(rows, seed=42)
    b = c2b.contract_level_split(rows, seed=42)
    assert {r["contract_id"] for r in a["train"]} == {r["contract_id"] for r in b["train"]}


# --------------------------------------------------------------------------- #
# convert_cuad plumbing
# --------------------------------------------------------------------------- #
def test_convert_cuad_counts(cuad):
    rows, stats = c2b.convert_cuad(cuad)
    assert len(rows) == 2  # two contracts, one paragraph each
    assert stats.contracts == 2
    assert stats.impossible_skipped == 1  # the Non-Compete entry
    assert stats.qa_pairs_used == 4


def test_convert_cuad_limit(cuad):
    rows, stats = c2b.convert_cuad(cuad, limit=1)
    assert len(rows) == 1
    assert stats.contracts == 1
