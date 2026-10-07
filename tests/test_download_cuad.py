"""Tests for CUAD validation + stats (network-free, against the tiny fixture)."""

import json
import os

import pytest

from legal_ner import download_cuad as dl

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tiny_cuad.json")


@pytest.fixture
def cuad():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def test_validate_accepts_well_formed(cuad):
    dl.validate_cuad(cuad)  # must not raise


def test_validate_rejects_missing_data_key():
    with pytest.raises(ValueError):
        dl.validate_cuad({"nope": []})


def test_validate_rejects_empty_data():
    with pytest.raises(ValueError):
        dl.validate_cuad({"data": []})


def test_validate_rejects_missing_paragraphs():
    with pytest.raises(ValueError):
        dl.validate_cuad({"data": [{"title": "x"}]})


def test_stats_counts(cuad):
    stats = dl.cuad_stats(cuad)
    assert stats.contracts == 2
    assert stats.qa_pairs == 5          # 3 in Alpha + 2 in Beta
    assert stats.impossible == 1        # the Non-Compete entry
    # Categories detected: Governing Law, Parties, Non-Compete, Audit Rights,
    # Source Code Escrow = 5 distinct.
    assert stats.categories == 5


def test_load_cuad_json_round_trip(tmp_path, cuad):
    path = tmp_path / "CUAD_v1.json"
    path.write_text(json.dumps(cuad), encoding="utf-8")
    loaded = dl.load_cuad_json(str(path))
    assert loaded["data"][0]["title"] == "AlphaContract"


def test_format_stats_mentions_expected_41(cuad):
    msg = dl.format_stats(dl.cuad_stats(cuad))
    assert "expected 41" in msg


# --------------------------------------------------------------------------- #
# Hugging Face flat -> nested adaptation
# --------------------------------------------------------------------------- #
def _hf_rows():
    # Flat HF shape: answers is a dict of parallel lists; empty text => impossible.
    return [
        {"title": "AlphaContract", "id": "a1",
         "context": "This Agreement shall be governed by the laws of Delaware.",
         "question": 'parts related to "Governing Law" ...',
         "answers": {"text": ["shall be governed by the laws of Delaware"], "answer_start": [15]}},
        {"title": "AlphaContract", "id": "a2",
         "context": "This Agreement shall be governed by the laws of Delaware.",
         "question": 'parts related to "Non-Compete" ...',
         "answers": {"text": [], "answer_start": []}},  # impossible
        {"title": "BetaContract", "id": "b1",
         "context": "Source Code Escrow applies here.",
         "question": 'parts related to "Source Code Escrow" ...',
         "answers": {"text": ["Source Code Escrow"], "answer_start": [0]}},
    ]


def test_hf_rows_to_cuad_groups_by_title():
    cuad = dl.hf_rows_to_cuad(_hf_rows())
    titles = [e["title"] for e in cuad["data"]]
    assert titles == ["AlphaContract", "BetaContract"]
    # Alpha has two qas, one of them impossible.
    alpha = cuad["data"][0]["paragraphs"][0]["qas"]
    assert len(alpha) == 2
    assert alpha[1]["is_impossible"] is True
    assert alpha[0]["is_impossible"] is False
    assert alpha[0]["answers"][0] == {"text": "shall be governed by the laws of Delaware",
                                       "answer_start": 15}


def test_hf_adapted_shape_validates_and_counts():
    cuad = dl.hf_rows_to_cuad(_hf_rows())
    dl.validate_cuad(cuad)  # must not raise
    stats = dl.cuad_stats(cuad)
    assert stats.contracts == 2
    assert stats.qa_pairs == 3
    assert stats.impossible == 1


def test_hf_adapted_feeds_converter():
    # End-to-end: HF adaptation -> convert_cuad -> reconstruction passes.
    from legal_ner import cuad_to_bio as c2b
    cuad = dl.hf_rows_to_cuad(_hf_rows())
    rows, _ = c2b.convert_cuad(cuad)
    assert c2b.validate_reconstruction(cuad, rows, sample_size=len(rows)) == []
