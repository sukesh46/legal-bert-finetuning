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
