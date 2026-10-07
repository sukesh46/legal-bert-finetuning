"""Tests for BIO label-schema construction and labels.json persistence."""

import json

from legal_ner import config, labels


def test_83_labels_total():
    assert labels.num_labels() == 83
    assert len(labels.build_bio_labels()) == 83


def test_label_list_structure():
    bio = labels.build_bio_labels()
    assert bio[0] == "O"
    # First category's B- then I- immediately follow "O".
    first = config.CUAD_CATEGORIES[0]
    assert bio[1] == f"B-{first}"
    assert bio[2] == f"I-{first}"


def test_label_maps_round_trip_in_memory():
    label2id, id2label = labels.build_label_maps()
    assert len(label2id) == 83
    for label, idx in label2id.items():
        assert id2label[idx] == label


def test_every_category_has_b_and_i():
    bio = set(labels.build_bio_labels())
    for cat in config.CUAD_CATEGORIES:
        assert f"B-{cat}" in bio
        assert f"I-{cat}" in bio


def test_save_and_load_labels_json(tmp_path):
    path = tmp_path / "labels.json"
    labels.save_labels(str(path))

    # File is valid JSON with the expected top-level keys.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"categories", "labels", "label2id", "id2label"}
    assert len(payload["categories"]) == 41

    # Round-trips back to consistent maps.
    label2id, id2label = labels.load_labels(str(path))
    assert len(label2id) == 83
    assert label2id["O"] == 0
    for label, idx in label2id.items():
        assert id2label[idx] == label


def test_load_rejects_inconsistent_maps(tmp_path):
    path = tmp_path / "bad_labels.json"
    bad = {
        "categories": ["Governing Law"],
        "labels": ["O", "B-Governing Law", "I-Governing Law"],
        "label2id": {"O": 0, "B-Governing Law": 1, "I-Governing Law": 2},
        # Corrupt: id 2 maps to the wrong label.
        "id2label": {"0": "O", "1": "B-Governing Law", "2": "O"},
    }
    path.write_text(json.dumps(bad), encoding="utf-8")

    try:
        labels.load_labels(str(path))
    except ValueError:
        return
    raise AssertionError("Expected ValueError on inconsistent label maps")
