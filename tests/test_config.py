"""Tests for the single-source-of-truth config module."""

from legal_ner import config


def test_exactly_41_unique_categories():
    assert len(config.CUAD_CATEGORIES) == 41
    assert len(set(config.CUAD_CATEGORIES)) == 41


def test_window_and_stride_sane():
    assert config.WINDOW_SIZE == 384
    assert 0 < config.STRIDE < config.WINDOW_SIZE


def test_split_ratios_sum_to_one():
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-9


def test_overlap_priority_is_permutation_of_categories():
    assert sorted(config.OVERLAP_PRIORITY) == sorted(config.CUAD_CATEGORIES)


def test_priority_rank_is_complete_and_zero_based():
    assert len(config.PRIORITY_RANK) == 41
    ranks = sorted(config.PRIORITY_RANK.values())
    assert ranks == list(range(41))


def test_generic_categories_rank_lowest():
    # "Parties" and "Document Name" are the frequent/structural categories and must
    # lose overlap ties to rarer clauses -> they should have the two highest ranks.
    rank = config.PRIORITY_RANK
    generic = {rank["Parties"], rank["Document Name"]}
    assert generic == {39, 40}
