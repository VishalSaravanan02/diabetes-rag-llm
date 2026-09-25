"""Hand-computed checks for the retrieval metrics (run: python -m pytest -q)."""

import math

import pytest

from src.evaluation.metrics import (
    hit_at_k, mean, mrr_at_k, ndcg_at_k, percentile, pmid_ranking, recall_at_k,
)


def test_pmid_ranking_collapses_chunks_keeping_best_rank():
    results = [{"pmid": "A"}, {"pmid": "B"}, {"pmid": "A"}, {"pmid": 3}]
    assert pmid_ranking(results) == ["A", "B", "3"]


RANKING = ["p1", "g1", "p2", "g2", "p3"]
GOLD = {"g1", "g2"}


def test_hit_at_k():
    assert hit_at_k(RANKING, GOLD, 1) == 0.0
    assert hit_at_k(RANKING, GOLD, 2) == 1.0


def test_recall_at_k():
    assert recall_at_k(RANKING, GOLD, 2) == 0.5
    assert recall_at_k(RANKING, GOLD, 4) == 1.0
    assert recall_at_k(RANKING, set(), 4) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(RANKING, GOLD) == 0.5          # first gold at rank 2
    assert mrr_at_k(RANKING, {"zzz"}) == 0.0
    assert mrr_at_k(RANKING, {"g2"}, k=3) == 0.0   # rank 4 is outside k=3


def test_ndcg_at_k():
    # gold at ranks 2 and 4 ; ideal = gold at ranks 1 and 2
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKING, GOLD) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["g1", "g2"], GOLD) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], set()) == 0.0


def test_mean_and_percentile():
    assert mean([1, 2, 3]) == 2
    assert math.isnan(mean([]))
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([5], 95) == 5
    assert math.isnan(percentile([], 50))
