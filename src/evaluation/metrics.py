"""
Retrieval metrics, computed at PAPER level.

The retriever returns chunks, and several chunks can come from one paper. A
question's "gold" answer is a set of PMIDs, so we first collapse the ranked
chunks into a ranked list of unique PMIDs (pmid_ranking), then score that.

All functions are pure (no I/O), so they are easy to test by hand.
"""

import math


def pmid_ranking(results):
    """Ranked chunk results (best first) -> unique PMIDs in rank order."""
    seen, ranking = set(), []
    for r in results:
        pmid = str(r["pmid"])
        if pmid not in seen:
            seen.add(pmid)
            ranking.append(pmid)
    return ranking


def hit_at_k(ranking, gold, k):
    """1.0 if ANY gold paper is in the top k, else 0.0."""
    return 1.0 if set(ranking[:k]) & set(gold) else 0.0


def recall_at_k(ranking, gold, k):
    """Share of the gold papers found in the top k."""
    gold = set(gold)
    return len(set(ranking[:k]) & gold) / len(gold) if gold else 0.0


def mrr_at_k(ranking, gold, k=10):
    """1 / rank of the first gold paper (0 if none in the top k)."""
    gold = set(gold)
    for rank, pmid in enumerate(ranking[:k], start=1):
        if pmid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking, gold, k=10):
    """
    Normalised discounted cumulative gain with binary relevance, in [0, 1].
    Rewards putting gold papers near the top; 1.0 = all gold papers ranked first.
    """
    gold = set(gold)
    dcg = sum(1 / math.log2(rank + 1) for rank, pmid in enumerate(ranking[:k], start=1) if pmid in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(values, p):
    """p-th percentile (0-100) with linear interpolation; nan for no values."""
    values = sorted(values)
    if not values:
        return float("nan")
    pos = (len(values) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)
