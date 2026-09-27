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


def ndcg_at_k(ranking, gold, k=10, grades=None):
    """
    Normalised discounted cumulative gain, in [0, 1].

    Rewards putting relevant papers near the top. With `grades` ({pmid: 0-2}),
    a directly answering paper (2) counts more than a partial one (1); without
    grades every gold paper counts 1. 1.0 = the best possible order.
    """
    gold = set(gold)
    gain = {p: (grades.get(p, 1) if grades else 1) for p in gold}
    dcg = sum(gain[pmid] / math.log2(rank + 1)
              for rank, pmid in enumerate(ranking[:k], start=1) if pmid in gold)
    ideal_gains = sorted(gain.values(), reverse=True)[:k]
    ideal = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_gains, start=1))
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


def bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=0):
    """
    Confidence interval for the mean of per-question scores, by resampling questions.

    "If we had asked a different set of questions like these, where would the average
    plausibly land?" A wide interval means the test set is too small to be sure.
    Returns (low, high); (nan, nan) for no values. Fixed seed = reproducible.
    """
    import random

    values = list(values)
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_resamples))
    tail = (1 - confidence) / 2
    return means[int(tail * n_resamples)], means[int((1 - tail) * n_resamples) - 1]
