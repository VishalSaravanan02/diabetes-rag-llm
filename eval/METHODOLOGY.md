# Evaluation methodology

How this project measures quality, and why it is done this way.

## Two separate questions

1. **Retrieval:** does the search step find the right papers? Measured exactly, with no LLM
   involved, so it is cheap to run after every change.
2. **Answers:** is the generated answer faithful to the sources, relevant and correctly cited?
   This needs an LLM judge, so it is measured less often (baseline and final configurations).

Keeping them apart shows *where* a problem is: a bad answer from good sources is a
generation problem; a good-looking answer from the wrong sources is a retrieval problem.

## The test set (in progress)

Target composition, about 80 questions:

| Type | Count | How it's made | Gold answer |
|---|---|---|---|
| Single-paper fact | ~40 | Drafted by an LLM from a sampled abstract, then reviewed, edited or dropped by a person | That abstract's PMID |
| Multi-paper synthesis | ~15 | Written by hand | Relevant PMIDs found by pooling (below) |
| Definitional / basic | ~15 | Written by hand | Relevant PMIDs found by pooling |
| Unanswerable | ~10 | Written by hand: plausible but not covered, plus off-topic | None: the system should decline |

**Drafting (single-paper questions).** `eval/draft_questions.py` samples abstracts evenly across
publication years (fixed random seed) and asks a *different* model from the one that answers in the
app (llama3.1 vs llama3) to write one realistic question per abstract, in its own words, plus a short
reference answer from the abstract. Drafts that refer to "this study" or copy the title are flagged.

**Human review.** Every draft is kept (often after editing) or dropped in `eval/review_app.py`.
Drafts are dropped if they are vague, trivial, answerable from general knowledge alone, or not
clearly supported by the abstract. Questions that reuse the abstract's exact wording are reworded,
because real users paraphrase, and copied wording makes keyword matching look unrealistically good.

**Pooling (multi-paper and definitional questions).** The union of the top results from several
retrieval methods is judged relevant or not by hand; the relevant PMIDs become the gold set. Using
several methods reduces the bias of judging only what one method already finds.

**Splits and freezing.** About 30% of questions are marked `dev` (used while tuning) and 70% `test`
(used only to report final results), so improvements are not tuned to the reported numbers. Once
built, the set is frozen as `eval/questions_v1.jsonl`; any later change becomes `questions_v2`.

## Retrieval metrics

Computed at paper level: the top 50 retrieved chunks are collapsed to a ranked list of unique PMIDs.

| Metric | Plain meaning |
|---|---|
| Hit@k | Was *any* right paper in the top k? |
| Recall@k | What share of the right papers was in the top k? |
| MRR@10 | How high was the first right paper? (1 = first place, 0.5 = second, ...) |
| nDCG@10 | Were the right papers near the top? (0 to 1) |
| False reject rate | Share of answerable questions where no source passed the relevance threshold |
| Abstain rate (unanswerable) | Share of unanswerable questions where no source passed the threshold |
| Latency p50 / p95 | Search time per question, median and 95th percentile |

## Run tracking

Every run is saved as `results/runs/<run_id>/run.json` with the git commit (and whether there were
uncommitted code changes), every setting in `config.py`, the index metadata, the metrics and the
per-question results. `python -m src.evaluation.compare_runs` builds `results/summary.md`.

## Known limitations

- The test set is small (about 80 questions), so differences of a few points can be noise.
- Single-paper questions have exactly one gold paper, although other papers may also answer them;
  this underestimates recall somewhat, equally for every configuration.
- Questions were drafted by an LLM from the same corpus, which may favour some topics.
