"""
Measure search quality: for each question, does the right paper come back, and how high?

No LLM is involved, so this is exact, cheap and takes seconds.

For every question with gold PMIDs, the top CANDIDATES chunks are retrieved,
collapsed to a ranked list of unique papers, and scored with hit@k, recall@k,
MRR@10 and nDCG@10. Search latency is measured too.

It also checks the relevance threshold (MIN_SIMILARITY) without calling the LLM:
  - false_reject_rate: answerable questions where NOTHING passes the threshold
  - abstain_rate_unanswerable: unanswerable questions correctly returning nothing

Usage (from the project root):
    python -m src.evaluation.run_retrieval_eval --name baseline
    python -m src.evaluation.run_retrieval_eval --questions eval/reviewed.jsonl --name preview
    python -m src.evaluation.run_retrieval_eval --split dev --name E1_title
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from config import MIN_SIMILARITY, ROOT_DIR
from src.evaluation.metrics import (
    hit_at_k, mean, mrr_at_k, ndcg_at_k, percentile, pmid_ranking, recall_at_k,
)
from src.evaluation.questions import load_questions
from src.evaluation.runs import config_snapshot, git_info, new_run_id, save_run
from src.retriever import index_info, retrieve_candidates
from src.store import IndexNotReadyError

DEFAULT_QUESTIONS = ROOT_DIR / "eval" / "questions_v1.jsonl"
CANDIDATES = 50
KS = (1, 5, 10)


def evaluate(questions, n_candidates=CANDIDATES):
    per_question, latencies = [], []
    for q in questions:
        t0 = time.perf_counter()
        results = retrieve_candidates(q["question"], n_candidates)
        latencies.append((time.perf_counter() - t0) * 1000)

        ranking = pmid_ranking(results)
        top_score = results[0]["score"] if results else float("-inf")
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold_pmids": q["gold_pmids"],
            "top10": ranking[:10],
            "top_score": round(top_score, 4),
            "passes_threshold": top_score >= MIN_SIMILARITY,
        }
        if q["gold_pmids"]:
            gold = q["gold_pmids"]
            first = next((i for i, p in enumerate(ranking, 1) if p in gold), None)
            row["first_gold_rank"] = first
            for k in KS:
                row[f"hit@{k}"] = hit_at_k(ranking, gold, k)
                row[f"recall@{k}"] = recall_at_k(ranking, gold, k)
            row["mrr@10"] = mrr_at_k(ranking, gold, 10)
            row["ndcg@10"] = ndcg_at_k(ranking, gold, 10)
        per_question.append(row)

    answerable = [r for r in per_question if r["gold_pmids"]]
    unanswerable = [r for r in per_question if not r["gold_pmids"]]
    metrics = {"n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}
    for key in [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]:
        metrics[key] = round(mean(r[key] for r in answerable), 4)
    metrics["false_reject_rate"] = round(mean(not r["passes_threshold"] for r in answerable), 4)
    if unanswerable:
        metrics["abstain_rate_unanswerable"] = round(mean(not r["passes_threshold"] for r in unanswerable), 4)
    metrics["latency_ms_p50"] = round(percentile(latencies, 50), 1)
    metrics["latency_ms_p95"] = round(percentile(latencies, 95), 1)
    return metrics, per_question


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality (no LLM needed).")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--split", choices=["dev", "test"], help="Only use this split.")
    parser.add_argument("--name", required=True, help="Short run name, e.g. baseline or E1_title.")
    parser.add_argument("--candidates", type=int, default=CANDIDATES,
                        help="Chunks retrieved per question before collapsing to papers.")
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions, split=args.split)
        info = index_info()                                  # loads the index + model once
    except (FileNotFoundError, ValueError, IndexNotReadyError) as e:
        sys.exit(str(e))
    if not questions:
        sys.exit("No questions to evaluate (check --questions and --split).")

    retrieve_candidates("warm-up query", 1)                  # so timing excludes first-call setup
    metrics, per_question = evaluate(questions, args.candidates)

    qpath = Path(args.questions)
    run = {
        "run_id": new_run_id(args.name),
        "name": args.name,
        "kind": "retrieval",
        "questions_file": str(qpath.resolve().relative_to(ROOT_DIR)) if qpath.resolve().is_relative_to(ROOT_DIR) else str(qpath),
        "questions_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
        "split": args.split or "all",
        "candidates": args.candidates,
        "git": git_info(),
        "config": config_snapshot(),
        "index_meta": {k: v for k, v in info.items() if k != "chunks_meta"},
        "metrics": metrics,
        "per_question": per_question,
    }
    path = save_run(run)

    print(f"\nRetrieval evaluation: {run['name']}  ({metrics['n_answerable']} answerable"
          f"{', %d unanswerable' % metrics['n_unanswerable'] if metrics['n_unanswerable'] else ''}, "
          f"split: {run['split']})")
    print("-" * 60)
    for key in ("hit@1", "hit@5", "hit@10", "recall@5", "recall@10", "mrr@10", "ndcg@10"):
        print(f"  {key:<10} {metrics[key]:.3f}")
    print(f"  {'false_reject_rate':<26} {metrics['false_reject_rate']:.3f}   (answerable questions with no source above {MIN_SIMILARITY})")
    if "abstain_rate_unanswerable" in metrics:
        print(f"  {'abstain_rate_unanswerable':<26} {metrics['abstain_rate_unanswerable']:.3f}")
    print(f"  latency p50 / p95          {metrics['latency_ms_p50']} / {metrics['latency_ms_p95']} ms")
    if run["git"]["dirty"]:
        print("\n  note: uncommitted code changes; the git commit alone doesn't describe this run")

    misses = [r for r in per_question if r["gold_pmids"] and not r["hit@10"]]
    if misses:
        print(f"\nNot found in the top 10 ({len(misses)}):")
        for r in misses[:10]:
            print(f"  - {r['question'][:90]}")
    print(f"\nSaved: {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
