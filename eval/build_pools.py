"""
Build candidate "pools" of papers for questions that have no single correct paper.

For broad questions ("What is insulin resistance?"), many papers in the corpus
may be relevant. To decide which ones count as correct, we pool the top results
of TWO different search methods and then judge each pooled paper by hand:

  1. dense   - the system's own meaning-based search (MiniLM + FAISS)
  2. keyword - TF-IDF over title + abstract (scikit-learn), matching words

Using two methods reduces pooling bias: judging only what the current system
finds would let it mark its own homework, and a better method later could find
good papers nobody judged.

Writes
  eval/pools.jsonl                        which papers were pooled, by which method (committed)
  ~/Downloads/pools_for_review.json       the same, with titles and abstracts, for judging

Usage (from the project root):
    python -m eval.build_pools                      # 8 + 8 papers per question
    python -m eval.build_pools --depth 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from config import ABSTRACTS_FILE, MIN_SIMILARITY, ROOT_DIR
from src.evaluation.metrics import pmid_ranking
from src.retriever import retrieve_candidates
from src.store import IndexNotReadyError

QUESTIONS = ROOT_DIR / "eval" / "hard_questions.jsonl"
POOLS = ROOT_DIR / "eval" / "pools.jsonl"
EXPORT = Path(os.path.expanduser("~/Downloads/pools_for_review.json"))


def keyword_search(papers):
    """Return a function question -> ranked PMIDs, using TF-IDF over title + abstract."""
    docs = [f"{p.get('title') or ''} {p.get('abstract') or ''}" for p in papers]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)
    pmids = [p["pmid"] for p in papers]

    def search(question, k):
        scores = linear_kernel(vectorizer.transform([question]), matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [pmids[i] for i in top if scores[i] > 0]
    return search


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pool candidate papers for broad questions.")
    parser.add_argument("--depth", type=int, default=8, help="Papers taken from each method (default 8).")
    args = parser.parse_args(argv)

    with open(ABSTRACTS_FILE, encoding="utf-8") as f:
        papers = json.load(f)
    by_pmid = {p["pmid"]: p for p in papers}
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print(f"Pooling {len(questions)} questions: top {args.depth} from dense + top {args.depth} from keyword search.")
    keyword = keyword_search(papers)

    pools, export = [], []
    for q in questions:
        try:
            dense_results = retrieve_candidates(q["question"], 60)
        except IndexNotReadyError as e:
            sys.exit(str(e))
        dense = pmid_ranking(dense_results)[:args.depth]
        kw = keyword(q["question"], args.depth)
        top_score = dense_results[0]["score"] if dense_results else float("-inf")

        pooled = []
        for pmid in dict.fromkeys(dense + kw):          # union, dense first, no duplicates
            pooled.append({
                "pmid": pmid,
                "dense_rank": dense.index(pmid) + 1 if pmid in dense else None,
                "keyword_rank": kw.index(pmid) + 1 if pmid in kw else None,
            })
        pools.append({"id": q["id"], "type": q["type"], "question": q["question"],
                      "top_dense_score": round(top_score, 4),
                      "passes_threshold": top_score >= MIN_SIMILARITY, "pool": pooled})
        export.append({**pools[-1], "pool": [
            {**p, "title": by_pmid[p["pmid"]].get("title"), "year": by_pmid[p["pmid"]].get("year"),
             "abstract": by_pmid[p["pmid"]].get("abstract")} for p in pooled]})

        both = sum(1 for p in pooled if p["dense_rank"] and p["keyword_rank"])
        flag = "" if q["type"] != "unanswerable" else ("  <- passes threshold" if top_score >= MIN_SIMILARITY else "  (below threshold)")
        print(f"  {q['id']}  pool {len(pooled):>2} papers ({both} found by both)  top score {top_score:.3f}{flag}")

    with open(POOLS, "w", encoding="utf-8") as f:
        for p in pools:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(EXPORT, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=1, ensure_ascii=False)

    total = sum(len(p["pool"]) for p in pools)
    print(f"\n{total} pooled papers in total.")
    print(f"Saved {POOLS.relative_to(ROOT_DIR)} and {EXPORT}")
    print("Next: upload pools_for_review.json to Claude for the relevance pre-screen.")


if __name__ == "__main__":
    main()
