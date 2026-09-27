"""
Freeze the test set: combine every reviewed question into eval/questions_v1.jsonl.

Inputs (all committed):
  eval/reviewed.jsonl            single-paper questions: LLM-drafted, pre-screened, author-reviewed
  eval/hard_questions.jsonl      definitional, multi-paper and unanswerable questions
  eval/relevance_judgments.json  graded answer keys (0-2) for the harder questions

Output:
  eval/questions_v1.jsonl        one question per line: id, type, question, reference, gold_pmids, grades
  eval/questions_v1.meta.json    counts, input fingerprints and how each type was built

"Frozen" means the file is never edited again. Any change becomes questions_v2, so
results measured on v1 stay comparable with each other.

Usage (from the project root):
    python -m eval.freeze_questions
"""

import datetime as dt
import hashlib
import json
import sys
from collections import Counter

from config import ROOT_DIR
from src.evaluation.questions import load_questions

EVAL = ROOT_DIR / "eval"
REVIEWED = EVAL / "reviewed.jsonl"
HARD = EVAL / "hard_questions.jsonl"
JUDGMENTS = EVAL / "relevance_judgments.json"
OUT = EVAL / "questions_v1.jsonl"
META = EVAL / "questions_v1.meta.json"

SOURCES = {
    "single_fact": "Drafted by llama3.1 from one abstract; pre-screened by Claude; kept, edited or dropped by the author "
                   "after checking the reference answer against the abstract. Gold = the source paper (grade 2).",
    "definitional": "Written with Claude. Gold = pooled candidates (dense top 8 + TF-IDF top 8), pre-screened by Claude, "
                    "verified by the author, graded 0-2.",
    "multi_paper": "Written with Claude. Gold as for definitional questions.",
    "unanswerable": "Written with Claude; outside the corpus. Pools checked: no paper answers them. Gold = none.",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    singles = load_questions(REVIEWED)                    # keeps only 'keep' decisions, last one wins
    with open(HARD, encoding="utf-8") as f:
        hard = [json.loads(line) for line in f if line.strip()]
    judgments = json.load(open(JUDGMENTS, encoding="utf-8"))["questions"]

    records = []
    for n, q in enumerate(singles, 1):
        pmid = str(q["gold_pmids"][0])
        records.append({
            "id": f"s{n:02d}", "type": "single_fact", "question": q["question"],
            "reference": q.get("reference"), "gold_pmids": [pmid], "grades": {pmid: 2},
            "origin": {"candidate_id": q.get("candidate_id"), "edited_by_reviewer": q.get("edited")},
        })
    for q in hard:
        j = judgments.get(q["id"])
        if j is None or not j.get("confirmed"):
            sys.exit(f"Question {q['id']} has no confirmed relevance judgment in {JUDGMENTS.name}.")
        records.append({
            "id": q["id"], "type": q["type"], "question": q["question"], "reference": None,
            "gold_pmids": [str(p) for p in j["relevant_pmids"]],
            "grades": {str(p): int(g) for p, g in j.get("grades", {}).items()},
            "origin": {"changed_from_prescreen": j.get("changed_from_prescreen", [])},
        })

    # sanity checks
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for r in records:
        assert set(r["grades"]) == set(r["gold_pmids"]), f"{r['id']}: grades don't match gold PMIDs"
        assert (r["type"] == "unanswerable") == (not r["gold_pmids"]), f"{r['id']}: gold doesn't fit its type"
    return records


def main(argv=None):
    force = "--force" in (argv if argv is not None else sys.argv[1:])
    if OUT.exists() and not force:
        sys.exit(f"{OUT.name} already exists and is frozen. Build a new version (questions_v2) instead of overwriting it.")

    records = build()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["type"] for r in records)
    meta = {
        "version": 1,
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_questions": len(records),
        "by_type": dict(counts),
        "gold_papers": sum(len(r["gold_pmids"]) for r in records),
        "how_built": SOURCES,
        "inputs_sha256": {p.name: sha256(p) for p in (REVIEWED, HARD, JUDGMENTS)},
        "questions_sha256": sha256(OUT),
        "splits": "none: all questions are used for reporting (too few for a separate dev split); "
                  "Phase 6 experiments are fixed in advance and each is run once.",
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Frozen {len(records)} questions -> {OUT.relative_to(ROOT_DIR)}")
    for t in ("single_fact", "definitional", "multi_paper", "unanswerable"):
        print(f"  {t:<13} {counts.get(t, 0)}")
    print(f"Details -> {META.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
