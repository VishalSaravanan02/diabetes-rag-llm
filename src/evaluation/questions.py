"""
Load evaluation questions from a JSONL file (one JSON object per line).

Accepts both:
  - the frozen test set (eval/questions_v1.jsonl), and
  - the review log (eval/reviewed.jsonl), where each line is a keep/drop
    decision; the LAST decision per candidate wins and dropped ones are skipped.

Each returned question has at least: id, question, type, gold_pmids (list, may be
empty for unanswerable questions) and, if assigned, split ("dev" or "test").
"""

import json
from pathlib import Path


def load_questions(path, split=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No questions file at {path}")

    records = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n} is not valid JSON ({e})") from e
            key = rec.get("candidate_id") or rec.get("id") or f"line{n}"
            records[key] = rec                      # later lines override earlier ones

    questions = []
    for key, rec in records.items():
        if rec.get("decision") == "drop":
            continue
        if not rec.get("question"):
            raise ValueError(f"{path}: record {key} has no question")
        q = dict(rec)
        q.setdefault("id", key)
        q["gold_pmids"] = [str(p) for p in rec.get("gold_pmids", [])]
        q.setdefault("type", "single_fact")
        if split and q.get("split") != split:
            continue
        questions.append(q)
    return questions
