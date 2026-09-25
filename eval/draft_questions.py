"""
Draft candidate test questions from real abstracts, for a human to review.

For a sample of abstracts (stratified by publication year, fixed random seed),
an LLM writes one question that the abstract answers, plus a short reference
answer. The abstract's PMID becomes the question's "gold" answer: a good search
system should return that paper for that question.

Drafts are NOT the test set. Every draft is reviewed (kept, edited or dropped)
with the review app:  python -m streamlit run eval/review_app.py

The drafting model should differ from the model that answers questions in the
app (config.LLM_MODEL), so the test isn't written in the answerer's own style.

Usage (from the project root; Ollama must be running):
    python -m eval.draft_questions                     # 120 drafts with llama3.1
    python -m eval.draft_questions --n 150 --model qwen2.5:7b

Resumable: re-running skips papers that already have a draft.
"""

import argparse
import datetime as dt
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import ollama

from config import ABSTRACTS_FILE, LLM_MODEL, ROOT_DIR

OUT_FILE = ROOT_DIR / "eval" / "candidates.jsonl"
MIN_CHARS = 600          # enough content for a specific question

PROMPT = """Here is a PubMed abstract.

Title: {title}
Abstract: {abstract}

Write ONE question that:
- a clinician or researcher might realistically ask,
- is answered by a SPECIFIC finding in this abstract (not by general knowledge alone),
- makes sense on its own: never refer to "this study", "the authors", "the abstract" or "the paper",
- uses your own wording: do not copy the title or distinctive phrases from the abstract,
- is under 25 words.

Also write a reference answer of 1-2 sentences, based ONLY on the abstract.

Respond with JSON only: {{"question": "...", "reference": "..."}}"""

SELF_REFERENCE = re.compile(r"\b(this|the) (study|paper|abstract|article|trial|authors?|research)\b", re.I)
STOPWORDS = set("""a an and are as at be by for from has have in is it of on or that the their to was were
what which with how does do did can could would should among between patients patient study
this that these about find found known any people""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def title_overlap(question, title):
    """Share of the question's content words that also appear in the title (0-1)."""
    q = content_words(question)
    return len(q & content_words(title)) / len(q) if q else 0.0


def flags_for(question, reference, title):
    flags = []
    if SELF_REFERENCE.search(question):
        flags.append("refers to 'this study' or similar")
    if title_overlap(question, title) >= 0.6:
        flags.append("copies many words from the title")
    if len(question.split()) > 30:
        flags.append("long question")
    if len(reference.split()) < 5:
        flags.append("very short reference answer")
    return flags


def sample_abstracts(abstracts, n, seed):
    """Up to n abstracts, spread evenly across publication years."""
    by_year = defaultdict(list)
    for a in abstracts:
        if len(a.get("abstract") or "") >= MIN_CHARS and a.get("year"):
            by_year[a["year"]].append(a)
    rng = random.Random(seed)
    for items in by_year.values():
        rng.shuffle(items)
    years = sorted(by_year)
    picked, i = [], 0
    while len(picked) < n and any(by_year[y] for y in years):   # round-robin over years
        y = years[i % len(years)]
        if by_year[y]:
            picked.append(by_year[y].pop())
        i += 1
    return picked


def draft_one(model, paper):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(title=paper["title"], abstract=paper["abstract"])}],
        format="json",
        options={"temperature": 0.3},
    )
    data = json.loads(response["message"]["content"])
    question = str(data.get("question", "")).strip()
    reference = str(data.get("reference", "")).strip()
    if not question or not reference:
        raise ValueError("empty question or reference")
    return question, reference


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draft candidate test questions for review.")
    parser.add_argument("--n", type=int, default=120, help="How many drafts in total (default 120).")
    parser.add_argument("--model", default="llama3.1", help="Ollama model that writes the drafts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args(argv)

    if args.model == LLM_MODEL:
        print(f"WARNING: drafting with {args.model}, the same model that answers questions in the app. "
              "A different model avoids writing the test in the answerer's own style.\n")

    with open(ABSTRACTS_FILE, encoding="utf-8") as f:
        abstracts = json.load(f)

    out = Path(args.out).resolve()
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["pmid"] for line in f if line.strip()}

    todo = [p for p in sample_abstracts(abstracts, args.n, args.seed) if p["pmid"] not in done]
    print(f"{len(done)} drafts already in {out.name}; drafting {len(todo)} more with {args.model}.")
    if not todo:
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, paper in enumerate(todo, 1):
            try:
                question, reference = draft_one(args.model, paper)
            except ConnectionError:
                sys.exit("Can't reach Ollama. Start it with `ollama serve`, then re-run (progress is kept).")
            except Exception as e:  # noqa: BLE001 - one bad draft shouldn't stop the run
                failed += 1
                print(f"  [{i}/{len(todo)}] PMID {paper['pmid']}: skipped ({e})")
                continue
            record = {
                "candidate_id": f"c{paper['pmid']}",
                "pmid": paper["pmid"],
                "type": "single_fact",
                "question": question,
                "reference": reference,
                "gold_pmids": [paper["pmid"]],
                "flags": flags_for(question, reference, paper["title"]),
                "drafted_by": args.model,
                "drafted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()                                  # progress survives interruptions
            flag = f"  [!] {'; '.join(record['flags'])}" if record["flags"] else ""
            print(f"  [{i}/{len(todo)}] ({paper['year']}) {question[:80]}{flag}")

    print(f"\nDone. {len(todo) - failed} drafted, {failed} skipped -> {out.relative_to(ROOT_DIR)}")
    print("Next: python -m streamlit run eval/review_app.py")


if __name__ == "__main__":
    main()
