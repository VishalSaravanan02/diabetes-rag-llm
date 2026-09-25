"""
Choose config.MIN_SIMILARITY from evidence instead of guessing.

For a set of on-topic questions (should find sources) and off-topic questions
(should find nothing), this records the TOP-1 cosine similarity of each, then
suggests a threshold that keeps on-topic questions and rejects off-topic ones.

Re-run whenever the embedding model or EMBED_WITH_TITLE changes: scores from
different models are not comparable.

Usage (from the project root, after building the index):
    python -m scripts.calibrate_threshold
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EMBEDDING_MODEL, MIN_SIMILARITY  # noqa: E402
from src.retriever import retrieve_candidates  # noqa: E402
from src.store import IndexNotReadyError  # noqa: E402

ON_TOPIC = [
    "What is HbA1c and why is it measured?",
    "What causes insulin resistance?",
    "Do SGLT2 inhibitors reduce heart failure hospitalisation?",
    "What are the side effects of metformin?",
    "How does GLP-1 receptor agonist treatment affect body weight?",
    "What are the risk factors for gestational diabetes?",
    "How is diabetic retinopathy screened and treated?",
    "What is the role of autoantibodies in type 1 diabetes?",
    "Does bariatric surgery lead to remission of type 2 diabetes?",
    "What causes diabetic foot ulcers and how are they managed?",
    "How does diabetes affect kidney function?",
    "Is periodontitis associated with diabetes?",
    "What are the benefits of continuous glucose monitoring?",
    "How effective is lifestyle intervention for preventing type 2 diabetes?",
    "What is the relationship between obesity and type 2 diabetes?",
    "How does diabetes increase cardiovascular risk?",
    "What are the symptoms of diabetic peripheral neuropathy?",
    "How is hypoglycaemia defined and prevented?",
    "Does vitamin D supplementation improve glycaemic control?",
    "What is maturity-onset diabetes of the young (MODY)?",
]

OFF_TOPIC = [
    "What is the best pizza in Naples?",
    "How do rockets reach orbit?",
    "Who won the football World Cup in 2014?",
    "How do I change a car tyre?",
    "What is the capital of Australia?",
    "How do I learn to play the guitar?",
    "What is the plot of Hamlet?",
    "How does a blockchain work?",
    "What is the tallest mountain in Europe?",
    "How do I bake sourdough bread?",
]


def top1(question):
    results = retrieve_candidates(question, 1)
    return (results[0]["score"], results[0]["title"] or "") if results else (float("-inf"), "")


def main():
    print(f"Embedding model: {EMBEDDING_MODEL}   current MIN_SIMILARITY: {MIN_SIMILARITY}\n")
    try:
        on = sorted(((*top1(q), q) for q in ON_TOPIC), key=lambda x: x[0])
        off = sorted(((*top1(q), q) for q in OFF_TOPIC), key=lambda x: x[0], reverse=True)
    except IndexNotReadyError as e:
        sys.exit(str(e))

    print("ON-TOPIC (lowest first; these should all pass):")
    for score, title, q in on:
        print(f"  {score:.3f}  {q}\n         -> {title[:80]}")
    print("\nOFF-TOPIC (highest first; these should all be rejected):")
    for score, _, q in off:
        print(f"  {score:.3f}  {q}")

    lowest_on, highest_off = on[0][0], off[0][0]
    print(f"\nLowest on-topic score:   {lowest_on:.3f}")
    print(f"Highest off-topic score: {highest_off:.3f}")

    if lowest_on > highest_off:
        suggestion = round((lowest_on + highest_off) / 2, 2)
        print(f"\nThe two groups are cleanly separated. Suggested MIN_SIMILARITY = {suggestion}")
        print("(the midpoint of the gap, which leaves a safety margin on both sides)")
    else:
        suggestion = round(lowest_on - 0.01, 2)
        leaked = sum(1 for s, _, _ in off if s >= suggestion)
        print("\nThe groups overlap: no threshold separates them perfectly.")
        print(f"Keeping every on-topic question needs MIN_SIMILARITY = {suggestion}, "
              f"which lets {leaked}/{len(off)} off-topic questions through.")
        print("Prefer keeping on-topic questions: the LLM prompt can still refuse weak context.")

    passing_now = sum(1 for s, _, _ in on if s >= MIN_SIMILARITY)
    rejected_now = sum(1 for s, _, _ in off if s < MIN_SIMILARITY)
    print(f"\nWith the current value ({MIN_SIMILARITY}): {passing_now}/{len(on)} on-topic pass, "
          f"{rejected_now}/{len(off)} off-topic rejected.")


if __name__ == "__main__":
    main()
