"""
Ask the PubMed RAG system questions from the terminal.

Usage (from the project root):
    python main.py "What is HbA1c?"      # one question
    python main.py                       # interactive; type 'exit' to stop
"""

import sys
import textwrap

from config import TOP_K
from src.pipeline import STATUS_ERROR, STATUS_NO_RESULTS, answer_question
from src.store import IndexNotReadyError

PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{}/"


def print_result(result):
    print()
    if result.status == STATUS_NO_RESULTS:
        print("No sufficiently relevant papers found in the knowledge base.")
        print("Try rephrasing, or ask something more specific to diabetes research.")
        return

    if result.status == STATUS_ERROR:
        print(f"Error: {result.error}")
    else:
        print("=== Answer ===\n")
        for paragraph in result.answer.split("\n"):
            print(textwrap.fill(paragraph, width=100) if paragraph else "")

    print("\n=== Sources ===")
    for i, s in enumerate(result.sources, 1):
        byline = ", ".join(str(x) for x in (s.get("authors_display"), s.get("journal"), s.get("year")) if x)
        print(f"[{i}] {s.get('title') or 'Untitled'}")
        print(f"    {byline}")
        print(f"    PMID {s['pmid']}  similarity {s['score']:.2f}  {PUBMED_URL.format(s['pmid'])}")


def main():
    try:
        if len(sys.argv) > 1:
            print_result(answer_question(" ".join(sys.argv[1:]), top_k=TOP_K))
            return

        print("=== PubMed Diabetes RAG (local LLM) ===")
        print("Type a question, or 'exit' to stop.")
        while True:
            question = input("\n> ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if question:
                print_result(answer_question(question, top_k=TOP_K))
    except IndexNotReadyError as e:
        sys.exit(str(e))
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
