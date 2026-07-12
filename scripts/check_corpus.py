"""
Quick quality check on a fetched abstracts file.

Usage:
    python check_corpus.py                                  # defaults to data/diabetes_abstracts.json
    python check_corpus.py --path data/diabetes_abstracts.json --sample 20

Prints:
  - total record count
  - % of records whose abstract mentions "diabet" (a crude on-topic proxy)
  - abstract length stats (spot missing/tiny abstracts)
  - year distribution
  - top journals
  - a random sample of titles to eyeball
  - duplicate-PMID check
"""

import argparse
import json
import random
import statistics
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Spot-check a fetched abstracts corpus.")
    parser.add_argument("--path", default="data/diabetes_abstracts.json")
    parser.add_argument("--sample", type=int, default=20, help="How many random titles to print.")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    print(f"\n{'='*70}\nCORPUS CHECK: {args.path}\n{'='*70}")
    print(f"Total records: {n}")

    # On-topic proxy
    on_topic = sum(1 for a in data if "diabet" in (a.get("abstract", "") + a.get("title", "")).lower())
    print(f"Mention 'diabet' in title/abstract: {on_topic}/{n} ({100*on_topic/n:.1f}%)")

    # Abstract length stats
    lengths = [len(a.get("abstract", "")) for a in data]
    empty = sum(1 for L in lengths if L == 0)
    tiny = sum(1 for L in lengths if 0 < L < 100)
    print(f"Empty abstracts: {empty}   | Very short (<100 chars): {tiny}")
    print(f"Abstract length — min {min(lengths)}, median {int(statistics.median(lengths))}, max {max(lengths)}")

    # Year distribution
    years = Counter(a.get("year") for a in data if a.get("year"))
    print("\nYear distribution (top 8):")
    for yr, c in sorted(years.items(), key=lambda x: -x[1])[:8]:
        print(f"  {yr}: {c}")

    # Top journals
    journals = Counter(a.get("journal", "") for a in data if a.get("journal"))
    print("\nTop 8 journals:")
    for j, c in journals.most_common(8):
        print(f"  {c:>4}  {j}")

    # Duplicate PMIDs
    pmids = [a.get("pmid") for a in data]
    dupes = [p for p, c in Counter(pmids).items() if c > 1]
    print(f"\nDuplicate PMIDs: {len(dupes)}" + (f"  -> {dupes[:5]}" if dupes else "  (none — good)"))

    # Random sample of titles
    print(f"\n{'-'*70}\nRandom sample of {args.sample} titles:\n{'-'*70}")
    for a in random.sample(data, min(args.sample, n)):
        yr = a.get("year", "????")
        print(f"  [{yr}] {a.get('title','(no title)')[:100]}")

    print(f"\n{'='*70}")
    print("Eyeball the sample above: every title should be clearly diabetes-related.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()