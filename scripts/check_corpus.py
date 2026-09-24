"""
Quick quality check on a fetched abstracts file.

Usage (from the project root):
    python -m scripts.check_corpus                        # defaults to config.ABSTRACTS_FILE
    python -m scripts.check_corpus --path data/other.json --sample 20

Prints:
  - total record count, and the fetch details from the .meta.json sidecar (if present)
  - % of records whose title/abstract mentions "diabet" (a crude on-topic proxy)
  - abstract length stats (spot missing/tiny abstracts)
  - full year distribution, and how many records have no year
  - top journals and top publication types (study designs)
  - duplicate-PMID check
  - a random sample of titles to eyeball
  - a short list of warnings, if anything looks off
"""

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

# Make the project root importable when run as `python scripts/check_corpus.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ABSTRACTS_FILE  # noqa: E402

LINE = "=" * 70


def _load(path):
    """Load the corpus, exiting with a clear message if it's missing, invalid or empty."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"No corpus found at {path}. Run `python -m src.fetch_data ...` first.")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid JSON ({e}).")
    if not isinstance(data, list):
        sys.exit(f"{path} should contain a list of records, found {type(data).__name__}.")
    if not data:
        sys.exit(f"{path} contains 0 records. Nothing to check.")
    return data


def _meta_path(path):
    path = Path(path)
    return path.with_name(path.stem + ".meta.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spot-check a fetched abstracts corpus.")
    parser.add_argument("--path", default=str(ABSTRACTS_FILE))
    parser.add_argument("--sample", type=int, default=20, help="How many random titles to print.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for a repeatable sample.")
    args = parser.parse_args(argv)

    data = _load(args.path)
    n = len(data)
    warnings = []

    print(f"\n{LINE}\nCORPUS CHECK: {args.path}\n{LINE}")
    print(f"Total records: {n}")

    # Fetch details from the sidecar written by src/fetch_data.py
    meta_file = _meta_path(args.path)
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        print(f"Fetch mode: {meta.get('mode', '?')}   fetched at: {meta.get('fetched_at', '?')}")
        if meta.get("mode") == "per_year":
            print(f"Years {meta.get('from_year')}-{meta.get('to_year')}, "
                  f"up to {meta.get('per_year')} per year")
    else:
        print("(no .meta.json sidecar: this corpus was fetched with an older fetcher)")

    # On-topic proxy
    on_topic = sum(
        1 for a in data
        if "diabet" in ((a.get("abstract") or "") + (a.get("title") or "")).lower()
    )
    on_topic_pct = 100 * on_topic / n
    print(f"Mention 'diabet' in title/abstract: {on_topic}/{n} ({on_topic_pct:.1f}%)")
    if on_topic_pct < 95:
        warnings.append(f"Only {on_topic_pct:.1f}% of records mention diabetes. Check the query.")

    # Abstract length stats
    lengths = [len(a.get("abstract") or "") for a in data]
    empty = sum(1 for L in lengths if L == 0)
    tiny = sum(1 for L in lengths if 0 < L < 100)
    print(f"Empty abstracts: {empty}   | Very short (<100 chars): {tiny}")
    print(f"Abstract length: min {min(lengths)}, median {int(statistics.median(lengths))}, max {max(lengths)}")
    if empty:
        warnings.append(f"{empty} records have an empty abstract.")

    # Year distribution (full, in year order, with a small bar chart)
    years = Counter(a.get("year") for a in data if a.get("year"))
    no_year = n - sum(years.values())
    print("\nYear distribution:")
    if years:
        biggest = max(years.values())
        for yr in sorted(years):
            bar = "#" * max(1, round(30 * years[yr] / biggest))
            print(f"  {yr}: {years[yr]:>5}  {bar}")
        top_share = 100 * biggest / n
        if top_share > 50:
            top_year = max(years, key=years.get)
            warnings.append(f"{top_share:.0f}% of records are from {top_year}. The corpus is unbalanced.")
    print(f"  no year: {no_year}")
    if no_year / n > 0.02:
        warnings.append(f"{no_year} records ({100 * no_year / n:.1f}%) have no year.")

    # Top journals
    journals = Counter(a.get("journal") for a in data if a.get("journal"))
    print("\nTop 8 journals:")
    for j, c in journals.most_common(8):
        print(f"  {c:>5}  {j}")

    # Top publication types (study designs)
    if any("publication_types" in a for a in data):
        types = Counter(t for a in data for t in (a.get("publication_types") or []))
        print("\nTop 10 publication types (a paper can have several):")
        for t, c in types.most_common(10):
            print(f"  {c:>5}  {t}")
    else:
        print("\n(no publication_types field: fetched before study types were collected)")

    # Duplicate PMIDs
    pmids = [a.get("pmid") for a in data]
    dupes = [p for p, c in Counter(pmids).items() if c > 1]
    print(f"\nDuplicate PMIDs: {len(dupes)}" + (f"  -> {dupes[:5]}" if dupes else "  (none, good)"))
    if dupes:
        warnings.append(f"{len(dupes)} duplicate PMIDs.")

    # Random sample of titles
    rng = random.Random(args.seed)
    k = min(args.sample, n)
    print(f"\n{'-' * 70}\nRandom sample of {k} titles:\n{'-' * 70}")
    for a in rng.sample(data, k):
        yr = a.get("year") or "????"
        print(f"  [{yr}] {(a.get('title') or '(no title)')[:100]}")

    print(f"\n{LINE}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("No warnings.")
    print("Eyeball the sample above: every title should be clearly diabetes-related.")
    print(f"{LINE}\n")


if __name__ == "__main__":
    main()