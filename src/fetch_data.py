"""
Fetch PubMed abstracts for a topic and save them with rich metadata.

Design notes
------------
- Uses the Entrez **history server** (usehistory="y"): esearch stores the full
  result set on NCBI's side and returns a WebEnv + query_key handle. We then
  page through it with efetch in batches, instead of pulling thousands of IDs
  into one giant request.
- Fetches in **XML** (retmode="xml") and parses structured records, so each
  article's PMID comes from its own record. This removes the old string-splitting
  alignment bug entirely — a chunk can never be attributed to the wrong paper.
- Respects NCBI rate limits: 3 requests/sec without an API key, 10/sec with one
  (set NCBI_API_KEY). We sleep between batches and retry with backoff on failure.

Environment variables
---------------------
ENTREZ_EMAIL   (required by NCBI)   e.g. export ENTREZ_EMAIL="you@example.com"
NCBI_API_KEY   (optional, faster)   e.g. export NCBI_API_KEY="..."
"""

import argparse
import json
import os
import time
from pathlib import Path

from Bio import Entrez

# Load variables from a local .env file (if present) into the environment.
# .env is git-ignored, so real values never get committed — the code only ever
# refers to the variable *names*. If python-dotenv isn't installed, we fail
# soft: the script still works as long as the vars are exported in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ── NCBI credentials ─────────────────────────────────────────────────────────
Entrez.email = os.environ.get("ENTREZ_EMAIL", "your@email.com")
_API_KEY = os.environ.get("NCBI_API_KEY")
if _API_KEY:
    Entrez.api_key = _API_KEY

# With an API key NCBI allows 10 req/s, otherwise 3 req/s. Stay well under.
_SLEEP_BETWEEN_BATCHES = 0.15 if _API_KEY else 0.4

# Default topic query. MeSH Major Topic keeps results genuinely on-subject,
# hasabstract drops citations with no abstract text, english[lang] filters language.
DEFAULT_QUERY = (
    '"diabetes mellitus"[MeSH Major Topic] '
    "AND hasabstract[text] "
    "AND english[lang]"
)


def _search(query, max_results):
    """Run esearch with history enabled; return (webenv, query_key, count)."""
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=0,            # we don't need the IDs inline; history holds them
        usehistory="y",
        sort="relevance",
    )
    record = Entrez.read(handle)
    handle.close()

    count = min(int(record["Count"]), max_results)
    return record["WebEnv"], record["QueryKey"], count


def _fetch_batch(webenv, query_key, start, batch_size, max_retries=3):
    """Fetch one XML batch from the history server, with retry + backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            handle = Entrez.efetch(
                db="pubmed",
                retmode="xml",
                retstart=start,
                retmax=batch_size,
                webenv=webenv,
                query_key=query_key,
            )
            records = Entrez.read(handle)
            handle.close()
            return records["PubmedArticle"]
        except Exception as e:  # noqa: BLE001 - network/parse errors both retry
            wait = 2 ** attempt
            print(f"    ! batch at {start} failed ({e}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
    print(f"    ! giving up on batch at {start}")
    return []


def _parse_article(article):
    """Pull the fields we care about out of one PubmedArticle XML record."""
    medline = article["MedlineCitation"]
    pmid = str(medline["PMID"])

    art = medline["Article"]

    # Title
    title = str(art.get("ArticleTitle", "")).strip()

    # Abstract: may be split into labelled sections (BACKGROUND, METHODS, ...).
    abstract = ""
    if "Abstract" in art:
        parts = art["Abstract"].get("AbstractText", [])
        chunks = []
        for p in parts:
            label = p.attributes.get("Label") if hasattr(p, "attributes") else None
            text = str(p).strip()
            chunks.append(f"{label}: {text}" if label else text)
        abstract = " ".join(chunks).strip()

    # Journal + year
    journal = str(art.get("Journal", {}).get("Title", "")).strip()
    year = None
    try:
        pubdate = art["Journal"]["JournalIssue"]["PubDate"]
        year = int(pubdate["Year"]) if "Year" in pubdate else None
    except (KeyError, ValueError, TypeError):
        year = None

    # Authors: first three surnames, then "et al." if more.
    authors = []
    for a in art.get("AuthorList", []):
        last = a.get("LastName")
        if last:
            authors.append(str(last))
    if len(authors) > 3:
        authors = authors[:3] + ["et al."]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": authors,
    }


def fetch_pubmed(query=DEFAULT_QUERY, max_results=5000, batch_size=200, out_path="data/diabetes_abstracts.json"):
    """Fetch abstracts for `query` and write them (with metadata) to `out_path`."""
    if Entrez.email == "your@email.com":
        print("WARNING: ENTREZ_EMAIL is not set. NCBI requires a real email; "
              "set it with `export ENTREZ_EMAIL=you@example.com`.")

    print(f"Searching PubMed:\n  {query}\n")
    webenv, query_key, count = _search(query, max_results)
    print(f"Will fetch {count} records in batches of {batch_size}.\n")

    articles = []
    for start in range(0, count, batch_size):
        print(f"  fetching {start + 1}-{min(start + batch_size, count)} of {count} ...")
        batch = _fetch_batch(webenv, query_key, start, batch_size)
        for article in batch:
            try:
                parsed = _parse_article(article)
                if parsed["abstract"]:          # skip anything with an empty abstract
                    articles.append(parsed)
            except Exception as e:              # noqa: BLE001 - one bad record shouldn't kill the run
                print(f"    ! skipped a record: {e}")
        time.sleep(_SLEEP_BETWEEN_BATCHES)

    # De-duplicate by PMID (histories can occasionally repeat)
    seen, deduped = set(), []
    for a in articles:
        if a["pmid"] not in seen:
            seen.add(a["pmid"])
            deduped.append(a)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(deduped)} abstracts (with metadata) to {out}")
    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts with metadata.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query.")
    parser.add_argument("--max-results", type=int, default=5000, help="Max abstracts to fetch.")
    parser.add_argument("--batch-size", type=int, default=200, help="Records per efetch call.")
    parser.add_argument("--out", default="data/diabetes_abstracts.json", help="Output JSON path.")
    args = parser.parse_args()

    fetch_pubmed(
        query=args.query,
        max_results=args.max_results,
        batch_size=args.batch_size,
        out_path=args.out,
    )