"""
Fetch PubMed abstracts for a topic and save them with rich metadata.

Two modes
---------
1. Single query (original behaviour):
       python -m src.fetch_data --max-results 5000
   One relevance-sorted search. PubMed's "Best Match" ranking favours recent
   papers, so a capped fetch ends up almost entirely from the latest year.

2. Per-year balanced fetch (recommended):
       python -m src.fetch_data --from-year 2010 --to-year 2026 --per-year 300
   One search per publication year, taking the top `--per-year` papers of each
   year. This gives even coverage across time, so the knowledge base contains
   foundational work as well as the newest research.

Design notes
------------
- Uses the Entrez **history server** (usehistory="y"): esearch stores the full
  result set on NCBI's side and returns a WebEnv + query_key handle. We then
  page through it with efetch in batches, instead of pulling thousands of IDs
  into one giant request.
- Fetches in **XML** (retmode="xml") and parses structured records, so each
  article's PMID comes from its own record. This removes the old string-splitting
  alignment bug entirely: a chunk can never be attributed to the wrong paper.
- Respects NCBI rate limits: 3 requests/sec without an API key, 10/sec with one
  (set NCBI_API_KEY). We sleep between batches and retry with backoff on failure.
- Writes a small sidecar file next to the output (<name>.meta.json) recording
  the query, mode, year range and per-year counts, so the corpus is reproducible.
- Refuses to overwrite an existing corpus unless --overwrite is given, and
  writes via a temporary file so a crash never leaves a half-written corpus.

Environment variables
---------------------
ENTREZ_EMAIL   (required by NCBI)   e.g. export ENTREZ_EMAIL="you@example.com"
NCBI_API_KEY   (optional, faster)   e.g. export NCBI_API_KEY="..."
"""

import argparse
import datetime as dt
import json
import os
import re
import ssl
import sys
import time
from collections import Counter
from pathlib import Path

from Bio import Entrez

from config import ABSTRACTS_FILE

# Load variables from a local .env file (if present) into the environment.
# .env is git-ignored, so real values never get committed: the code only ever
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

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

SSL_HELP = """\
Couldn't make a secure connection to PubMed: Python doesn't trust the SSL
certificate it received (CERTIFICATE_VERIFY_FAILED).

Common fixes:
  - macOS with Python from python.org: run "Install Certificates.command"
    (Applications > Python 3.x), or create the virtual environment with
    Homebrew's Python instead.
  - On a work or university network, a VPN, or with antivirus "web protection",
    HTTPS traffic may be intercepted. Try another network, or ask for the
    network's root certificate.
See "Troubleshooting" in the README for details."""


class FetchError(RuntimeError):
    """A PubMed request failed in a way that retrying won't fix."""


def _is_ssl_cert_error(exc):
    """True for certificate-verification failures (retrying can't fix these)."""
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLCertVerificationError) or isinstance(exc, ssl.SSLCertVerificationError)


def _explain(exc):
    """A readable one-paragraph explanation of a network error."""
    if _is_ssl_cert_error(exc):
        return SSL_HELP
    reason = getattr(exc, "reason", exc)
    return f"Couldn't reach PubMed ({reason}). Check your internet connection and try again."


# ── NCBI calls ───────────────────────────────────────────────────────────────
def _search(query, max_results, year=None, max_retries=3):
    """
    Run esearch with history enabled.

    If `year` is given, results are restricted to that publication year
    (datetype="pdat" = publication date).

    Returns (webenv, query_key, n_to_fetch, n_matched).
    """
    kwargs = dict(
        db="pubmed",
        term=query,
        retmax=0,            # we don't need the IDs inline; history holds them
        usehistory="y",
        sort="relevance",
    )
    if year is not None:
        kwargs.update(datetype="pdat", mindate=str(year), maxdate=str(year))

    for attempt in range(1, max_retries + 1):
        try:
            handle = Entrez.esearch(**kwargs)
            record = Entrez.read(handle)
            handle.close()
            break
        except Exception as e:  # noqa: BLE001 - network/parse errors both retry
            if _is_ssl_cert_error(e) or attempt == max_retries:
                raise FetchError(_explain(e)) from e
            wait = 2 ** attempt
            print(f"    ! search failed ({e}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)

    matched = int(record["Count"])
    return record["WebEnv"], record["QueryKey"], min(matched, max_results), matched


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
            if _is_ssl_cert_error(e):
                raise FetchError(_explain(e)) from e
            wait = 2 ** attempt
            print(f"    ! batch at {start} failed ({e}); retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)
    print(f"    ! giving up on batch at {start}")
    return None                      # None = failed (different from an empty batch)


# ── Parsing ──────────────────────────────────────────────────────────────────
def _extract_year(art):
    """
    Publication year of an article, trying three places in order:

    1. Journal issue date <PubDate><Year>       e.g. 2019
    2. Journal issue date <PubDate><MedlineDate> e.g. "2019 Dec-2020 Jan" -> 2019
    3. Electronic publication date <ArticleDate><Year>
    """
    try:
        pubdate = art["Journal"]["JournalIssue"]["PubDate"]
    except (KeyError, TypeError):
        pubdate = {}

    if "Year" in pubdate:
        try:
            return int(pubdate["Year"])
        except (ValueError, TypeError):
            pass

    match = _YEAR_RE.search(str(pubdate.get("MedlineDate", "")))
    if match:
        return int(match.group(0))

    for date in art.get("ArticleDate", []):
        if "Year" in date:
            try:
                return int(date["Year"])
            except (ValueError, TypeError):
                continue

    return None


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
        sections = []
        for p in parts:
            label = p.attributes.get("Label") if hasattr(p, "attributes") else None
            text = str(p).strip()
            sections.append(f"{label}: {text}" if label else text)
        abstract = " ".join(sections).strip()

    # Journal + year
    journal = str(art.get("Journal", {}).get("Title", "")).strip()
    year = _extract_year(art)

    # Authors: first three surnames, then "et al." if more.
    # (Collective/group authors have no LastName and are skipped.)
    authors = []
    for a in art.get("AuthorList", []):
        last = a.get("LastName")
        if last:
            authors.append(str(last))
    if len(authors) > 3:
        authors = authors[:3] + ["et al."]

    # Study design, e.g. "Randomized Controlled Trial", "Meta-Analysis", "Review".
    publication_types = [str(pt).strip() for pt in art.get("PublicationTypeList", [])]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": authors,
        "publication_types": publication_types,
    }


def _collect(webenv, query_key, count, batch_size, seen):
    """
    Fetch `count` records from a history-server result set and parse them.

    Skips records with an empty abstract and PMIDs already in `seen` (which is
    updated in place), so de-duplication works across several searches.
    Returns (articles, n_skipped_empty, n_skipped_duplicate, n_missing), where
    n_missing counts records in batches that failed even after retries.
    """
    articles, skipped_empty, skipped_dup, missing = [], 0, 0, 0
    for start in range(0, count, batch_size):
        # The last batch may be smaller: never ask for more than `count` records in total.
        n = min(batch_size, count - start)
        print(f"    fetching {start + 1}-{start + n} of {count} ...")
        batch = _fetch_batch(webenv, query_key, start, n)
        if batch is None:
            missing += n
            continue
        for article in batch:
            try:
                parsed = _parse_article(article)
            except Exception as e:  # noqa: BLE001 - one bad record shouldn't kill the run
                print(f"    ! skipped a record: {e}")
                continue
            if not parsed["abstract"]:
                skipped_empty += 1
                continue
            if parsed["pmid"] in seen:
                skipped_dup += 1
                continue
            seen.add(parsed["pmid"])
            articles.append(parsed)
        time.sleep(_SLEEP_BETWEEN_BATCHES)
    return articles, skipped_empty, skipped_dup, missing


# ── Output ───────────────────────────────────────────────────────────────────
def meta_path_for(out_path):
    """data/diabetes_abstracts.json -> data/diabetes_abstracts.meta.json"""
    out_path = Path(out_path)
    return out_path.with_name(out_path.stem + ".meta.json")


def _write_json_atomic(path, obj):
    """Write JSON via a temporary file, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _check_email():
    if Entrez.email == "your@email.com":
        print("WARNING: ENTREZ_EMAIL is not set. NCBI requires a real email; "
              "set it in .env or with `export ENTREZ_EMAIL=you@example.com`.")


def _save(articles, out_path, meta):
    """Save the corpus and its meta sidecar, and print a short summary."""
    meta["records"] = len(articles)
    meta["year_distribution"] = dict(sorted(
        Counter(a["year"] for a in articles if a["year"]).items()
    ))
    meta["records_without_year"] = sum(1 for a in articles if not a["year"])
    meta["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    _write_json_atomic(out_path, articles)
    _write_json_atomic(meta_path_for(out_path), meta)

    print(f"\nSaved {len(articles)} abstracts (with metadata) to {out_path}")
    print(f"Saved fetch details to {meta_path_for(out_path)}")

    missing = meta.get("missing_from_failed_batches", 0)
    if missing:
        print(f"\nWARNING: {missing} records could not be downloaded after several retries, "
              "so this corpus is INCOMPLETE.\nRe-run the same command with --overwrite "
              "(e.g. on a more stable connection).")


# ── Public API ───────────────────────────────────────────────────────────────
def fetch_pubmed(query=DEFAULT_QUERY, max_results=5000, batch_size=200, out_path=ABSTRACTS_FILE):
    """Single relevance-sorted search. Fetch up to `max_results` abstracts."""
    _check_email()
    print(f"Searching PubMed:\n  {query}\n")
    webenv, query_key, count, matched = _search(query, max_results)
    print(f"{matched} matching records; will fetch {count} in batches of {batch_size}.\n")

    articles, skipped_empty, skipped_dup, missing = _collect(webenv, query_key, count, batch_size, set())

    meta = {
        "mode": "single_query",
        "query": query,
        "sort": "relevance",
        "max_results": max_results,
        "matched": matched,
        "skipped_empty_abstract": skipped_empty,
        "skipped_duplicate": skipped_dup,
        "missing_from_failed_batches": missing,
    }
    _save(articles, out_path, meta)
    return articles


def fetch_pubmed_by_year(from_year, to_year, per_year=300, query=DEFAULT_QUERY,
                         batch_size=200, out_path=ABSTRACTS_FILE):
    """One search per publication year; take the top `per_year` papers of each year."""
    _check_email()
    print(f"Searching PubMed year by year, {from_year}-{to_year}, up to {per_year} per year:\n  {query}\n")

    seen, articles, per_year_stats = set(), [], {}
    for year in range(from_year, to_year + 1):
        webenv, query_key, count, matched = _search(query, per_year, year=year)
        print(f"  {year}: {matched} matching records, fetching {count}")
        year_articles, skipped_empty, skipped_dup, missing = _collect(
            webenv, query_key, count, batch_size, seen
        )
        articles.extend(year_articles)
        per_year_stats[str(year)] = {
            "matched": matched,
            "requested": count,
            "saved": len(year_articles),
            "skipped_empty_abstract": skipped_empty,
            "skipped_duplicate": skipped_dup,
            "missing_from_failed_batches": missing,
        }
        if len(year_articles) < per_year and matched >= per_year:
            print(f"    note: saved {len(year_articles)} (some had no abstract or were duplicates)")

    meta = {
        "mode": "per_year",
        "query": query,
        "sort": "relevance (within each year)",
        "date_type": "pdat (publication date)",
        "from_year": from_year,
        "to_year": to_year,
        "per_year": per_year,
        "per_year_stats": per_year_stats,
        "missing_from_failed_batches": sum(v["missing_from_failed_batches"] for v in per_year_stats.values()),
    }
    _save(articles, out_path, meta)
    return articles


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts with metadata.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query.")
    parser.add_argument("--max-results", type=int, default=5000,
                        help="Single-query mode: max abstracts to fetch.")
    parser.add_argument("--from-year", type=int, help="Per-year mode: first publication year.")
    parser.add_argument("--to-year", type=int, help="Per-year mode: last publication year.")
    parser.add_argument("--per-year", type=int, default=300,
                        help="Per-year mode: max abstracts per year (default 300).")
    parser.add_argument("--batch-size", type=int, default=200, help="Records per efetch call.")
    parser.add_argument("--out", default=str(ABSTRACTS_FILE), help="Output JSON path.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output file if it already exists.")
    args = parser.parse_args(argv)

    # Validate the per-year options
    if (args.from_year is None) != (args.to_year is None):
        parser.error("--from-year and --to-year must be used together.")
    if args.from_year is not None and args.from_year > args.to_year:
        parser.error("--from-year must not be after --to-year.")
    if args.per_year < 1 or args.max_results < 1 or args.batch_size < 1:
        parser.error("--per-year, --max-results and --batch-size must be positive.")

    # Protect an existing corpus from being replaced by accident
    if Path(args.out).exists() and not args.overwrite:
        print(f"{args.out} already exists. Rename it first (to keep it), or re-run with --overwrite.")
        sys.exit(1)

    try:
        _run(args)
    except FetchError as e:
        sys.exit(f"\n{e}")


def _run(args):
    if args.from_year is not None:
        fetch_pubmed_by_year(
            from_year=args.from_year,
            to_year=args.to_year,
            per_year=args.per_year,
            query=args.query,
            batch_size=args.batch_size,
            out_path=args.out,
        )
    else:
        fetch_pubmed(
            query=args.query,
            max_results=args.max_results,
            batch_size=args.batch_size,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()