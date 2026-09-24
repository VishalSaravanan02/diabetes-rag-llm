# Decisions Log

## D1: Corpus coverage (2026-09-25)
**Decision:** Per-year balanced fetch, 2010-2026, top 300 papers per year by relevance.
**Why:** The previous single-query fetch was 99% from 2026, because PubMed's Best Match
ranking favours recent papers and the fetch stopped at 5,000. That left the knowledge base
without foundational literature, which basic questions depend on.
**Result:** 5,089 abstracts (11 skipped: no abstract or duplicate), 98.5% mention diabetes,
0 duplicate PMIDs, every year 2010-2026 represented (207-355 papers per year by issue year).
Includes 159 meta-analyses, 171 systematic reviews, 274 RCTs, 754 reviews.
**Note:** Searches use publication date (includes online-first), while the saved `year` is the
journal issue year, so per-year counts in the corpus differ slightly from 300.
