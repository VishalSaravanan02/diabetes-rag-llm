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

## D2: Drop chunks.pkl (2026-09-25)
**Decision:** `chunks.json` is the single source of truth; the pickle copy is removed.
**Why:** One human-readable file instead of two copies that can drift apart; no pickle loading.

## D3: Author format on chunks (2026-09-25)
**Decision:** Keep `authors` as a list and add `authors_display` (e.g. "Smith, Lee, Patel et al.").
**Why:** The list is data for later processing; the string is ready for the UI and the LLM prompt.

## D4: Similarity metric (2026-09-25)
**Decision:** Cosine similarity: L2-normalised embeddings in a FAISS `IndexFlatIP`.
**Why:** Scores fall in [-1, 1] with higher = more relevant, which is easy to threshold and explain.
The old setup showed users a squared L2 "distance" where lower was better.

## D4b: MIN_SIMILARITY = 0.47 (2026-09-25)
**Decision:** Set the relevance threshold to 0.47 for all-MiniLM-L6-v2 (title prefix off).
**Evidence:** `scripts/calibrate_threshold.py` on 20 on-topic and 10 off-topic questions.
On-topic top-1 scores: 0.649-0.843. Off-topic: 0.201-0.298. The groups are cleanly separated;
0.47 is the midpoint of the gap. The previous value (0.30) passed "best pizza in Naples"
by only 0.002.
**Revisit:** whenever the embedding model or EMBED_WITH_TITLE changes, and in Phase 5 with
the unanswerable-question set.
