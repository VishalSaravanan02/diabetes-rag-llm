"""
Search the FAISS index for the chunks most relevant to a question.

Scores are cosine similarities in [-1, 1]: higher = more relevant. Results
below config.MIN_SIMILARITY are dropped, so a clearly off-topic question
returns nothing instead of the "least bad" chunks.

On first use, the index, chunks and embedding model are loaded once and kept
in memory. The index is checked against chunks.json and config first (see
src/store.py), so a stale index fails loudly instead of returning wrong sources.

Usage (from the project root):
    python -m src.retriever "What is HbA1c?"
    python -m src.retriever                      # interactive
"""

import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FAISS_INDEX_FILE, MIN_SIMILARITY, TOP_K
from src.store import (
    REBUILD_HINT,
    IndexNotReadyError,
    check_index_consistency,
    load_chunks,
    read_index_meta,
)

# Fields copied from each chunk into every search result
RESULT_FIELDS = (
    "text", "pmid", "title", "journal", "year", "authors", "authors_display",
    "publication_types", "chunk_index", "n_chunks_in_source",
)

# Lazy-loaded globals
_index = None
_chunks = None
_model = None
_meta = None


def _load():
    """Load the index, chunks and model once; later calls reuse them."""
    global _index, _chunks, _model, _meta
    if _index is not None:
        return

    if not FAISS_INDEX_FILE.exists():
        raise IndexNotReadyError(f"FAISS index not found at {FAISS_INDEX_FILE}. {REBUILD_HINT}")

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    chunks, _ = load_chunks()
    meta = read_index_meta()
    check_index_consistency(index, chunks, meta)

    _model = SentenceTransformer(EMBEDDING_MODEL)
    _index, _chunks, _meta = index, chunks, meta
    print(f"Loaded index: {_index.ntotal} chunks, model {EMBEDDING_MODEL}.")


def index_info():
    """Metadata about the loaded index (for the app's sidebar, eval run records, ...)."""
    _load()
    return dict(_meta)


def retrieve_candidates(query, n):
    """
    The top-n chunks for `query`, best first, with NO similarity threshold.
    Used by evaluation, threshold calibration and (later) hybrid search.
    """
    _load()
    n = min(n, _index.ntotal)           # FAISS pads with -1 if n > number of vectors
    if n <= 0:
        return []

    q = _model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, ids = _index.search(q, n)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:                     # padding slot, not a real chunk
            continue
        chunk = _chunks[idx]
        result = {field: chunk.get(field) for field in RESULT_FIELDS}
        result["score"] = float(score)
        results.append(result)
    return results


def retrieve(query, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
    """The top_k chunks for `query` whose cosine similarity is at least `min_similarity`."""
    return [r for r in retrieve_candidates(query, top_k) if r["score"] >= min_similarity]


def _print_results(query):
    results = retrieve_candidates(query, TOP_K)
    print(f"\nQ: {query}")
    if not results:
        print("  (index is empty)")
    for i, r in enumerate(results, 1):
        flag = "" if r["score"] >= MIN_SIMILARITY else "   [below threshold]"
        print(f"  {i}. score {r['score']:.3f}  PMID {r['pmid']}  ({r['year']}){flag}")
        print(f"     {(r['title'] or '')[:95]}")


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            _print_results(" ".join(sys.argv[1:]))
        else:
            print("Type a question (or 'exit').")
            while (q := input("\n> ").strip()).lower() not in ("exit", "quit"):
                if q:
                    _print_results(q)
    except IndexNotReadyError as e:
        sys.exit(str(e))
