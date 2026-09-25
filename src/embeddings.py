"""
Embed every chunk and build the FAISS search index.

Reads   data/chunks.json            (written by src/preprocess.py)
Writes  data/vector_index.faiss     the vectors
        data/index_meta.json        what the index was built from

Similarity: embeddings are L2-normalised and stored in an IndexFlatIP (inner
product). For unit-length vectors, inner product == cosine similarity, so
search scores fall in [-1, 1] and higher means more relevant.

index_meta.json records the embedding model, the EMBED_WITH_TITLE setting and a
SHA-256 fingerprint of chunks.json. The retriever checks these on load and
refuses to search a stale index (see src/store.py).

Usage (from the project root):
    python -m src.embeddings
"""

import datetime as dt
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNKS_FILE,
    EMBED_WITH_TITLE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    INDEX_META_FILE,
)
from src.store import embedding_input, file_sha256, load_chunks


def build_index(batch_size=64):
    chunks, chunks_meta = load_chunks()
    if not chunks:
        raise ValueError(f"{CHUNKS_FILE} contains no chunks. Run `python -m src.preprocess` first.")

    texts = [embedding_input(c, EMBED_WITH_TITLE) for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL} "
          f"(title prefix: {'on' if EMBED_WITH_TITLE else 'off'}) ...")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit length, so inner product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "embed_with_title": EMBED_WITH_TITLE,
        "dim": int(embeddings.shape[1]),
        "n_vectors": int(index.ntotal),
        "metric": "cosine (IndexFlatIP on L2-normalised vectors)",
        "chunks_sha256": file_sha256(CHUNKS_FILE),
        "chunks_meta": chunks_meta,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nIndex: {index.ntotal} vectors of dimension {meta['dim']} -> {FAISS_INDEX_FILE}")
    print(f"Index metadata -> {INDEX_META_FILE}")
    return index, meta


if __name__ == "__main__":
    build_index()
