"""
Shared helpers for reading the chunks file and checking the search index.

Both src/embeddings.py (which builds the index) and src/retriever.py (which
searches it) go through these functions, so they always agree on file formats.

The consistency check exists to prevent one specific, silent bug: searching an
index that was built from different chunks or a different embedding model than
the ones currently configured. That returns confident-looking but wrong sources,
so instead we stop with a clear message saying what to rebuild.
"""

import hashlib
import json

from config import (
    CHUNKS_FILE,
    EMBED_WITH_TITLE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    INDEX_META_FILE,
)

REBUILD_HINT = "Rebuild with:  python -m src.preprocess  &&  python -m src.embeddings"


class IndexNotReadyError(RuntimeError):
    """The index is missing, or doesn't match the current chunks/config."""


def load_chunks(path=CHUNKS_FILE):
    """
    Load chunks.json. Returns (chunks, meta).

    Supports the current format {"meta": {...}, "chunks": [...]} and the old
    flat-list format (returned with an empty meta).
    """
    if not path.exists():
        raise IndexNotReadyError(f"Chunks file not found at {path}. {REBUILD_HINT}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data["chunks"], data.get("meta", {})


def file_sha256(path):
    """Fingerprint of a file: changes if even one byte of the file changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def embedding_input(chunk, with_title=EMBED_WITH_TITLE):
    """The exact text that gets embedded for a chunk."""
    if with_title and chunk.get("title"):
        return f"{chunk['title']}\n\n{chunk['text']}"
    return chunk["text"]


def read_index_meta(path=INDEX_META_FILE):
    if not path.exists():
        raise IndexNotReadyError(f"Index metadata not found at {path}. {REBUILD_HINT}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_index_consistency(index, chunks, meta):
    """
    Raise IndexNotReadyError if the loaded index doesn't match the chunks file
    or the current config. Returns None when everything lines up.
    """
    problems = []

    if index.ntotal != len(chunks):
        problems.append(f"index has {index.ntotal} vectors but chunks.json has {len(chunks)} chunks")
    if meta.get("n_vectors") != index.ntotal:
        problems.append("index_meta.json doesn't describe this index file")
    if meta.get("chunks_sha256") != file_sha256(CHUNKS_FILE):
        problems.append("chunks.json has changed since the index was built")
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        problems.append(
            f"index was built with '{meta.get('embedding_model')}' "
            f"but config.EMBEDDING_MODEL is '{EMBEDDING_MODEL}'"
        )
    if meta.get("embed_with_title") != EMBED_WITH_TITLE:
        problems.append(
            f"index was built with EMBED_WITH_TITLE={meta.get('embed_with_title')} "
            f"but config says {EMBED_WITH_TITLE}"
        )

    if problems:
        details = "\n  - ".join(problems)
        raise IndexNotReadyError(f"The search index is out of date:\n  - {details}\n{REBUILD_HINT}")


def index_files_exist():
    return FAISS_INDEX_FILE.exists() and INDEX_META_FILE.exists() and CHUNKS_FILE.exists()
