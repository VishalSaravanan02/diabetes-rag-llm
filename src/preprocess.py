"""
Clean, chunk, and serialise PubMed abstracts for indexing.

What this does
--------------
- Loads the fetched abstracts (with metadata) from ABSTRACTS_FILE.
- Drops abstracts shorter than MIN_ABSTRACT_LENGTH (removes stubs/placeholders
  like the handful of 5-char "abstracts" PubMed occasionally returns).
- Splits each abstract into chunks using one of two strategies (config flag):
    * "recursive" — LangChain RecursiveCharacterTextSplitter (character-based).
    * "sentence"  — sentence-aware packing: split into sentences, then pack
                    them up to CHUNK_SIZE chars with CHUNK_OVERLAP overlap so
                    chunks never cut a sentence in half.
- Carries full source metadata (pmid, title, journal, year, authors, study
  types) onto EVERY chunk, so downstream retrieval can build rich citations
  ("Smith, Lee, Patel et al., Diabetes Care, 2016") and evidence badges.
- Writes a self-describing JSON file: a `meta` header recording the exact build
  parameters, plus the `chunks` list. The header makes every eval run traceable
  back to the chunking config that produced it.

Output schema (CHUNKS_FILE)
---------------------------
{
  "meta": { "chunk_strategy": ..., "chunk_size": ..., "total_chunks": ..., ... },
  "chunks": [
    {"text": ..., "pmid": ..., "title": ..., "journal": ..., "year": ...,
     "authors": ["Smith", "Lee", "Patel", "et al."],
     "authors_display": "Smith, Lee, Patel et al.",
     "publication_types": ["Journal Article", "Randomized Controlled Trial"],
     "chunk_index": 0, "n_chunks_in_source": 3},
    ...
  ]
}
"""

import json
import re
from datetime import datetime, timezone

# LangChain 1.x moved the splitters into their own package; the old
# `langchain.text_splitter` path no longer exists, so there is no fallback.
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    ABSTRACTS_FILE,
    CHUNKS_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_STRATEGY,
    MIN_ABSTRACT_LENGTH,
    EMBEDDING_MODEL,
)


def format_authors(authors):
    """["Smith", "Lee", "Patel", "et al."] -> "Smith, Lee, Patel et al." """
    authors = [a for a in (authors or []) if a]
    if not authors:
        return ""
    if authors[-1] == "et al.":
        return ", ".join(authors[:-1]) + " et al."
    return ", ".join(authors)


def _fetch_meta():
    """Read the fetch record written next to the abstracts file, if present."""
    meta_file = ABSTRACTS_FILE.with_name(ABSTRACTS_FILE.stem + ".meta.json")
    if not meta_file.exists():
        return {}
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    keep = ("mode", "query", "from_year", "to_year", "per_year", "records", "fetched_at")
    return {k: meta[k] for k in keep if k in meta}


def clean_text(text):
    """Collapse whitespace/newlines to single spaces and strip the ends."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Sentence-aware chunking ──────────────────────────────────────────────────

# Common abbreviations whose trailing period should NOT end a sentence.
_ABBREVIATIONS = {
    "e.g.", "i.e.", "vs.", "etc.", "cf.", "al.", "Dr.", "Fig.", "No.",
    "approx.", "ca.", "kg.", "mg.", "mL.", "wk.", "yr.", "mo.",
}


def _protect_abbreviations(text):
    """Temporarily mask abbreviation periods so the splitter won't break on them."""
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "<PRD>"))
    # Also protect decimals like "3.5" and single capital initials like "S. aureus"
    text = re.sub(r"(\d)\.(\d)", r"\1<PRD>\2", text)
    text = re.sub(r"\b([A-Z])\.", r"\1<PRD>", text)
    return text


def split_into_sentences(text):
    """Lightweight, dependency-free sentence splitter tuned for abstracts."""
    protected = _protect_abbreviations(text)
    # Split on ., !, or ? followed by whitespace.
    pieces = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [p.replace("<PRD>", ".").strip() for p in pieces if p.strip()]
    return sentences


def _pack_sentences(sentences, chunk_size, overlap):
    """Greedily pack sentences into chunks up to chunk_size chars, with overlap.

    Overlap is applied by carrying trailing sentences from the previous chunk
    into the next one until roughly `overlap` characters are re-covered.
    """
    chunks = []
    current, current_len = [], 0

    for sent in sentences:
        # +1 accounts for the joining space
        if current and current_len + len(sent) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Build overlap tail from the end of the chunk we just closed.
            tail, tail_len = [], 0
            for s in reversed(current):
                if tail_len + len(s) + 1 > overlap:
                    break
                tail.insert(0, s)
                tail_len += len(s) + 1
            current, current_len = tail[:], tail_len
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Main pipeline ────────────────────────────────────────────────────────────

def _chunk_text(text, strategy, splitter):
    if strategy == "sentence":
        return _pack_sentences(split_into_sentences(text), CHUNK_SIZE, CHUNK_OVERLAP)
    return splitter.split_text(text)  # "recursive"


def preprocess():
    if not ABSTRACTS_FILE.exists():
        raise FileNotFoundError(
            f"Abstracts file not found at {ABSTRACTS_FILE}. Run fetch_data.py first."
        )

    with open(ABSTRACTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total papers loaded: {len(data)}")
    print(f"Chunking strategy: {CHUNK_STRATEGY} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []
    used, skipped_short = 0, 0

    for item in data:
        abstract = item.get("abstract", "")
        if not abstract or len(abstract) < MIN_ABSTRACT_LENGTH:
            skipped_short += 1
            continue

        used += 1
        cleaned = clean_text(abstract)
        pieces = _chunk_text(cleaned, CHUNK_STRATEGY, splitter)

        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "pmid": item.get("pmid"),
                "title": item.get("title"),
                "journal": item.get("journal"),
                "year": item.get("year"),
                "authors": item.get("authors", []),
                "authors_display": format_authors(item.get("authors")),
                "publication_types": item.get("publication_types", []),
                "chunk_index": i,
                "n_chunks_in_source": len(pieces),
            })

    print(f"Abstracts used: {used}   | skipped (too short): {skipped_short}")
    print(f"Total chunks created: {len(all_chunks)}")

    output = {
        "meta": {
            "source_abstracts": len(data),
            "abstracts_used": used,
            "abstracts_skipped_short": skipped_short,
            "min_abstract_length": MIN_ABSTRACT_LENGTH,
            "chunk_strategy": CHUNK_STRATEGY,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "total_chunks": len(all_chunks),
            "embedding_model_hint": EMBEDDING_MODEL,
            "source_fetch": _fetch_meta(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "chunks": all_chunks,
    }

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to {CHUNKS_FILE}")
    return output


if __name__ == "__main__":
    preprocess()