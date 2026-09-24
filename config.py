from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR          = ROOT_DIR / "data"
ABSTRACTS_FILE    = DATA_DIR / "diabetes_abstracts.json"
CHUNKS_FILE       = DATA_DIR / "chunks.json"
FAISS_INDEX_FILE  = DATA_DIR / "vector_index.faiss"
INDEX_META_FILE   = DATA_DIR / "index_meta.json"   # what the index was built from

# Models
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBED_WITH_TITLE = False     # embed "title + chunk" instead of just the chunk (Phase 6 experiment)
LLM_MODEL        = "llama3"
LLM_TEMPERATURE  = 0.0       # 0 = same question, same answer (reproducible)

# Chunking
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 100
CHUNK_STRATEGY      = "recursive"   # "recursive" | "sentence"
MIN_ABSTRACT_LENGTH = 200           # drop abstracts shorter than this (stubs/placeholders)

# Retrieval
TOP_K          = 5
MIN_SIMILARITY = 0.47   # cosine similarity (higher = more relevant); calibrate with
                        # scripts/calibrate_threshold.py whenever the embedding model changes
