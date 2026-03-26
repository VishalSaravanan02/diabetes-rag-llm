from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR          = ROOT_DIR / "data"
ABSTRACTS_FILE    = DATA_DIR / "diabetes_abstracts.json"
CHUNKS_FILE       = DATA_DIR / "chunks.json"
CHUNKS_PKL        = DATA_DIR / "chunks.pkl"
FAISS_INDEX_FILE  = DATA_DIR / "vector_index.faiss"

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama3"

# Chunking
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# Retrieval
TOP_K              = 5
DISTANCE_THRESHOLD = 1.5