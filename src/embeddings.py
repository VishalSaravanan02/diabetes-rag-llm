import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

# -----------------------------
# 1. Load the chunks
# -----------------------------
CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks loaded: {len(chunks)}")

# -----------------------------
# 2. Initialize embedding model
# -----------------------------
# You can use any sentence-transformers model you like
# 'all-MiniLM-L6-v2' is small and fast
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 3. Generate embeddings
# -----------------------------
print("Generating embeddings...")
embeddings = model.encode(chunks, show_progress_bar=True)

print(f"Embeddings shape: {embeddings.shape}")

# -----------------------------
# 4. Create FAISS vector store
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)

print(f"Number of vectors in FAISS index: {index.ntotal}")

# -----------------------------
# 5. Save FAISS index and chunks
# -----------------------------
VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data"
faiss.write_index(index, str(VECTOR_STORE_DIR / "vector_index.faiss"))

# Save chunks as reference
with open(VECTOR_STORE_DIR / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved in data/")