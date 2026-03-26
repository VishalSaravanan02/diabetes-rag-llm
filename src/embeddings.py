import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from config import CHUNKS_FILE, FAISS_INDEX_FILE, CHUNKS_PKL, EMBEDDING_MODEL


def build_index():
    """
    Load chunks, generate embeddings, build a FAISS index, and save to disk.
    Chunks are stored as dicts with 'text' and 'pmid' keys.
    """

    # Load chunks
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_FILE}. "
            "Please run preprocess.py first."
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError("Chunks file is empty. Please re-run preprocess.py.")

    print(f"Total chunks loaded: {len(chunks)}")

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"Number of vectors in FAISS index: {index.ntotal}")

    # Save index and chunks to disk
    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"FAISS index saved to {FAISS_INDEX_FILE}")

    with open(CHUNKS_PKL, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {CHUNKS_PKL}")

    print("\nDone! Knowledge base is ready.")


if __name__ == "__main__":
    build_index()