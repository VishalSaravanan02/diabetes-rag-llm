from sentence_transformers import SentenceTransformer
import faiss
import pickle
from config import FAISS_INDEX_FILE, CHUNKS_PKL, EMBEDDING_MODEL, TOP_K, DISTANCE_THRESHOLD

# Lazy-loaded globals
_index  = None
_chunks = None
_model  = None


def _load():
    """
    Lazily load the FAISS index, chunks, and embedding model.
    Only runs once — subsequent calls reuse the loaded globals.
    """
    global _index, _chunks, _model

    if _index is not None:
        return

    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_FILE}. "
            "Please run embeddings.py first to build the index."
        )

    if not CHUNKS_PKL.exists():
        raise FileNotFoundError(
            f"Chunks file not found at {CHUNKS_PKL}. "
            "Please run embeddings.py first to build the index."
        )

    print("Loading FAISS index and chunks...")
    _index = faiss.read_index(str(FAISS_INDEX_FILE))

    with open(CHUNKS_PKL, "rb") as f:
        _chunks = pickle.load(f)

    print("Loading embedding model...")
    _model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loaded {len(_chunks)} chunks and FAISS index with {_index.ntotal} vectors.")


def retrieve(query, top_k=TOP_K):
    """
    Retrieve top-k most relevant chunks for a query.
    Filters out chunks whose L2 distance exceeds DISTANCE_THRESHOLD.

    Returns a list of dicts with keys: 'chunk', 'pmid' and 'distance'.
    """
    _load()

    query_embedding = _model.encode([query])
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if dist <= DISTANCE_THRESHOLD:
            results.append({
                "chunk": _chunks[idx]["text"],
                "pmid": _chunks[idx]["pmid"],
                "distance": float(dist)
            })

    return results


if __name__ == "__main__":
    user_query = "What are the genetic causes of hyperinsulinism?"
    top_chunks = retrieve(user_query, top_k=TOP_K)

    print("\nTop relevant chunks:\n")
    for i, result in enumerate(top_chunks, 1):
        print(f"{i}. [Distance: {result['distance']:.4f}]\n{result['chunk']}\n")