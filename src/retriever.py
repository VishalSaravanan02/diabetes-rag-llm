from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data"


index = faiss.read_index(str(VECTOR_STORE_DIR / "vector_index.faiss"))


with open(VECTOR_STORE_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks and FAISS index with {index.ntotal} vectors.")


model = SentenceTransformer('all-MiniLM-L6-v2')


def retrieve(query, top_k=5):
    """
    Retrieve top-k most relevant chunks for a query.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":
    user_query = "What are the genetic causes of hyperinsulinism?"
    top_chunks = retrieve(user_query, top_k=5)

    print("\nTop relevant chunks:\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"{i}. {chunk}\n")
