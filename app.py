# app.py
from src.retriever import retrieve  # your FAISS + embeddings retriever
import ollama  # Python client for local Ollama

# -----------------------------
# 1. Choose your local model
# -----------------------------
MODEL_NAME = "llama2"  # change if you pulled a different model locally

# -----------------------------
# 2. Generate answer function
# -----------------------------
def generate_answer(query, top_k=5):
    """
    Retrieve top-k relevant chunks and generate an answer
    using a local LLM (Ollama).
    """
    # Step 1: Retrieve top relevant chunks
    top_chunks = retrieve(query, top_k=top_k)

    # Step 2: Combine chunks into context
    context = "\n\n".join(top_chunks)

    # Step 3: Create the prompt
    prompt = (
        f"You are a helpful medical research assistant. "
        f"Answer the question using only the context below. "
        f"If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # Step 4: Call Ollama generate
    response = ollama.generate(model=MODEL_NAME, prompt=prompt)

    # ✔️ Use .response to get the generated text
    return response.response

# -----------------------------
# 3. Run interactive QA app
# -----------------------------
if __name__ == "__main__":
    print("=== PubMed RAG QA System (Local LLM) ===")

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() in ("exit", "quit"):
            print("Exiting app...")
            break

        try:
            answer = generate_answer(user_query, top_k=5)
            print("\n=== Generated Answer ===\n")
            print(answer)
        except Exception as e:
            print(f"Error generating answer: {e}")