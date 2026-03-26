from src.retriever import retrieve
from src.generator import generate_answer
from config import TOP_K


def answer_question(query, top_k=TOP_K):
    """
    Retrieve relevant chunks from the FAISS index and generate
    a grounded answer using the local LLM.

    Args:
        query:  The user's question
        top_k:  Number of chunks to retrieve (default from config)

    Returns:
        A tuple of (answer, sources) where sources is a list of PMIDs
    """

    # Retrieve relevant chunks
    results = retrieve(query, top_k=top_k)

    if not results:
        return "I couldn't find any relevant information in the knowledge base for that question.", []

    # Build context from retrieved chunks
    context = "\n\n".join([r["chunk"] for r in results])

    # Generate answer
    answer = generate_answer(query, context)

    return answer


if __name__ == "__main__":
    print("=== PubMed RAG QA System (Local LLM) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("\nEnter your question: ").strip()

        if not user_query:
            continue

        if user_query.lower() in ("exit", "quit"):
            print("Exiting app...")
            break

        try:
            answer = answer_question(user_query, top_k=TOP_K)
            print("\n=== Generated Answer ===\n")
            print(answer)
        except Exception as e:
            print(f"Error: {str(e)}")