import ollama
from config import LLM_MODEL

def generate_answer(question, context):
    prompt = f"""
    You are a biomedical research assistant.

    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"