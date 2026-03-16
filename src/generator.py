import ollama

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

    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]