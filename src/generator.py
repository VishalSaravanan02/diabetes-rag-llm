# generator.py

from dotenv import load_dotenv
import os
from openai import OpenAI
#from openai.error import OpenAIError
from retriever import retrieve

# -----------------------------
# 1. Load .env and initialize OpenAI
# -----------------------------
load_dotenv()  # Loads .env file automatically

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found! Did you create a .env file with your key?")

client = OpenAI(api_key=api_key)

# -----------------------------
# 2. Function to generate answer
# -----------------------------
def generate_answer(query, top_k=5):
    """
    Generate an answer using top-k retrieved chunks from PubMed abstracts.
    """
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)
    context = "\n\n".join(chunks)

    # Step 2: Prepare messages for Chat API
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": f"""
Using ONLY the following PubMed abstracts, answer the question clearly.
Do not add any information not present in the abstracts.

Context:
{context}

Question:
{query}

Answer:
"""}
    ]

    # Step 3: Call OpenAI Chat API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("OpenAI API error:", e)
        return "Sorry, could not generate an answer."

# -----------------------------
# 3. Test the generator
# -----------------------------
if __name__ == "__main__":
    question = "What are the genetic causes of hyperinsulinism?"
    answer = generate_answer(question)
    print("\nAnswer:\n")
    print(answer)