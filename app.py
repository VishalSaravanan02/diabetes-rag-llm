import streamlit as st
import ollama

st.set_page_config(page_title="Diabetes RAG LLM", page_icon="🧬")
st.title("Diabetes RAG LLM")
st.write("Ask questions and get answers from your RAG model!")

query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": query}]
        )
        st.success(response["message"]["content"])
    except Exception as e:
        st.error(f"Error getting response: {e}")

st.markdown("---")
st.markdown("💡 Built with Streamlit + Ollama")

