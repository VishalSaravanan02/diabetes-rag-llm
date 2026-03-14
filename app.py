import streamlit as st
from retriever import retrieve
from generator import answer_question

st.title("Medical Query Assistant (RAG)")

query = st.text_input("Ask a medical question:")
if query:
    st.write("Searching PubMed...")
    context = retrieve(query, "diabetes")
    answer = answer_question(query, context)
    st.write("**Answer:**")
    st.write(answer)
    st.write("**Sources:**")
    for c in context:
        st.write("-", c[:150], "...")  # Show first 150 chars of each chunk