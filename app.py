import streamlit as st
from src.retriever import retrieve
from src.generator import generate_answer
from config import TOP_K

# Page config
st.set_page_config(
    page_title="Diabetes RAG LLM",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Diabetes RAG LLM")
st.write("Ask biomedical questions and get answers grounded in PubMed research.")
st.markdown("---")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What are the genetic causes of Type 1 diabetes?"
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Ask", use_container_width=True)
with col2:
    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=TOP_K)

# RAG Pipeline
if submit and query:

    # Step 1 — Retrieve relevant chunks
    with st.spinner("Searching PubMed knowledge base..."):
        results = retrieve(query, top_k=top_k)

    if not results:
        st.warning(
            "No relevant information found in the knowledge base for that question. "
            "Try rephrasing or asking something more specific to diabetes research."
        )
    else:
        # Step 2 — Build context from retrieved chunks
        context = "\n\n".join([r["chunk"] for r in results])

        # Step 3 — Generate answer
        with st.spinner("Generating answer from retrieved context..."):
            answer = generate_answer(query, context)

        # Display answer
        st.markdown("### 💬 Answer")
        st.success(answer)

        # Display retrieved sources
        st.markdown("### 📚 Retrieved Sources")
        st.caption(f"{len(results)} chunk(s) retrieved from the knowledge base.")

        for i, result in enumerate(results, 1):
            pmid = result["pmid"]
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

            with st.expander(f"Source {i} — PMID: {pmid} — Relevance distance: {result['distance']:.4f}"):
                st.write(result["chunk"])
                st.markdown(f"🔗 [View on PubMed]({pubmed_url})")

elif submit and not query:
    st.warning("Please enter a question before clicking Ask.")

# Footer
st.markdown("---")
st.markdown("💡 Built with Streamlit · FAISS · Sentence Transformers · Ollama")