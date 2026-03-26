# 🧬 Diabetes RAG LLM

A Retrieval-Augmented Generation (RAG) system that answers biomedical questions about diabetes using research abstracts fetched from PubMed. It combines semantic search via FAISS with a locally-running LLM through Ollama, and exposes the system through an interactive Streamlit web interface.

---

## 📌 Overview

This project implements a full RAG pipeline:

1. **Fetch** — PubMed abstracts on diabetes are retrieved using the NCBI Entrez API
2. **Preprocess** — Abstracts are cleaned and chunked into overlapping segments for better retrieval
3. **Embed** — Chunks are encoded using `all-MiniLM-L6-v2` and stored in a FAISS vector index
4. **Retrieve** — At query time, the most semantically relevant chunks are retrieved with distance-based filtering
5. **Generate** — A local LLM (LLaMA via Ollama) generates a grounded answer using the retrieved context
6. **Interface** — A Streamlit web app provides a clean UI for querying the system, with source citations

---

## 🗂️ Project Structure

```
diabetes-rag-llm/
│
├── app.py                  # Streamlit web interface
├── main.py                 # CLI entry point for the RAG QA system
├── config.py               # Centralised configuration (paths, models, parameters)
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── fetch_data.py       # Fetches abstracts from PubMed via Entrez
│   ├── preprocess.py       # Cleans and chunks abstracts using LangChain text splitter
│   ├── embeddings.py       # Generates embeddings and builds FAISS index
│   ├── retriever.py        # Loads FAISS index and retrieves relevant chunks
│   └── generator.py        # Generates answers using Ollama (LLaMA 3)
│
└── data/
    ├── diabetes_abstracts.json   # Raw PubMed abstracts (generated)
    ├── chunks.json               # Preprocessed text chunks (generated)
    ├── chunks.pkl                # Serialised chunks for retrieval (generated)
    └── vector_index.faiss        # FAISS vector index (generated)
```

---

## ⚙️ Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- LLaMA 3 pulled via Ollama:

```bash
ollama pull llama3
```

> **Note:** `all-MiniLM-L6-v2` is used for embeddings via `sentence-transformers` (installed as a Python package — no Ollama pull needed).

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/VishalSaravanan02/diabetes-rag-llm.git
cd diabetes-rag-llm
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your NCBI Entrez email

NCBI requires a valid email for API access. Set it as an environment variable:

```bash
export ENTREZ_EMAIL="your@email.com"   # macOS/Linux
set ENTREZ_EMAIL="your@email.com"      # Windows
```

---

## 🔧 Building the Knowledge Base

Run these steps once to fetch data and build the vector index. Always run from the **project root** using `-m`:

### Step 1 — Fetch PubMed abstracts

```bash
python -m src.fetch_data
```

Fetches 50 PubMed abstracts about diabetes and saves them to `data/diabetes_abstracts.json`.

### Step 2 — Preprocess and chunk the abstracts

```bash
python -m src.preprocess
```

Cleans and splits abstracts into 500-character chunks with 100-character overlap, saved to `data/chunks.json`.

### Step 3 — Generate embeddings and build FAISS index

```bash
python -m src.embeddings
```

Encodes all chunks using `all-MiniLM-L6-v2` and saves the FAISS index to `data/vector_index.faiss`.

---

## 💬 Running the Application

### Option A — Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Features:
- Ask biomedical questions grounded in PubMed research
- Adjust the number of retrieved sources using the slider
- View the retrieved PubMed chunks that informed the answer, with relevance distance scores

### Option B — CLI Interactive Mode

```bash
python main.py
```

Type your question at the prompt and press Enter. Type `exit` or `quit` to stop.

---

## ⚙️ Configuration

All key settings are centralised in `config.py` at the project root:

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `"llama3"` | Ollama model used for generation |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Sentence transformer model for embeddings |
| `CHUNK_SIZE` | `500` | Character size of each text chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `DISTANCE_THRESHOLD` | `1.5` | Max L2 distance for a chunk to be considered relevant |

To switch models or tune retrieval, edit `config.py` — changes apply across the whole project automatically.

> **Tip:** If answers seem too brief or retrieval returns no results, try increasing `TOP_K` or `DISTANCE_THRESHOLD` in `config.py`.

---

## 🧪 Example Query

**Question:** What is diabetes?

**Answer:** *(Generated from retrieved PubMed context using LLaMA 3)*

According to the context, Diabetes mellitus (DM) is a major contributor to disability and mortality, accounting for nearly 10% of all deaths in people aged 20 to 79 years.

**Retrieved Sources:** 5 chunks retrieved from the knowledge base, each with a relevance distance score shown in the UI.

> **Note:** Answers are grounded strictly in the retrieved PubMed abstracts. The richer and larger the knowledge base, the more comprehensive the answers. Consider increasing `max_results` in `fetch_data.py` for broader coverage.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Data source | PubMed via NCBI Entrez (`Biopython`) |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`faiss-cpu`) |
| LLM | LLaMA 3 via Ollama |
| Web interface | Streamlit |

---

## 🗺️ Roadmap

- [x] Build full RAG pipeline (fetch → preprocess → embed → retrieve → generate)
- [x] Create `config.py` to centralise model names, paths, and parameters
- [x] Fix `app.py` to use the full RAG pipeline instead of direct Ollama calls
- [x] Consolidate generation logic into `generator.py`
- [x] Standardise path handling across all `src/` files
- [x] Add distance threshold filtering in `retriever.py`
- [x] Add source chunk display in the Streamlit UI
- [x] Add light text cleaning in `preprocess.py`
- [x] Add batch fetching in `fetch_data.py`
- [ ] Add PMID metadata through the full pipeline for proper source attribution
- [ ] Increase abstract coverage (fetch 200+ abstracts for richer answers)
- [ ] Explore PubMed Central (PMC) full-text articles for deeper context
- [ ] Add streaming responses in the Streamlit UI

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Vishal Saravanan**
[GitHub](https://github.com/VishalSaravanan02)