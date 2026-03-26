# 🧬 Diabetes RAG LLM

A Retrieval-Augmented Generation (RAG) system that answers biomedical questions about diabetes using research abstracts fetched from PubMed. It combines semantic search via FAISS with a locally-running LLM through Ollama, and exposes the system through an interactive Streamlit web interface.

---

## 📌 Overview

This project implements a full RAG pipeline:

1. **Fetch** — PubMed abstracts on diabetes are retrieved using the NCBI Entrez API
2. **Preprocess** — Abstracts are cleaned and chunked into overlapping segments, with PMID metadata attached to every chunk
3. **Embed** — Chunks are encoded using `all-MiniLM-L6-v2` and stored in a FAISS vector index
4. **Retrieve** — At query time, the most semantically relevant chunks are retrieved with distance-based filtering
5. **Generate** — A local LLM (LLaMA via Ollama) generates a grounded answer using the retrieved context
6. **Interface** — A Streamlit web app displays the answer alongside cited PubMed sources with clickable links
7. **Evaluate** — A RAGAS evaluation pipeline measures pipeline quality using 4 metrics with local Hugging Face models

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
│   ├── preprocess.py       # Cleans and chunks abstracts, attaches PMID to each chunk
│   ├── embeddings.py       # Generates embeddings and builds FAISS index
│   ├── retriever.py        # Loads FAISS index and retrieves relevant chunks with PMID
│   ├── generator.py        # Generates answers using Ollama (LLaMA 3)
│   └── evaluate.py         # RAGAS evaluation pipeline with local Hugging Face models
│
└── data/
    ├── diabetes_abstracts.json     # Raw PubMed abstracts (generated)
    ├── chunks.json                 # Preprocessed text chunks with PMID metadata (generated)
    ├── chunks.pkl                  # Serialised chunks for retrieval (generated)
    ├── vector_index.faiss          # FAISS vector index (generated)
    └── evaluation_results.json     # RAGAS evaluation scores (generated)
```

---

## ⚙️ Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally
- LLaMA 3 pulled via Ollama:

```bash
ollama pull llama3
```

- BAAI/bge-large-en-v1.5 downloaded via Hugging Face CLI (used for evaluation):

```bash
pip install huggingface_hub
huggingface-cli download BAAI/bge-large-en-v1.5
```

> **Note:** `all-MiniLM-L6-v2` is used for embeddings via `sentence-transformers` (installed as a Python package — no separate download needed).

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/VishalSaravanan02/diabetes-rag-llm.git
cd diabetes-rag-llm
```

### 2. Create and activate a virtual environment (Python 3.11+)

```bash
python3.11 -m venv .venv
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

Cleans and splits abstracts into 500-character chunks with 100-character overlap. Each chunk is saved with its source PMID attached, so source attribution is preserved through the full pipeline.

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
- View the retrieved chunks that informed the answer, each showing:
  - The source PMID
  - The relevance distance score
  - A clickable **View on PubMed** link to the original paper

### Option B — CLI Interactive Mode

```bash
python main.py
```

Type your question at the prompt and press Enter. Type `exit` or `quit` to stop.

---

## 📊 Evaluating the Pipeline

The evaluation pipeline uses [RAGAS](https://docs.ragas.io/) with a local LLM (llama3 via Ollama) and a local Hugging Face embedding model to measure pipeline quality across 4 metrics.

### Run the evaluation

```bash
python -m src.evaluate
```

This runs 10 test questions through the full RAG pipeline and scores each one. Results are printed to the terminal and saved to `data/evaluation_results.json`.

### Metrics explained

| Metric | Description |
|---|---|
| **Faithfulness** | Are answers grounded in the retrieved context? Measures hallucination. |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Precision** | Are the retrieved chunks relevant to the question? |
| **Context Recall** | Do the retrieved chunks contain all the information needed to answer? |

### Baseline results (50 abstracts, llama3, top_k=5)

| Metric | Score |
|---|---|
| Faithfulness | 0.8333 |
| Answer Relevancy | 0.5377 |
| Context Precision | 0.0000 |
| Context Recall | 0.5000 |

**Interpretation:**
- **Faithfulness (0.83)** — when answers are generated, they are well grounded in the retrieved PubMed context
- **Answer Relevancy (0.54)** — moderate relevancy; some questions returned "I don't know" due to limited abstract coverage
- **Context Precision (0.00)** — scoring affected by llama3's inconsistent adherence to the evaluation prompt format
- **Context Recall (0.50)** — retrieval covers about half the necessary information; increasing the number of abstracts would directly improve this score

> **Note:** Scores are expected to improve significantly by fetching more abstracts (200+) and using full paper text instead of abstracts only.

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

Diabetes mellitus (DM) is a major contributor to disability and mortality, accounting for nearly 10% of all deaths in people aged 20 to 79 years.

**Retrieved Sources:** 5 chunks retrieved from the knowledge base, each displaying:
- Source PMID (e.g. `PMID: 41884250`)
- Relevance distance score
- Clickable link to the original PubMed paper

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
| Evaluation | RAGAS + `BAAI/bge-large-en-v1.5` (Hugging Face) |

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
- [x] Add PMID metadata through the full pipeline for proper source attribution
- [x] Add clickable PubMed links in the Streamlit UI
- [x] Add RAGAS evaluation pipeline with local Hugging Face models
- [ ] Increase abstract coverage (fetch 200+ abstracts for richer answers)
- [ ] Explore PubMed Central (PMC) full-text articles for deeper context
- [ ] Add streaming responses in the Streamlit UI
- [ ] Implement hybrid retrieval (BM25 + FAISS) for better search quality

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Vishal Saravanan**
[GitHub](https://github.com/VishalSaravanan02)