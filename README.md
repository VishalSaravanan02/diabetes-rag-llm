# 🧬 Diabetes RAG LLM

A Retrieval-Augmented Generation (RAG) system that answers biomedical questions about diabetes using research abstracts fetched from PubMed. It combines semantic search via FAISS with a locally-running LLM through Ollama, and exposes the system through an interactive Streamlit web interface.

---

## 📌 Overview

This project implements a full RAG pipeline:

1. **Fetch** — PubMed abstracts on diabetes are retrieved using the NCBI Entrez API
2. **Preprocess** — Abstracts are chunked into overlapping segments for better retrieval
3. **Embed** — Chunks are encoded using `all-MiniLM-L6-v2` and stored in a FAISS vector index
4. **Retrieve** — At query time, the most semantically relevant chunks are retrieved
5. **Generate** — A local LLM (LLaMA via Ollama) generates a grounded answer using the retrieved context
6. **Interface** — A Streamlit web app provides a clean UI for querying the system

---

## 🗂️ Project Structure

```
diabetes-rag-llm/
│
├── app.py                  # Streamlit web interface
├── main.py                 # CLI entry point for the RAG QA system
├── requirements.txt        # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── fetch_data.py       # Fetches abstracts from PubMed via Entrez
│   ├── preprocess.py       # Chunks abstracts using LangChain text splitter
│   ├── embeddings.py       # Generates embeddings and builds FAISS index
│   ├── retriever.py        # Loads FAISS index and retrieves relevant chunks
│   └── generator.py        # Generates answers using Ollama (LLaMA 3.1)
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
- The following Ollama models pulled:

```bash
ollama pull llama3.1       # For answer generation (used in generator.py / app.py)
ollama pull llama2         # For answer generation (used in main.py)
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

---

## 🔧 Building the Knowledge Base

Run these steps once to fetch data and build the vector index.

### Step 1 — Fetch PubMed abstracts

```bash
python src/fetch_data.py
```

This fetches 50 PubMed abstracts about diabetes and saves them to `data/diabetes_abstracts.json`.

### Step 2 — Preprocess and chunk the abstracts

```bash
python src/preprocess.py
```

Splits abstracts into 500-token chunks with 100-token overlap and saves them to `data/chunks.json`.

### Step 3 — Generate embeddings and build FAISS index

```bash
python src/embeddings.py
```

Encodes all chunks using `all-MiniLM-L6-v2` and saves the FAISS index to `data/vector_index.faiss`.

---

## 💬 Running the Application

### Option A — Streamlit Web UI (recommended)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B — CLI Interactive Mode

```bash
python main.py
```

Type your question at the prompt and press Enter. Type `exit` or `quit` to stop.

---

## 🧪 Example Query

**Question:** What is diabetes?

**Answer:** *(Generated from retrieved PubMed context using LLaMA)*

Diabetes is a group of metabolic disorders characterized by high blood sugar levels. It occurs when the body either does not produce enough insulin (a hormone that regulates blood sugar levels) or cannot effectively use the insulin it produces.

There are several types of diabetes, including:

1. **Type 1 Diabetes:** An autoimmune disease in which the pancreas is unable to produce insulin, resulting in high blood sugar levels.
2. **Type 2 Diabetes:** The most common form of diabetes, characterized by insulin resistance and impaired insulin secretion.
3. **Gestational Diabetes:** A type of diabetes that develops during pregnancy, usually in the second or third trimester.
4. **LADA (Latent Autoimmune Diabetes in Adults):** A form of type 1 diabetes that resembles type 2 diabetes in terms of symptoms and progression.

Common risk factors include family history, obesity, physical inactivity, high blood pressure, high cholesterol, age, and ethnicity. Symptoms can include increased thirst and urination, fatigue, blurred vision, slow wound healing, and frequent infections.

If left untreated, diabetes can lead to serious complications such as kidney damage (nephropathy), nerve damage (neuropathy), eye damage (retinopathy), foot ulcers, and increased risk of heart disease and stroke. With proper treatment — including medication, lifestyle changes, and insulin therapy where necessary — many people with diabetes are able to manage their condition effectively.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Data source | PubMed via NCBI Entrez (`Biopython`) |
| Text chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`faiss-cpu`) |
| LLM | LLaMA 2 / LLaMA 3.1 via Ollama |
| Web interface | Streamlit |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 Author

**Vishal Saravanan**  
[GitHub](https://github.com/VishalSaravanan02)
