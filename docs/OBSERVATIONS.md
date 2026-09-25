# Observations

Findings from running the system, kept as evidence for later phases and as failure cases.

## 2026-09-25: first end-to-end runs (Phase 3)
Setup: all-MiniLM-L6-v2, cosine, MIN_SIMILARITY 0.47, llama3 (temperature 0), 5,080 abstracts.

1. **Weak retrieval for definitional questions.** "What is HbA1c?" ranked a paper on pregnancy
   outcomes in type 1 diabetes first; the most relevant paper ("Hemoglobin A1c: a reliable and
   accurate test...") came 3rd (top-5) or 5th (top-10). MiniLM matches the term, not the topic.
   -> Phase 6: stronger/biomedical embeddings, hybrid BM25, reranker.
2. **Inconsistent citation format.** llama3 sometimes writes `[PMID:24626616]` as instructed,
   sometimes `[2] PMID:28164640`, and sometimes copies whole excerpt headers.
   -> Phase 7: measure citation validity; tune prompt; compare llama3.1 (Decision D7).
3. **Possible unsupported statement.** The HbA1c answer (top-10) says it reflects average glucose
   over 2-3 months: medically correct, but possibly from the model's own knowledge rather than
   the excerpts. -> Phase 5: faithfulness metric.
4. **Duplicate papers in results.** Several chunks of one abstract can occupy multiple slots
   (SGLT2: PMID 35191660 at #1 and #5; HbA1c top-10: one meta-analysis at #4 and #9).
   -> Phase 6: consider collapsing results to one entry per paper.
5. **Refusals and errors behave correctly.** Off-topic questions are rejected before the LLM is
   called; Ollama being down shows a clear error; temperature 0 gives identical answers to
   identical questions.

## 2026-09-25: clean-clone test (Phase 4)
Fresh clone of `feat/chunking-v2`, new virtual environment, README followed step by step.

6. **Fully reproducible.** Same 5,089 abstracts, same 21,210 chunks, and a word-for-word identical
   answer and sources for "What is HbA1c?" as the original project.
7. **SSL certificate failure with python.org Python.** The venv created from the python.org
   installer failed to fetch (`CERTIFICATE_VERIFY_FAILED: self-signed certificate in certificate
   chain`); Homebrew's Python worked. Fixed: the fetcher now explains the cause and fixes instead of
   printing a traceback; README has a Troubleshooting section.
8. **Wrong Streamlit picked up with Anaconda active.** `streamlit run` ran Anaconda's copy (Python 3.13,
   no faiss) instead of the venv's. Fixed: README uses `python -m streamlit run app.py`.
9. **Silent incomplete corpus (found while fixing 7).** A batch that failed after retries was skipped
   without warning. Fixed: missing records are counted, saved in the meta file, and reported loudly.
