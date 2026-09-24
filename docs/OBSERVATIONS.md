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
