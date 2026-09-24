"""
Turn retrieved PubMed excerpts into an answer with the local LLM (via Ollama).

- The prompt lists each excerpt with its PMID, title, journal and year, and asks
  the model to cite sources as [PMID:12345678]. (Phase 7 verifies citations.)
- If the excerpts don't answer the question, the model is told to reply with
  NO_ANSWER exactly, so refusals can be detected and measured later.
- Failures raise GenerationError. An error is never returned as if it were an
  answer, so it can't be displayed or evaluated as one.
"""

import ollama

from config import LLM_MODEL, LLM_TEMPERATURE

NO_ANSWER = "The retrieved abstracts don't answer this question."

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the question using ONLY the "
    "numbered PubMed excerpts provided.\n"
    "- Cite the source of every factual sentence as [PMID:12345678], using the PMIDs "
    "shown in the excerpts.\n"
    "- If the excerpts do not contain the answer, reply with exactly this sentence "
    f"and nothing else: {NO_ANSWER}\n"
    "- Be concise and factual. Mention disagreement between sources if there is any.\n"
    "- Do not give personal medical advice."
)


class GenerationError(RuntimeError):
    """The LLM could not produce an answer. Never returned as answer text."""


def format_context(results):
    """Number each excerpt and label it with its paper details."""
    blocks = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] PMID:{r['pmid']} | {r.get('title') or 'Untitled'}"
        details = ", ".join(str(x) for x in (r.get("journal"), r.get("year")) if x)
        if details:
            header += f" ({details})"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n".join(blocks)


def build_messages(question, results):
    user = f"Excerpts:\n\n{format_context(results)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_answer(question, results):
    """Answer `question` from the retrieved `results` (list of dicts from the retriever)."""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=build_messages(question, results),
            options={"temperature": LLM_TEMPERATURE},
        )
    except ConnectionError as e:
        raise GenerationError(
            "Can't reach Ollama. Start it with `ollama serve` (or `brew services start ollama`)."
        ) from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise GenerationError(
                f"Ollama doesn't have the model '{LLM_MODEL}'. Download it with `ollama pull {LLM_MODEL}`."
            ) from e
        raise GenerationError(f"Ollama returned an error: {e.error}") from e
    except Exception as e:  # noqa: BLE001 - anything else is still a generation failure
        raise GenerationError(f"Unexpected error while generating an answer: {e}") from e

    answer = (response["message"]["content"] or "").strip()
    if not answer:
        raise GenerationError("The model returned an empty answer.")
    return answer
