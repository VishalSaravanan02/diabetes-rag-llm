"""
The one entry point for answering a question: retrieve, then generate.

The CLI (main.py), the Streamlit app (app.py) and the evaluation all call
answer_question(), so they behave identically and there is one place to change.

answer_question() always returns a RAGResult, never raises for normal failures:
    status "ok"          -> answer is set, sources are the excerpts used
    status "no_results"  -> nothing passed the relevance threshold; the LLM was not called
    status "error"       -> retrieval worked but generation failed; see `error`
Setup problems (a missing or out-of-date index) still raise IndexNotReadyError,
because they need fixing before anything can work.
"""

from dataclasses import asdict, dataclass, field

from config import TOP_K
from src.generator import NO_ANSWER, GenerationError, generate_answer
from src.retriever import retrieve

STATUS_OK = "ok"
STATUS_NO_RESULTS = "no_results"
STATUS_ERROR = "error"


@dataclass
class RAGResult:
    question: str
    answer: str | None = None
    sources: list[dict] = field(default_factory=list)
    status: str = STATUS_OK
    error: str | None = None

    @property
    def refused(self):
        """True if the model said the excerpts don't answer the question."""
        return self.answer is not None and NO_ANSWER.lower() in self.answer.lower()

    def to_dict(self):
        return {**asdict(self), "refused": self.refused}


def answer_question(question: str, top_k: int = TOP_K) -> RAGResult:
    question = question.strip()
    sources = retrieve(question, top_k=top_k)

    if not sources:                       # nothing relevant: don't call the LLM
        return RAGResult(question, status=STATUS_NO_RESULTS)

    try:
        answer = generate_answer(question, sources)
    except GenerationError as e:
        return RAGResult(question, sources=sources, status=STATUS_ERROR, error=str(e))

    return RAGResult(question, answer=answer, sources=sources)
