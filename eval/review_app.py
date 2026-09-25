"""
Review drafted test questions: keep (optionally edited) or drop each one.

Run from the project root:
    python -m streamlit run eval/review_app.py

Reads  eval/candidates.jsonl   (from eval/draft_questions.py)
Writes eval/reviewed.jsonl     (one line per decision; the last decision per question wins)

Stop any time: decisions are saved immediately, and the app resumes at the
first question you haven't decided on yet.
"""

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from config import ABSTRACTS_FILE  # noqa: E402

CANDIDATES = ROOT / "eval" / "candidates.jsonl"
REVIEWED = ROOT / "eval" / "reviewed.jsonl"


def read_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_papers():
    with open(ABSTRACTS_FILE, encoding="utf-8") as f:
        return {p["pmid"]: p for p in json.load(f)}


def decisions():
    latest = {}
    for rec in read_jsonl(REVIEWED):
        latest[rec["candidate_id"]] = rec
    return latest


def record_decision(candidate, decision, question, reference):
    rec = {
        "candidate_id": candidate["candidate_id"],
        "pmid": candidate["pmid"],
        "type": candidate["type"],
        "decision": decision,
        "question": question.strip(),
        "reference": reference.strip(),
        "gold_pmids": candidate["gold_pmids"],
        "edited": question.strip() != candidate["question"] or reference.strip() != candidate["reference"],
        "drafted_by": candidate.get("drafted_by"),
        "reviewed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(REVIEWED, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def undo_last():
    lines = REVIEWED.read_text(encoding="utf-8").splitlines(keepends=True) if REVIEWED.exists() else []
    if lines:
        REVIEWED.write_text("".join(lines[:-1]), encoding="utf-8")


# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Question review", page_icon="📝", layout="wide")

candidates = read_jsonl(CANDIDATES)
if not candidates:
    st.error(f"No drafts found at {CANDIDATES.relative_to(ROOT)}. Run `python -m eval.draft_questions` first.")
    st.stop()

done = decisions()
kept = sum(1 for d in done.values() if d["decision"] == "keep")
dropped = sum(1 for d in done.values() if d["decision"] == "drop")
pending = [c for c in candidates if c["candidate_id"] not in done]

with st.sidebar:
    st.header("Progress")
    target = st.number_input("Target: questions to keep", min_value=1, value=40)
    st.progress(min(kept / target, 1.0), text=f"Kept {kept} of {target}")
    st.caption(f"Dropped {dropped} · Remaining drafts {len(pending)}")
    if st.button("↩️ Undo last decision", disabled=not done):
        undo_last()
        st.rerun()

    st.header("Keep a question if")
    st.markdown(
        "- a clinician or researcher could **realistically ask** it\n"
        "- **this abstract answers it** with a specific finding\n"
        "- it **makes sense on its own** (no \"this study\")\n"
        "- it **doesn't copy the title**; reword it if it does\n"
        "- the **reference answer is correct** according to the abstract\n\n"
        "**Edit freely**: fixing a good-but-clumsy draft is better than dropping it.\n\n"
        "**Drop** if it's vague, trivial, answerable from general knowledge alone, "
        "or the abstract doesn't clearly support the answer."
    )

if kept >= target:
    st.success(f"🎉 Target reached: {kept} questions kept. You can stop here, or keep going for spares.")
if not pending:
    st.info("All drafts reviewed. Draft more with `python -m eval.draft_questions --n 160` if you need them.")
    st.stop()

candidate = pending[0]
paper = load_papers().get(candidate["pmid"], {})

st.title("📝 Review test questions")
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader(paper.get("title", "(paper not found in corpus)"))
    details = " · ".join(str(x) for x in (paper.get("journal"), paper.get("year")) if x)
    st.caption(f"PMID {candidate['pmid']} · {details}")
    types = [t for t in paper.get("publication_types", []) if t != "Journal Article"]
    if types:
        st.caption("Study type: " + ", ".join(types))
    st.markdown((paper.get("abstract") or "").replace("$", "\\$"))

with right:
    if candidate.get("flags"):
        st.warning("Check: " + "; ".join(candidate["flags"]))
    with st.form(key=f"review_{candidate['candidate_id']}"):
        question = st.text_area("Question", value=candidate["question"], height=110)
        reference = st.text_area("Reference answer (from the abstract)", value=candidate["reference"], height=150)
        c1, c2 = st.columns(2)
        keep = c1.form_submit_button("✅ Keep", type="primary", width="stretch")
        drop = c2.form_submit_button("🗑️ Drop", width="stretch")
    st.caption(f"Drafted by {candidate.get('drafted_by', '?')}. Your edits are saved when you click Keep.")

if keep:
    if not question.strip() or not reference.strip():
        st.error("Question and reference answer can't be empty.")
    else:
        record_decision(candidate, "keep", question, reference)
        st.rerun()
if drop:
    record_decision(candidate, "drop", candidate["question"], candidate["reference"])
    st.rerun()
