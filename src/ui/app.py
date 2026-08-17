"""
Streamlit front end.

Streamlit's execution model, because it explains the caching
-----------------------------------------------------------
Streamlit re-runs this ENTIRE script from the top on every interaction —
every keystroke in a text box, every button click. Without caching, that
would reload bge-m3 and the reranker on every single event, which is
several seconds of GPU work per keypress.

`@st.cache_resource` is for objects that should exist once per process and
be shared across every run and every session: model handles, DB connections.
(`@st.cache_data` is the other one — for serialisable RETURN VALUES, which
it copies per caller. Using it for a model would try to pickle a GPU
handle.)

What this UI is actually for
----------------------------
Not decoration. The assessment is judged on trustworthiness, so the
interface has to make the system's reasoning inspectable: which passages
were retrieved, what the reranker scored them, where the answer stopped if
it stopped, and how each claim scored on groundedness. A refusal with the
closest passages shown is a MORE useful output than a confident guess, and
the UI should present it that way rather than as an error.

Usage
-----
    streamlit run src/ui/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import REFUSAL_MESSAGE, TARGET_RELEASE  # noqa: E402

st.set_page_config(page_title="3GPP Spec Assistant", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run takes ~30s)...")
def load_stack():
    from src.generation.answer import Generator
    from src.retrieval.pipeline import Retriever
    from src.verification.groundedness import Verifier
    r = Retriever()
    return r, Generator(), Verifier(judge="nli"), r.store.count()


retriever, generator, verifier, n_chunks = load_stack()

st.title("3GPP Spec Assistant")
st.caption(
    f"Grounded question answering over Release {TARGET_RELEASE} specifications · "
    f"{n_chunks:,} indexed passages · answers cite clause and version, or refuse"
)

with st.sidebar:
    st.subheader("Controls")
    do_verify = st.toggle("Groundedness verification", value=True,
                          help="Control #4 — checks each claim against its cited passage")
    show_debug = st.toggle("Show retrieval detail", value=False)
    st.divider()
    st.markdown(
        "**Try these**\n\n"
        "- What does the UE do when it receives an RRCReconfiguration?\n"
        "- When is timer T310 started and stopped?\n"
        "- What is 38.331 clause 5.3.5.3?\n"
        "- What is the capital of France?  ← should refuse"
    )

question = st.text_input("Question", placeholder="Ask about 5G NR / 5GC specifications...")

if question:
    timings = {}

    t = time.time()
    with st.spinner("Retrieving..."):
        res = retriever.retrieve(question)
    timings["retrieve"] = time.time() - t

    # ---- gate ------------------------------------------------------
    if res.refused:
        st.warning("**Refused before calling the model.**  \n" + REFUSAL_MESSAGE)
        st.caption(f"Reason: {res.reason}")
        st.info(
            "This is the relevance gate. Nothing was generated, so nothing could be "
            "invented — the question was rejected on retrieval evidence alone."
        )
        if res.hits:
            st.subheader("Closest passages found")
            for i, h in enumerate(res.hits[:5], 1):
                with st.expander(f"[{i}] {h['citation']} — {h.get('clause_title','')[:70]}"):
                    st.caption(h.get("breadcrumb", ""))
                    st.text(h["text"][:2000])
        st.stop()

    # ---- generate --------------------------------------------------
    t = time.time()
    with st.spinner("Reading the passages..."):
        ans = generator.generate(question, res.hits)
    timings["generate"] = time.time() - t

    verdict = None
    if do_verify and not ans.refused:
        from src.verification.groundedness import apply
        t = time.time()
        with st.spinner("Verifying each claim against its source..."):
            verdict = verifier.verify(ans)
            ans = apply(ans, verdict)
        timings["verify"] = time.time() - t

    # ---- present ---------------------------------------------------
    if ans.refused:
        st.warning("**Refused.**  \n" + ans.text)
        st.caption(f"Reason: {ans.refusal_reason}")
        if ans.missing:
            st.caption(f"Missing: {ans.missing}")
    else:
        if verdict:
            colour = {"grounded": "🟢", "partial": "🟡"}.get(verdict.label, "🔴")
            st.markdown(f"{colour} **{verdict.label.title()}** · groundedness {verdict.score:.2f}")
            if verdict.label == "partial":
                st.caption(f"{len(verdict.dropped)} claim(s) removed — could not be verified "
                           f"against the cited passage.")
        st.markdown(ans.text)

        if ans.claims:
            st.subheader("Claims and sources")
            for c in ans.claims:
                force = f"  `{c.normative}`" if c.normative != "none" else ""
                st.markdown(f"- {c.text}{force}")
                for cite in c.citations:
                    st.caption(f"    ↳ {cite}")

        if ans.invalid_citations:
            st.error(f"Model emitted out-of-range passage numbers {ans.invalid_citations} — "
                     f"dropped by citation validation.")

    # ---- evidence --------------------------------------------------
    st.subheader("Retrieved passages")
    for i, h in enumerate(res.hits, 1):
        score = h.get("rerank_score")
        label = f"[{i}] {h['citation']} — {h.get('clause_title','')[:64]}"
        if score is not None:
            label += f"   (rerank {score:+.2f})"
        with st.expander(label):
            st.caption(h.get("breadcrumb", ""))
            st.text(h["text"][:3000])

    if show_debug:
        st.subheader("Pipeline detail")
        st.json({
            "plan": {"specs": res.plan.specs, "clauses": res.plan.clauses} if res.plan else {},
            "top_rerank_score": res.top_score,
            "timings_s": {k: round(v, 2) for k, v in timings.items()},
            "per_claim": [
                {"text": cv.text[:90], "score": round(cv.score, 3),
                 "contradicted": cv.contradicted, "detail": cv.detail}
                for cv in (verdict.claims if verdict else [])
            ],
        })
