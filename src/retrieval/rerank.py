"""
Stage two of retrieval: the cross-encoder reranker.

The difference from the embedder, in one line
---------------------------------------------
The embedder is a BI-encoder: it turns the query into a vector and each chunk
into a vector, separately, and compares the two numbers. It never sees them
together. That is what makes it fast enough to search 9,000 chunks — and it
is also its ceiling, because a chunk's single vector has to be a good summary
of it for every possible question, which is impossible.

A CROSS-encoder concatenates query and chunk and runs them through the
transformer jointly, so attention can connect "the UE" in the question to
"the UE shall" in the passage, and can notice that the passage is about
RRCResume when the question asked about RRCReconfiguration. It produces one
relevance score. It is far more accurate and roughly a thousand times more
expensive per pair — which is why it only ever sees the ~20 candidates the
first stage already narrowed down to.

Cheap and shallow first, expensive and sharp second. That funnel is the
single highest-leverage structural decision in the retrieval half of this
system.

The score is a sigmoid output in [0, 1] — check this, do not assume
-------------------------------------------------------------------
bge-reranker-v2-m3 is a single-logit regression head, and the natural
assumption (which this file made in its first draft) is that `predict()`
hands back that raw logit, unbounded and roughly in -10..+10.

It does not. sentence-transformers applies the activation named in the model
config, which for this model is Sigmoid, so every score is squashed into
[0, 1]. Verify it rather than trusting either claim:

    Reranker().model.activation_fn        # -> Sigmoid()

This is not a pedantic detail. RERANK_SCORE_THRESHOLD started life at 0.0,
which reads as a sensible midpoint for a signed logit and is in fact BELOW
THE ENTIRE RANGE of a sigmoid. The gate compares `best < threshold`, so it
could never fire: control #2 was switched off, and nothing in the system
would have said so — every query simply passed through to the LLM. The only
reason it surfaced is that eval/calibrate.py plots the score distribution
before fitting a cut, instead of fitting one and reporting a number.

The scores are still not calibrated ACROSS queries — a 0.6 for one question
does not mean what it means for another — which is why the threshold is
fitted empirically on the gold set rather than reasoned about.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import DEVICE, RERANK_BATCH, RERANK_FP16, RERANK_MODEL  # noqa: E402


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str = DEVICE,
                 fp16: bool = RERANK_FP16):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=1024)
        # See RERANK_FP16 in config. Note this changes the SCALE-SENSITIVE
        # number the relevance gate compares against, so the threshold must be
        # calibrated with the same precision the served path uses — which is
        # why eval/calibrate.py builds a normal Reranker() rather than its own.
        #
        # `.half()` on the CrossEncoder itself, NOT on `.model` inside it.
        # CrossEncoder subclasses nn.Sequential, so rebinding its `.model`
        # attribute replaces a registered submodule and silently corrupts the
        # forward path — it fails later and far away, in the tokeniser, with a
        # bare AttributeError. Calling .half() on the container recurses
        # properly and leaves the module graph alone.
        if fp16 and device == "cuda":
            self.model.half()

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
        return [float(s) for s in scores]

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        """
        Attach `rerank_score` to every hit and return the best `top_k`.

        We score the chunk body, not the contextual header + body. The header
        was built to help the FIRST stage find the chunk at all; feeding it
        here would let a chunk score well because its breadcrumb echoes the
        question's wording, which is not evidence that its content answers
        the question.
        """
        if not hits:
            return []
        scores = self.score(query, [h["text"] for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = s
        return sorted(hits, key=lambda h: -h["rerank_score"])[:top_k]
