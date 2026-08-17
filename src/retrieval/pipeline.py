"""
The retrieval pipeline, end to end.

    question
       |
       v
  [1] query understanding   pull out spec numbers and clause ids
       |
       +--------------------+
       v                    v
  [2] dense (bge-m3)   [2] BM25            top-30 each, in parallel
       |                    |
       +---------+----------+
                 v
  [3] RRF fusion                            -> top-20
                 v
  [4] cross-encoder rerank                  -> top-6
                 v
  [5] RELEVANCE GATE                        -> refuse, or hand to the LLM

Stage 5 is the one that matters most for the brief. Everything before it
makes good answers more likely; the gate is the only stage that makes a
confident wrong answer impossible in a whole class of cases, because it
refuses BEFORE the model is ever called. A model that is never asked cannot
invent anything. Every other control in this system inspects an answer that
already exists and tries to catch a problem; this one prevents the answer.

Reciprocal Rank Fusion
----------------------
                          1
    RRF(chunk) = sum  ----------
                lists  k + rank

with k = 60 and rank starting at 1. Two properties earn its place:

  - It uses RANK, not score. Cosine similarity (0..1) and BM25 (unbounded,
    corpus-dependent) are not on the same scale and cannot be added. Any
    normalisation scheme you invent to fix that becomes a hyperparameter
    that drifts with the corpus. Ranks are already comparable.

  - k damps the head of each list. Without it, rank 1 would be worth 1.0 and
    rank 2 only 0.5 — one retriever's top hit would dominate. With k=60 the
    difference between rank 1 and rank 2 is small, so a chunk that BOTH
    retrievers rank moderately well beats one that a single retriever loves.
    Agreement between two systems with different biases is the signal.

Usage
-----
    python -m src.retrieval.pipeline --query "When does the UE trigger T310?"
    python -m src.retrieval.pipeline --query "..." --explain
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    BM25_TOP_K,
    DENSE_TOP_K,
    FINAL_TOP_K,
    FUSED_TOP_K,
    RERANK_SCORE_THRESHOLD,
    RRF_K,
)
from src.index.bm25_store import BM25Store  # noqa: E402
from src.index.vector_store import VectorStore  # noqa: E402

# "38.331", "TS 38.331", "TS38.331"
_SPEC_RE = re.compile(r"\b(?:TS\s*|TR\s*)?(\d{2}\.\d{3})\b", re.I)
# "clause 5.3.5.3", "section 6.2.2", "§5.3.5"
_CLAUSE_RE = re.compile(r"(?:clause|section|§)\s*([0-9]+(?:\.[0-9A-Za-z]+)*)", re.I)


@dataclass
class QueryPlan:
    text: str
    specs: list[str] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)

    @property
    def is_pinpoint(self) -> bool:
        """The user named an exact location, so we can look it up directly."""
        return bool(self.clauses)


def parse_query(q: str) -> QueryPlan:
    """
    Pull structured intent out of the question before searching.

    This exists because of a real failure found while testing BM25: for the
    query "38.331 clause 5.3.5.3", the top hits were all clause 5.7.4.3 —
    which cross-references 5.3.5.3 seven times and therefore genuinely wins
    on term frequency. BM25 was not broken; the query simply wasn't a
    keyword query. It was a lookup, and lookups deserve a filter, not a
    similarity score.
    """
    specs = sorted({m.group(1) for m in _SPEC_RE.finditer(q)})
    clauses = sorted({m.group(1) for m in _CLAUSE_RE.finditer(q)})
    return QueryPlan(text=q, specs=specs, clauses=clauses)


def rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Result:
    hits: list[dict]
    refused: bool
    reason: str = ""
    plan: QueryPlan | None = None
    top_score: float | None = None


class Retriever:
    """
    The flags exist for the ablation table, not for production tuning. Every
    one of them defaults to the full pipeline, so the served path is whatever
    a bare `Retriever()` does and an ablation has to ask to be degraded. The
    alternative — a second, simplified retriever written just for eval — would
    measure the ablation harness rather than the system, which is the standard
    way an ablation table ends up flattering.
    """

    def __init__(
        self,
        load_reranker: bool = True,
        use_bm25: bool = True,
        use_gate: bool = True,
        use_query_understanding: bool = True,
    ):
        from src.index.embedder import Embedder

        self.store = VectorStore()
        self.bm25 = BM25Store.load() if use_bm25 else None
        self.embedder = Embedder()
        self.use_bm25 = use_bm25
        self.use_gate = use_gate
        self.use_query_understanding = use_query_understanding
        self.reranker = None
        if load_reranker:
            from src.retrieval.rerank import Reranker
            self.reranker = Reranker()

    # ----------------------------------------------------------------
    def _where(self, plan: QueryPlan) -> dict | None:
        """Restrict the dense search when the user named specific specs."""
        if not plan.specs:
            return None
        if len(plan.specs) == 1:
            return {"spec": plan.specs[0]}
        return {"$or": [{"spec": s} for s in plan.specs]}

    def _pinpoint(self, plan: QueryPlan) -> list[dict]:
        """
        Exact-location lookup. If the user asked for clause 5.3.5.3, the
        chunks whose clause_id IS 5.3.5.3 are not candidates to be ranked —
        they are the answer, and they go in front.
        """
        out: list[dict] = []
        for clause in plan.clauses:
            where: dict = {"clause_id": clause}
            if plan.specs:
                where = {"$and": [where, self._where(plan)]}
            try:
                res = self.store.col.get(
                    where=where, include=["documents", "metadatas"], limit=20
                )
            except Exception:
                continue
            from src.index.vector_store import from_metadata
            for cid, doc, md in zip(res["ids"], res["documents"], res["metadatas"]):
                h = from_metadata(md)
                h.update({"chunk_id": cid, "text": doc, "source": "pinpoint"})
                out.append(h)
        out.sort(key=lambda h: h.get("part", 1))
        return out

    # ----------------------------------------------------------------
    def retrieve(self, question: str, explain: bool = False,
                 top_k: int | None = None) -> Result:
        plan = parse_query(question) if self.use_query_understanding else QueryPlan(text=question)

        dense_hits = self.store.search(
            self.embedder.embed_queries([question])[0],
            k=DENSE_TOP_K,
            where=self._where(plan),
        )
        bm25_hits = self.bm25.search(question, k=BM25_TOP_K) if self.bm25 else []

        ranked = [[h["chunk_id"] for h in dense_hits]]
        if bm25_hits:
            ranked.append([cid for cid, _ in bm25_hits])
        fused = rrf(ranked)

        by_id = {h["chunk_id"]: h for h in dense_hits}
        missing = [cid for cid in fused if cid not in by_id]
        for h in self.store.get(missing):
            by_id[h["chunk_id"]] = h

        candidates = []
        for cid, s in sorted(fused.items(), key=lambda kv: -kv[1])[:FUSED_TOP_K]:
            h = by_id.get(cid)
            if h is None:
                continue
            h["rrf_score"] = s
            candidates.append(h)

        pinned = self._pinpoint(plan)
        if pinned:
            seen = {h["chunk_id"] for h in pinned}
            candidates = pinned + [c for c in candidates if c["chunk_id"] not in seen]

        if explain:
            print(f"\nplan: specs={plan.specs or '-'} clauses={plan.clauses or '-'}")
            print(f"dense {len(dense_hits)} | bm25 {len(bm25_hits)} | "
                  f"fused {len(fused)} | pinpoint {len(pinned)} | to rerank {len(candidates)}")

        if not candidates:
            return Result([], refused=True, reason="nothing retrieved", plan=plan)

        # `top_k` exists so the eval harness can measure recall at the depth of
        # the candidate pool as well as at the depth that actually reaches the
        # LLM. Recall@FINAL_TOP_K is what the answer sees; recall deeper in the
        # pool is the ceiling reranking has to work with, and the gap between
        # the two is exactly what the reranker is being paid to close.
        k = top_k or FINAL_TOP_K

        if self.reranker is None:
            return Result(candidates[:k], refused=False, plan=plan)

        # Score every candidate, then decide the ORDER separately.
        #
        # `rerank(..., k)` sorts by score and truncates, which silently undid
        # the pinpoint step above: the exact clause the user named was pushed to
        # the front of `candidates`, and then re-sorted straight back down. A
        # clause fetched by id is often a poor semantic match for the question
        # that asked for it — "What does clause 5.3.3.7 say?" shares almost no
        # wording with the clause's own text — so it scored ~0.4 against
        # passages that merely CITE 5.3.3.7 and lost. The observable symptom was
        # the model correctly reporting that the passages did not contain the
        # requested clause, which reads like a retrieval miss and is actually a
        # ranking bug one line downstream.
        scored = self.reranker.rerank(question, candidates, len(candidates))

        if pinned:
            pin_ids = [h["chunk_id"] for h in pinned]   # already in part order
            pin_set = set(pin_ids)
            by_id = {h["chunk_id"]: h for h in scored}
            head = [by_id[c] for c in pin_ids if c in by_id]
            tail = [h for h in scored if h["chunk_id"] not in pin_set]
            top = (head + tail)[:k]
        else:
            top = scored[:k]

        # The gate compares against the strongest evidence we found, not against
        # whatever happens to be first — for a pinpoint query that is the named
        # clause, whose score is low by nature.
        best = max(h["rerank_score"] for h in top)

        # ---- CONTROL #2: the relevance gate --------------------------
        # A pinpoint lookup is exempt: the user named the clause, so
        # "here is that clause" is the correct response even if the
        # reranker finds it a poor semantic match for their phrasing.
        if self.use_gate and best < RERANK_SCORE_THRESHOLD and not plan.is_pinpoint:
            return Result(
                top, refused=True,
                reason=f"best rerank score {best:.2f} < threshold {RERANK_SCORE_THRESHOLD}",
                plan=plan, top_score=best,
            )

        return Result(top, refused=False, plan=plan, top_score=best)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the retrieval pipeline")
    ap.add_argument("--query", required=True)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--no-rerank", action="store_true", help="ablation: skip stage 4 and 5")
    args = ap.parse_args()

    r = Retriever(load_reranker=not args.no_rerank)
    res = r.retrieve(args.query, explain=args.explain)

    print(f"\nquery: {args.query}")
    if res.refused:
        print(f"\nREFUSED — {res.reason}")
        print("closest passages, for the user to judge:")

    for i, h in enumerate(res.hits, 1):
        rs = h.get("rerank_score")
        tag = f"rerank {rs:+.2f}" if rs is not None else f"rrf {h.get('rrf_score', 0):.4f}"
        src = " [pinpoint]" if h.get("source") == "pinpoint" else ""
        print(f"\n{i}. {tag}{src}  {h['citation']}")
        print(f"   {h['breadcrumb'][:96]}")
        print(f"   {h['text'][:200].replace(chr(10), ' ')}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
