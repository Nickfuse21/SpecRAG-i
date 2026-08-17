"""
FastAPI service.

The one trap worth knowing about
--------------------------------
FastAPI runs on an ASGI event loop. A path function declared `async def`
runs ON that loop, and the loop is single-threaded. Our embedder, reranker
and NLI model are synchronous, CPU/GPU-bound calls that block for hundreds
of milliseconds to seconds. Put one inside an `async def` handler and it
blocks the entire event loop: not just that request, but every other
request, the health check, and the metrics endpoint. The server appears to
hang under trivial load, and nothing in the logs says why.

Two correct options. Either declare the handler `def` (not `async def`) and
let Starlette run it in its threadpool automatically, or keep it `async def`
and push the blocking work through `run_in_threadpool`. This file does the
second, explicitly, because the explicit version is the one you can point at
in an interview.

Models are loaded ONCE at startup, not per request. Loading bge-m3 takes
several seconds; doing that per request would dominate latency completely.
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import API_HOST, API_PORT, FINAL_TOP_K, REFUSAL_MESSAGE  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.generation.answer import Generator
    from src.retrieval.pipeline import Retriever
    from src.verification.groundedness import Verifier

    t0 = time.time()
    print("loading retriever (bge-m3 + reranker) ...")
    STATE["retriever"] = Retriever()
    print("loading generator ...")
    STATE["generator"] = Generator()
    print("loading verifier ...")
    STATE["verifier"] = Verifier(judge="nli")
    STATE["chunks"] = STATE["retriever"].store.count()
    print(f"ready in {time.time() - t0:.1f}s — {STATE['chunks']:,} chunks indexed")
    yield
    STATE.clear()


app = FastAPI(
    title="3GPP Spec Assistant",
    description="Grounded question answering over Rel-18 3GPP specifications.",
    version="1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(FINAL_TOP_K, ge=1, le=20)
    verify: bool = True


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    refusal_reason: str = ""
    stage: str                      # where it stopped: gate / generation / verification / answered
    groundedness: float | None = None
    verdict: str | None = None
    claims: list[dict] = []
    sources: list[dict] = []
    timings_ms: dict = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "chunks": STATE.get("chunks", 0)}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not STATE:
        raise HTTPException(503, "models still loading")

    timings: dict[str, float] = {}

    t = time.time()
    # top_k is passed through, not ignored: the request model advertised it
    # from the start, and an API that silently drops a parameter it documents
    # is worse than one that never offered it.
    res = await run_in_threadpool(
        STATE["retriever"].retrieve, req.question, False, req.top_k
    )
    timings["retrieve"] = round((time.time() - t) * 1000)

    sources = [
        {"n": i, "citation": h["citation"], "chunk_id": h["chunk_id"],
         "breadcrumb": h.get("breadcrumb", ""), "clause_title": h.get("clause_title", ""),
         "text": h["text"][:1500],
         "rerank_score": h.get("rerank_score")}
        for i, h in enumerate(res.hits, 1)
    ]

    # Stage 5 — the gate. No LLM call happens past this point if it fires.
    if res.refused:
        return AskResponse(
            question=req.question, answer=REFUSAL_MESSAGE, refused=True,
            refusal_reason=res.reason, stage="gate",
            sources=sources, timings_ms=timings,
        )

    t = time.time()
    ans = await run_in_threadpool(STATE["generator"].generate, req.question, res.hits)
    timings["generate"] = round((time.time() - t) * 1000)

    if ans.refused:
        return AskResponse(
            question=req.question, answer=ans.text, refused=True,
            refusal_reason=ans.refusal_reason, stage="generation",
            sources=sources, timings_ms=timings,
        )

    verdict = None
    if req.verify:
        from src.verification.groundedness import apply
        t = time.time()
        verdict = await run_in_threadpool(STATE["verifier"].verify, ans)
        ans = apply(ans, verdict)
        timings["verify"] = round((time.time() - t) * 1000)

    return AskResponse(
        question=req.question,
        answer=ans.text,
        refused=ans.refused,
        refusal_reason=ans.refusal_reason,
        stage="verification" if ans.refused else "answered",
        groundedness=round(verdict.score, 4) if verdict else None,
        verdict=verdict.label if verdict else None,
        claims=[{"text": c.text, "citations": c.citations, "normative": c.normative}
                for c in ans.claims],
        sources=sources,
        timings_ms=timings,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=False)
