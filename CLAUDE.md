# 3GPP RAG Assistant — project context

Read this first. It is the handoff for anyone (human or agent) picking this up.

## What this is

A retrieval-augmented question answering system over 3GPP Release-18
telecom specifications, built for a Graduate Engineer Trainee assessment.
The grading criteria are, in order: **minimal to near-zero hallucinations**,
quality/effectiveness of the solution, and the author's ability to explain
every design and code decision in an interview.

That last criterion is why this codebase is heavily commented with *why*,
not *what*. Do not strip those comments. They are a deliverable.

**Deadline: 17 August 2026.**

## Environment

- Windows, PowerShell, project at `...\Desktop\RAG project\rag3gpp`
- Python venv is at `RAG project\venv` — one level ABOVE the package, not
  inside it. Activate it, then `cd rag3gpp` before running anything; from the
  parent folder you get `No module named 'src'`. See SETUP.md.
- NVIDIA GPU present. The whole pipeline assumes CUDA.
- Gemini API key in `.env` (`GEMINI_API_KEY=`). Never commit `.env`.
- LibreOffice for `.doc` → `.docx` (already done; all 15 specs are `.docx`)

## GPU — resolved, but read the fp16 note

torch is now `2.13.0+cu126`, `cuda available: True`, on an RTX 3050 Laptop
(4 GB). The earlier CPU-only-build blocker is gone.

The 4 GB is the constraint that shapes everything. bge-m3 is ~2.27 GB in fp32
and a Windows desktop session already holds ~2 GB of the card, so the weights
do not fit. WDDM does not raise an error for that — it spills them to system
RAM and streams them back over PCIe on every forward pass. The symptom is a
run that completes correctly at **1 chunk/s** (~2.7 h for this corpus) with no
warning anywhere.

`EMBED_FP16` / `RERANK_FP16` / `NLI_FP16` in config exist for that reason, not
for throughput vanity: in fp16 the same build runs at **10.5 chunks/s** and
finishes in 15 min. Keep all three on unless `DEVICE = "cpu"`, where fp16 is
emulated and slower. If you load the embedder, reranker and NLI judge together
in fp32 they reserve ~4.35 GB and the spilling starts again silently.

If no GPU is available, the fallback is to edit `src/config.py`: set
`DEVICE = "cpu"`, `EMBED_MODEL = "BAAI/bge-small-en-v1.5"`, `EMBED_DIM = 384`,
`RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`, and the three `*_FP16`
flags to `False`. No other code changes — that is the payoff for keeping every
tunable in one file. Quality drops; the architecture and all four
hallucination controls are unaffected, and the ablation table is still honest.

## Pipeline

Offline:
```
download.py -> convert.py -> parse_docx.py -> chunk.py -> build.py (dense)
                                                       -> bm25_store.py (sparse)
```
Online:
```
question -> query understanding -> dense + BM25 -> RRF -> cross-encoder rerank
         -> RELEVANCE GATE -> Gemini (forced citations) -> groundedness verify
         -> answer | refusal
```

## The four hallucination controls — do not weaken these

1. **Version pinning.** One release only (Rel-18). Mixing releases lets
   retrieval return contradictory text that the model blends into a
   confident wrong answer. `TARGET_RELEASE` in config; manifest records what
   was actually pinned.

2. **Relevance gate** (`src/retrieval/pipeline.py`). If the best rerank score
   is below `RERANK_SCORE_THRESHOLD`, refuse **before calling the LLM**. It is
   the only control that prevents a wrong answer rather than detecting one — a
   model that is never asked cannot invent anything — and the only one that
   does not depend on the model cooperating.
   **Now calibrated to 0.90** (`eval/calibrate.py`, see Findings). Note the
   honest caveat from the ablation: on this gold set the gate did not reduce
   the hallucination rate, because control #3's explicit `answerable: false`
   field already refused every out-of-scope question. The gate's measured value
   here is that it does so without an LLM call and without trusting the model.

3. **Citation validity** (`src/generation/answer.py`). Passages are numbered
   `[1]..[n]` and the model may only emit integers. Citation strings are
   rendered afterwards from our own metadata. The model is structurally
   incapable of fabricating a citation, not merely discouraged from it.
   Out-of-range integers are dropped and reported.

4. **Groundedness verification** (`src/verification/groundedness.py`).
   Per-claim entailment against the claim's own cited passages, scored by
   two judges, taking the minimum. Thresholds 0.90 / 0.60 →
   grounded / partial / refuse. One contradicted claim refuses outright
   regardless of the mean.

## Invariants that must not be broken

These were expensive to get right. Changing them without understanding why
will silently reintroduce hallucinations.

- **A citation is never more specific than the truth.** Tiny clauses merge
  UP into an ancestor (whose id remains a true citation), never sideways
  into a sibling or down into a child.
- **A `shall` never loses its condition.** 3GPP procedural text nests with
  `1>` / `2>` / `3>` depth markers. `ancestor_map()` in `chunk.py` computes
  every line's enclosing condition chain, and any chunk containing a line
  also contains its chain verbatim. `find_orphans()` enforces this and
  **must report zero**. If it doesn't, do not build the index.
- **No synthesised text ever enters a chunk.** Anything we write would be
  indistinguishable from spec text to every downstream stage.
- **Normative force is preserved.** `shall` / `should` / `may` have distinct
  legal meanings (TR 21.801). Turning a `may` into a `must` invents a
  requirement even if every word came from the source.
- **Never split a table's header from its rows**, or an ASN.1 definition
  mid-body. Oversized blocks split at row / definition boundaries only.

## Status

Done and validated:

| Component | File |
|---|---|
| Config (every tunable) | `src/config.py` |
| Spec downloader, version-pinned | `src/ingest/download.py` |
| `.doc` → `.docx` | `src/ingest/convert.py` |
| Clause-tree parser | `src/ingest/parse_docx.py` |
| Structure-aware chunker | `src/ingest/chunk.py` |
| bge-m3 embedder | `src/index/embedder.py` |
| Chroma vector store | `src/index/vector_store.py` |
| Index builder | `src/index/build.py` |
| BM25 (hand-implemented) | `src/index/bm25_store.py` |
| Cross-encoder reranker | `src/retrieval/rerank.py` |
| Hybrid + RRF + gate | `src/retrieval/pipeline.py` |
| Gemini generation | `src/generation/answer.py` |
| Groundedness verification | `src/verification/groundedness.py` |
| FastAPI | `src/api/main.py` |
| Streamlit UI | `src/ui/app.py` |
| Retry/backoff for Gemini | `src/llm_retry.py` |
| Gold set (55 answerable + 22 not) | `eval/gold.jsonl` |
| Gate calibration | `eval/calibrate.py` |
| Ablation harness | `eval/run_eval.py` |
| Demo run-through | `src/demo.py` |

Corpus: 15 specs, 6,367 clauses, 4.94M tokens → **9,577 chunks**, 0 orphaned
conditions. Indexed: 9,577 dense vectors + BM25 over 53,134 terms.

## Findings from the calibration and eval pass

Three things that were wrong and are worth being able to explain:

1. **The gate had never fired.** `RERANK_SCORE_THRESHOLD = 0.0` assumed the
   reranker returns signed logits. It returns SIGMOID outputs in [0, 1], so
   `best < 0.0` was never true. Control #2 — the one the whole design leans on
   — was inert, and nothing surfaced it because the system still answered.
   Fitted to **0.90**: refuses 77% of unanswerable questions, 4% of answerable.
2. **fp16 is load-bearing on this GPU**, not a nicety. See the GPU section.
3. **The cross-encoder barely improves ranking** (+1.9pts MRR, +0.1pt nDCG) at
   ~40x the latency. Do not defend it on retrieval metrics; defend it on the
   fact that it produces the score the gate needs. There is no gate without it.

Also: `gemini-2.5-flash` now 404s for keys that had not already used it, and
`gemini-3.5-flash` served only 2/4 requests under free-tier load (503 + 429).
`GEN_MODEL` is `gemini-3.5-flash-lite` (4/4 at 1.3s) and `UTIL_MODEL` is a
DIFFERENT model on purpose — a judge should not be grading its own generator.

## What's left

1. **Answer ablation** (`python -m eval.run_eval --answers`) — retrieval arms
   are done and in the README; the gate/verification arms need a clean run.
2. **README results section** — fill in the answer table once (1) lands.
3. **Demo** — `python -m src.demo` walks four questions, one per stopping point.
4. Optional: the 2 answerable questions the gate now refuses are the price of
   the 10% budget. If that matters, re-run `eval/calibrate.py
   --max-false-refusal 0.02` and take the weaker gate knowingly.

## Run order

See **[SETUP.md](SETUP.md)** — it is the single source of truth for every
command. It used to be duplicated here and in README.md, and the three copies
drifted: this file still described a torch+cpu blocker after it was fixed, and
README.md still called the whole pipeline "scaffolded" after it was built.


## Conventions

- All tunables live in `src/config.py`. Nowhere else. This exists so Day-3
  ablations are reproducible instead of guesswork.
- Every module is runnable standalone with `python -m` and has a self-check
  or `--sample` / `--explain` mode. Verify each stage before building on it.
- Comments explain reasoning and failure modes, not syntax.
- `data/` is gitignored except `manifest.tsv`.
- Explain things to the user in Hinglish (Hindi-English, Roman script) —
  that is how they work, and they want to understand every piece, not just
  receive working code.
