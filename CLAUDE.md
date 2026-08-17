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
- Python venv at `venv\` — **activate before anything**:
  `.\venv\Scripts\Activate.ps1`
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
   is below `RERANK_SCORE_THRESHOLD`, refuse **before calling the LLM**. This
   is the only control that makes a wrong answer impossible rather than
   detectable — a model that is never asked cannot invent anything.
   `RERANK_SCORE_THRESHOLD = 0.0` is a **PLACEHOLDER and must be calibrated**
   on the eval set. See "What's left".

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

Corpus: 15 specs, 6,367 clauses, ~4.7M tokens → 9,476 chunks.

## What's left

1. **Unblock the GPU** (above), then run the index build.
2. `python -m src.ingest.chunk` currently reports **13 orphaned nested
   lines** across the full 15-spec corpus. The updated `chunk.py` prints the
   offending chunk ids and lines. Diagnose and fix — this is invariant #2.
3. **Calibrate `RERANK_SCORE_THRESHOLD`.** Build a gold set including
   deliberately out-of-scope questions, plot max-rerank-score distributions
   for answerable vs unanswerable, and cut where they separate. Until this
   is done the gate works but its number cannot be defended.
4. **Eval harness + ablation table**: Recall@k, MRR, nDCG, faithfulness,
   hallucination rate, correct-refusal rate. Ablations:
   naive → +hybrid → +rerank → +gate → +verification. This is the main
   evidence for the "quality and effectiveness" criterion.
5. **README** with architecture, design rationale, results table, and setup.
6. Demo run-through.

## Run order

```powershell
.\venv\Scripts\Activate.ps1

python -m src.ingest.download          # done
python -m src.ingest.convert           # done
python -m src.ingest.parse_docx        # done
python -m src.ingest.chunk             # re-run, must show 0 orphans

python -m src.index.embedder --self-test
python -m src.index.build --limit 200 --reset   # smoke test
python -m src.index.build --reset               # full, 10-30 min on GPU
python -m src.index.bm25_store --build

python -m src.retrieval.pipeline --query "When does the UE trigger T310?" --explain
python -m src.generation.answer --query "..."
python -m src.verification.groundedness --query "..."

uvicorn src.api.main:app --port 8000
streamlit run src/ui/app.py
```

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
