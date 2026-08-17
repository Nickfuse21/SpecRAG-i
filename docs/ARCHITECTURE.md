# 3GPP RAG Chatbot — Solution Design

**Goal:** A Retrieval-Augmented Generation chatbot over 3GPP telecom standards with **minimal-to-near-zero hallucination**.

**Core design thesis:** Hallucination is not fixed by a better prompt. It is fixed by *removing the model's opportunity to guess*. Every layer below either (a) increases the chance the right text is in front of the model, or (b) blocks the answer from leaving the system when it isn't grounded.

---

## 0. Stack decisions (locked)

| Concern | Choice | Why |
|---|---|---|
| Generation LLM | Gemini 2.5 Flash (API) | Free tier, strong instruction-following, 1M context, reliable JSON mode |
| Utility LLM | Gemini 2.5 Flash-Lite | Query rewriting + verification judge; cheap, fast |
| Embeddings | `BAAI/bge-m3` (local, GPU) | Domain terms (IE names, acronyms) survive better than with generic API embeddings; free, offline, reproducible |
| Reranker | `BAAI/bge-reranker-v2-m3` (local, GPU) | The single biggest precision jump in the pipeline |
| Lexical index | BM25 (`rank_bm25`) | Non-negotiable for telecom: exact tokens like `RRCReconfiguration`, `gNB-DU`, `PDU Session ID` |
| Vector store | Qdrant (local Docker) — fallback ChromaDB | Metadata filtering + persistence + a real DB story for the interview |
| Backend | FastAPI | Streaming `/chat`, plus a `/search` endpoint that exposes raw retrieval |
| Frontend | Streamlit | Chat + a **transparency panel** showing retrieved chunks, scores, and groundedness |

---

## 1. Corpus — scoped and version-pinned

**Domain: 5G (NR + 5GC), pinned to a single Release.**

Version pinning is itself a hallucination control. Mixing Rel-15 and Rel-18 text produces contradictory retrieved passages, and the LLM will silently blend them. We pin one release, record the exact version in every chunk's metadata, and display it in every citation.

Proposed set (~14 specs, downloaded from the open 3GPP FTP archive — no login required):

**RAN / air interface**
- TS 38.300 — NR and NG-RAN Overall Description
- TS 38.331 — RRC Protocol Specification
- TS 38.321 — MAC
- TS 38.322 — RLC
- TS 38.323 — PDCP
- TS 38.401 — NG-RAN Architecture Description
- TS 38.211 / 38.212 / 38.213 / 38.214 — Physical layer (channels, coding, control, data)

**Core network / system**
- TS 23.501 — 5G System Architecture
- TS 23.502 — Procedures for the 5G System
- TS 24.501 — NAS protocol for 5GS
- TS 33.501 — Security architecture and procedures for 5GS

**Vocabulary**
- TR 21.905 — Vocabulary for 3GPP Specifications → used to auto-build the acronym glossary

Source pattern: `https://www.3gpp.org/ftp/Specs/archive/<NN>_series/<spec>/<specnum>-<ver>.zip`
Each zip contains a `.doc`/`.docx`. Fallback/cross-check corpus: the **TSpec-LLM** HuggingFace dataset (3GPP specs pre-converted to markdown) — useful to validate our own parser output, not to replace it.

---

## 2. Ingestion — structure-aware parsing

This is where most 3GPP RAG projects quietly fail. A 3GPP spec is not prose; it is a numbered clause tree with tables, ASN.1, and normative keywords.

```
spec.zip
  └─ unzip → .doc / .docx
       └─ (if .doc) LibreOffice: soffice --headless --convert-to docx
            └─ python-docx walk
```

**What the parser must preserve**

1. **Clause hierarchy** — build a tree from Heading 1..6 styles. Every leaf carries a breadcrumb: `5 Procedures > 5.3 Connection control > 5.3.5 RRC reconfiguration > 5.3.5.3 Reception of RRCReconfiguration by the UE`
2. **Tables** → converted to GitHub-flavoured markdown, attached to their owning clause (never orphaned). Parameter tables are where most factual questions land.
3. **ASN.1 blocks** → detected by style/monospace + `::=` pattern, kept verbatim, tagged `type: asn1`, never re-wrapped.
4. **Normative verbs** — `shall` / `should` / `may` / `shall not` are legally meaningful. Never paraphrased anywhere downstream.
5. **Figure captions** — text kept, image dropped.
6. **Notes / Editor's notes** — kept but tagged, so they can be down-weighted.

**What the parser must drop:** foreword, scope boilerplate, change history annexes, revision-marks. These generate high-similarity noise and pollute retrieval.

**Output — one normalized JSONL record per clause:**

```json
{
  "chunk_id": "38331-i80-5.3.5.3-0",
  "spec_id": "TS 38.331",
  "spec_title": "NR; Radio Resource Control (RRC) protocol specification",
  "release": "Rel-18",
  "version": "18.8.0",
  "series": "38",
  "clause_id": "5.3.5.3",
  "clause_title": "Reception of an RRCReconfiguration by the UE",
  "breadcrumb": "5 Procedures > 5.3 Connection control > 5.3.5 RRC reconfiguration",
  "content_type": "prose | table | asn1",
  "text": "...",
  "token_count": 412,
  "source_url": "https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/38331-i80.zip"
}
```

---

## 3. Chunking — the clause is the atom

Rules, in priority order:

1. **Never split across clause boundaries.** A chunk belongs to exactly one clause.
2. Clause ≤ 1000 tokens → **one chunk**.
3. Clause > 1000 tokens → split at paragraph boundaries, ~800 tokens with ~120 token overlap, all children inherit identical clause metadata.
4. Clause < 80 tokens → **merge upward** into the parent clause (avoids useless stub chunks like a lone heading).
5. Tables are **never** split mid-table. An oversized table becomes its own chunk with the clause header repeated.

**Contextual header prepended to the embedded text of every chunk** (this alone is worth several points of retrieval accuracy):

```
[TS 38.331 v18.8.0 | Rel-18]
5 Procedures > 5.3 Connection control > 5.3.5 RRC reconfiguration
§5.3.5.3 Reception of an RRCReconfiguration by the UE
---
<clause text>
```

The header is embedded *and* shown to the LLM, so the model always knows which document and clause a passage came from — it cannot attribute a fact to the wrong spec.

---

## 4. Indexing — hybrid, always

Two indexes over the same chunk store:

- **Dense:** `bge-m3` embeddings → Qdrant collection, cosine, with payload = full metadata (enables filters like `spec_id = TS 23.501`).
- **Sparse:** BM25 over tokenized chunk text, with a telecom-aware tokenizer that does **not** break `gNB-DU`, `5G-GUTI`, `RRCReconfiguration` into fragments.

Why both: a user asking *"what is the IE `t304` used for"* will fail dense-only retrieval (embeddings smear rare identifiers) and succeed instantly on BM25. A user asking *"how does the UE recover when reconfiguration fails"* is the reverse case.

---

## 5. Retrieval pipeline

```
User question
   ↓
[5.1] Query understanding
   ├─ multi-turn condensation → standalone question (Flash-Lite)
   ├─ acronym expansion via TR 21.905 glossary  ("RACH" → "RACH Random Access Channel")
   └─ intent routing: definition | procedure | parameter | comparison | OUT_OF_SCOPE
   ↓
[5.2] Hybrid retrieval
   ├─ dense top-30
   ├─ BM25 top-30
   └─ Reciprocal Rank Fusion → top-20
   ↓
[5.3] Cross-encoder rerank (bge-reranker-v2-m3) → top-6
   ↓
[5.4] ★ RELEVANCE GATE ★
   if max_rerank_score < τ  →  REFUSE. Do not call the generator at all.
   ↓
[5.5] Small-to-big expansion
   each surviving chunk → expand to its full parent clause + breadcrumb, dedupe
   ↓
Context block (numbered [1]..[6], each with its citation header)
```

**§5.4 is the highest-leverage anti-hallucination component in the whole system.** If the corpus does not contain the answer, no LLM call happens, so there is nothing to hallucinate. τ is calibrated empirically on the eval set (§8).

The `OUT_OF_SCOPE` intent route catches the other class early — questions about pricing, vendors, opinions, or non-3GPP topics never reach retrieval.

---

## 6. Generation — constrained and citation-forced

Temperature 0.0–0.1. Structured JSON output.

**System prompt contract:**

- Answer **only** from the numbered CONTEXT passages. Your own telecom knowledge is not admissible.
- Every factual sentence must end with a citation: `[TS 38.331 §5.3.5.3]`.
- Reproduce normative verbs (`shall`, `should`, `may`) exactly as written — never soften or paraphrase them.
- Reproduce IE names, timer names, and parameter values verbatim.
- If the context does not fully answer the question, output `INSUFFICIENT_CONTEXT` and state precisely what is missing.
- Never merge facts across different releases or specs into one claim without citing both.
- No preamble, no "as an AI", no speculation about intent.

**Response schema:**

```json
{
  "answer": "...",
  "citations": [
    {"spec_id": "TS 38.331", "clause_id": "5.3.5.3", "version": "18.8.0", "chunk_id": "..."}
  ],
  "sufficient_context": true,
  "unanswered_aspects": []
}
```

Structured output matters because it makes the next layer *programmatic* rather than vibes-based.

---

## 7. Verification layer — the "near-zero" claim

Runs **after** generation, **before** the answer reaches the user. Three checks:

**7.1 Citation validity (deterministic, free)**
Every `chunk_id` / clause the model cited must exist in the retrieved set. A citation the model invented is caught by a set-membership test — no LLM needed. Invalid citation → regenerate once, then downgrade.

**7.2 Groundedness / NLI check (the real one)**
Decompose the answer into atomic claims. For each claim, test entailment against the retrieved context:
- Primary: a local NLI cross-encoder (fast, free, GPU).
- Secondary / tie-break: Flash-Lite as a strict judge returning `SUPPORTED | PARTIAL | UNSUPPORTED` per claim.

Any `UNSUPPORTED` claim is removed from the answer or the whole answer is downgraded.

**7.3 Abstention policy**
`groundedness_score = supported_claims / total_claims`

| Score | Behaviour |
|---|---|
| ≥ 0.9 | Return answer, badge **Grounded** |
| 0.6–0.9 | Return answer with unsupported sentences stripped, badge **Partially grounded** |
| < 0.6 | **Refuse.** Return: "I could not find this in the indexed 3GPP specifications." + show the closest passages so the user can judge for themselves |

**7.4 Auditability as a control**
Every answer ships with expandable source snippets showing the exact retrieved text. A system whose evidence is one click away is a system that cannot bluff — and it is also the single most convincing thing to show an evaluator live.

---

## 8. Evaluation — this is what gets graded

The brief grades "quality and effectiveness". That means measurable numbers, not a working demo alone.

**Gold set (~70 questions), built semi-automatically then human-verified:**
- 40 answerable questions, each with a known ground-truth clause (LLM drafts a question from a clause; you verify)
- 15 **adversarial / out-of-scope** ("What does TS 99.999 specify?", "What does a gNB cost?", "Summarise the Rel-25 roadmap") — the correct answer is a refusal
- 15 multi-hop / cross-spec (needs 23.501 + 23.502 together)

**Metrics**

| Layer | Metric |
|---|---|
| Retrieval | Recall@5, Recall@10, MRR, gold-clause hit rate |
| Answer | Faithfulness, answer relevance, citation precision |
| **Headline** | **Hallucination rate** = unsupported claims / total claims |
| **Headline** | **Correct-refusal rate** on the adversarial set |
| System | p50 / p95 latency, cost per query |

**Ablation table — build this, it is the strongest interview artifact you can bring:**

| Configuration | Recall@5 | Faithfulness | Hallucination rate | Correct refusal |
|---|---|---|---|---|
| Naive RAG (dense only, no rerank) | | | | |
| + hybrid (dense + BM25 + RRF) | | | | |
| + cross-encoder rerank | | | | |
| + relevance gate | | | | |
| + verification layer (**final**) | | | | |

Each row is a claim you can defend for ten minutes in the technical interview. That is the actual deliverable.

---

## 9. Repo layout

```
rag-3gpp/
├── README.md                 # architecture, results, ablations, demo GIF
├── ARCHITECTURE.md           # this document
├── requirements.txt
├── .env.example              # GEMINI_API_KEY=
├── docker-compose.yml        # qdrant
├── data/
│   ├── raw/                  # downloaded zips
│   ├── docx/                 # extracted
│   ├── parsed/               # clause-tree JSONL
│   └── chunks/               # final chunks JSONL
├── src/
│   ├── config.py             # all thresholds in one place
│   ├── ingest/
│   │   ├── download.py       # 3GPP FTP fetcher
│   │   ├── convert.py        # .doc → .docx
│   │   ├── parse_docx.py     # clause tree, tables, ASN.1
│   │   └── chunk.py          # structure-aware chunking
│   ├── index/
│   │   ├── embedder.py       # bge-m3
│   │   ├── vector_store.py   # Qdrant
│   │   └── bm25_store.py
│   ├── retrieval/
│   │   ├── query_processor.py  # condense, expand acronyms, route intent
│   │   ├── hybrid.py           # RRF fusion
│   │   ├── reranker.py
│   │   └── gate.py             # relevance threshold
│   ├── generation/
│   │   ├── prompts.py
│   │   └── generator.py        # Gemini, structured output
│   ├── verification/
│   │   ├── citation_check.py
│   │   ├── groundedness.py     # NLI + LLM judge
│   │   └── policy.py           # abstention thresholds
│   ├── api/
│   │   └── main.py             # FastAPI: /chat /search /health /ingest
│   └── ui/
│       └── app.py              # Streamlit + transparency panel
├── eval/
│   ├── gold_set.jsonl
│   ├── run_eval.py
│   └── results/                # ablation tables, charts
└── tests/
```

---

## 10. Build plan — Aug 14 → Aug 17

**Day 1 (Thu 14 Aug) — corpus and retrieval spine**
- Env, repo skeleton, Qdrant up
- `download.py` → pull the 14 specs
- `parse_docx.py` → clause tree with tables + ASN.1 (budget the most time here; it is the hard part)
- `chunk.py` → chunks JSONL, sanity-check 20 chunks by hand
- Embed + index into Qdrant, build BM25
- **Exit criteria:** a CLI query returns sensible top-10 chunks

**Day 2 (Fri 15 Aug) — the actual RAG**
- Hybrid + RRF, cross-encoder rerank, relevance gate
- Prompt engineering + Gemini structured generation with citations
- Verification layer (citation check → groundedness → abstention)
- FastAPI endpoints, Streamlit UI with transparency panel
- **Exit criteria:** end-to-end chat working, refuses out-of-scope questions

**Day 3 (Sat 16 Aug) — the grade**
- Build and verify the 70-question gold set
- Run eval, calibrate τ and the abstention thresholds
- Run all 5 ablation configurations, produce the table + charts
- README with architecture diagram, results, design rationale, limitations
- Screen-record a 3-minute demo (include a live refusal — that is the money shot)

**Sun 17 Aug — submit.** Buffer day, not a build day.

---

## 11. What to say in the interview

Have a crisp answer ready for each:

1. *Why is hallucination low?* → Four independent controls: version pinning, the relevance gate (no context → no LLM call), forced citations validated programmatically, and NLI groundedness with abstention. Cite your measured numbers.
2. *Why hybrid retrieval?* → Show a query where dense fails and BM25 saves it (`t304`), and the reverse.
3. *Why chunk by clause?* → 3GPP semantics are clause-scoped; splitting mid-clause severs a `shall` from its condition, which is exactly how a RAG system starts producing dangerous, confident, wrong answers.
4. *What are the limits?* → Cross-release comparison, figure-heavy content, and deep ASN.1 reasoning. Name them before the interviewer does.
5. *How would you scale it?* → Full 3GPP corpus, per-release collections with metadata filtering, batch reindex on new releases, caching, and a feedback loop on thumbs-down queries.
