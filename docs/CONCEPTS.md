# 3GPP RAG Chatbot — Complete Concept Handbook

Everything used in the design, explained from first principles. Read this before writing code.

**How to use this:** Part 0–1 are mandatory context. Parts 2–8 map 1:1 onto the pipeline stages. Part 9–10 are serving and grading. Part 11 is the interview drill. Anything marked **[INTERVIEW]** is a question they are likely to actually ask.

---

# PART 0 — Why RAG exists at all

## 0.1 Parametric vs non-parametric memory

An LLM stores knowledge in its weights. That is **parametric memory**. It has four fatal properties for a standards-QA task:

| Property | Consequence for 3GPP |
|---|---|
| Frozen at training cutoff | Rel-19 content may not exist in the model |
| Lossy compression | The model "remembers the gist" of TS 38.331, not the exact timer value |
| No provenance | It cannot tell you which clause a fact came from |
| No abstention signal | It does not know that it does not know |

**RAG** adds **non-parametric memory**: an external, updatable, citable text store. At query time we fetch the relevant text and put it *in the context window*, so the model reads rather than recalls.

The mental shift: we are converting the task from **"recall this fact"** (which LLMs do badly and confidently) into **"read this passage and extract the answer"** (which LLMs do very well).

## 0.2 What hallucination actually is — mechanically

An LLM is a next-token probability distribution: `P(token_t | token_1 ... token_{t-1})`. It is trained to maximise likelihood of plausible continuations. **There is no truth term anywhere in the objective function.**

So when you ask "what is the default value of timer t304 in NR?", the model produces a token sequence that *looks like* a correct answer — a plausible number, in the right format, with confident phrasing. Fluency and factuality are produced by the same machinery, which is exactly why hallucinations are so hard to spot.

Two distinct causes:

1. **Knowledge gap** — the fact was never in training, or was compressed away. Model interpolates.
2. **Context override failure** — the fact *is* in the provided context, but the model's parametric prior is stronger and contradicts it, so it answers from memory instead of the page.

RAG attacks (1). Prompt constraints, low temperature, and the verification layer attack (2).

## 0.3 The RAG failure taxonomy **[INTERVIEW]**

When a RAG system gives a wrong answer, exactly one of these happened. Know which one you are fixing at each layer:

| # | Failure | Where it happens | Our fix |
|---|---|---|---|
| F1 | The answer isn't in the corpus | Corpus scope | Relevance gate → refuse (Part 6.5) |
| F2 | It's in the corpus but retrieval missed it | Chunking / embedding / search | Hybrid + rerank (Parts 3–6) |
| F3 | Retrieved, but ranked below the cutoff | Ranking | Cross-encoder rerank (Part 6.3) |
| F4 | Retrieved and ranked, but chunk is truncated mid-fact | Chunking | Clause-atomic chunking (Part 3) |
| F5 | Context is perfect, model still invents | Generation | Constrained prompt + temp 0 (Part 7) |
| F6 | Answer is right but citation is wrong | Generation | Deterministic citation check (Part 8.1) |
| F7 | Model should have refused but answered anyway | Policy | Abstention thresholds (Part 8.3) |

The single most useful debugging habit: **before blaming the LLM, check whether the gold passage was even in the retrieved set.** ~70% of "the LLM hallucinated" bugs are actually F2/F3.

---

# PART 1 — The domain: 3GPP specifications

You cannot build good retrieval over documents you do not understand structurally.

## 1.1 What 3GPP is

3GPP (3rd Generation Partnership Project) is the standards body that writes the specifications for 3G/4G/5G. Its output is a large set of numbered documents, all **freely downloadable**, no login.

Two document types:

- **TS — Technical Specification.** Normative. This is the law. Implementations must comply.
- **TR — Technical Report.** Informative. Studies, background, vocabulary. Not binding.

## 1.2 Numbering — series, spec, version

**Spec number: `TS 38.331`**
- `38` = the **series** (a subject area)
- `331` = the document within that series

Series you care about:

| Series | Subject |
|---|---|
| 21 | Requirements, vocabulary (TR 21.905) |
| 22 | Service requirements |
| 23 | Technical realisation, system architecture (23.501, 23.502) |
| 24 | Signalling between UE and network — NAS (24.501) |
| 33 | Security (33.501) |
| 36 | LTE / E-UTRAN |
| 38 | **5G NR / NG-RAN** |

**Version: `V18.8.0` = x.y.z**
- `x` — increments with the **Release** the spec has been brought under
- `y` — technical content change
- `z` — editorial change only

**Releases** are 3GPP's feature freeze cycles: Rel-15 was the first 5G release, Rel-16, Rel-17, Rel-18 ("5G-Advanced"), Rel-19...

## 1.3 The filename encoding — a genuinely useful minute detail

FTP filenames look like `38331-i80.zip`. Those three characters are `x`,`y`,`z` of the version, where each character is a **base-36-ish digit**: `0-9` then `a=10, b=11, ... z=35`.

So:

```
38331-i80.zip  →  i = 18, 8 = 8, 0 = 0  →  TS 38.331 V18.8.0
38331-j30.zip  →  j = 19, 3 = 3, 0 = 0  →  TS 38.331 V19.3.0
38331-f00.zip  →  f = 15               →  TS 38.331 V15.0.0
```

And since `x` tracks the Release: `f`→Rel-15, `g`→Rel-16, `h`→Rel-17, `i`→Rel-18, `j`→Rel-19.

Your downloader parses this to pick exactly one release and record the version in metadata. **This is control #1 — version pinning.**

## 1.4 Anatomy of a spec — why naive RAG dies here

A 3GPP TS is **not prose**. It is a numbered clause tree containing several distinct content types, and a generic text splitter destroys all of them.

**(a) Clause hierarchy.** Everything is addressed by number:

```
5     Procedures
5.3   Connection control
5.3.5 RRC reconfiguration
5.3.5.1  General
5.3.5.2  Initiation
5.3.5.3  Reception of an RRCReconfiguration by the UE
```

The clause number *is* the citation format engineers use. If your chunks don't carry it, you cannot cite properly, and an uncitable RAG system is a hallucinating RAG system with extra steps.

**(b) Normative keywords.** 3GPP drafting rules (TR 21.801) give these precise legal meaning:

| Word | Meaning |
|---|---|
| `shall` | Mandatory requirement |
| `shall not` | Absolute prohibition |
| `should` | Recommendation — deviation allowed with justification |
| `may` | Permission, truly optional |
| `can` / `cannot` | Statement of capability, **not** a requirement |

If your LLM paraphrases "the UE shall initiate" into "the UE initiates", it has silently converted a mandatory requirement into a description. In a telecom review that is a serious error. **This is why the prompt forbids paraphrasing normative verbs.**

**(c) ASN.1 blocks.** ASN.1 (Abstract Syntax Notation One) is the formal language used to define RRC and NAS message structures. TS 38.331 is roughly half ASN.1:

```asn1
RRCReconfiguration-IEs ::= SEQUENCE {
    radioBearerConfig       RadioBearerConfig    OPTIONAL,
    secondaryCellGroup      OCTET STRING         OPTIONAL,
    measConfig              MeasConfig           OPTIONAL,
    ...
}
```

Properties that matter to you: it is whitespace/structure sensitive, `::=` is the definition operator, `OPTIONAL` and `SEQUENCE` are keywords, and **splitting an ASN.1 block mid-definition produces syntactically invalid garbage that embeds meaninglessly.** Detect it, keep it verbatim, tag it, never re-wrap it.

**(d) Tables.** Parameter values, timer ranges, and IE descriptions live in tables. A huge share of factual questions ("what is the range of t304?") are answered only by a table cell. If your parser drops tables — and `docx2txt`-style naive extraction does — those questions become unanswerable, and the LLM will happily invent a value.

**(e) IE names and identifiers.** `RRCReconfiguration`, `gNB-DU`, `5G-GUTI`, `t304`, `PDU Session ID`. These are camelCase/hyphenated rare tokens. **Embedding models are bad at them** (they get split into meaningless subword fragments and semantically smeared). This is the single strongest technical argument for hybrid retrieval — see Part 5.

**(f) Boilerplate to drop.** Foreword, scope, "the present document...", change history annexes. Every spec has near-identical text here, which means these chunks have high similarity to *everything* and pollute your top-k. Drop them at parse time.

---

# PART 2 — Ingestion: getting text out of Word

## 2.1 .doc vs .docx — two completely different formats

| | `.doc` | `.docx` |
|---|---|---|
| Era | Word 97–2003 | Word 2007+ |
| Format | **OLE2 Compound File** — a binary filesystem-in-a-file | **OOXML** — a ZIP archive of XML files |
| Parseable in Python | Painfully (`antiword`, `textract`) | Cleanly (`python-docx`) |

Older 3GPP specs ship as `.doc`. So the pipeline normalises everything to `.docx` first:

```bash
soffice --headless --convert-to docx --outdir ./docx input.doc
```

LibreOffice in headless mode. It is slow (~seconds per doc) but it is the only reliable converter, and you run it once, offline.

## 2.2 What a .docx actually is

Rename any `.docx` to `.zip` and open it. Inside:

```
word/document.xml    ← the content
word/styles.xml      ← style definitions (Heading 1, Normal, ...)
word/numbering.xml   ← list numbering
_rels/               ← relationships (images, links)
```

`document.xml` is a tree of paragraphs and runs:

```xml
<w:p>                              <!-- paragraph -->
  <w:pPr><w:pStyle w:val="Heading3"/></w:pPr>   <!-- paragraph properties: its style -->
  <w:r><w:t>Reception of an RRCReconfiguration</w:t></w:r>  <!-- run: text with formatting -->
</w:p>
```

Key concept: **a paragraph has a style name.** That is how you recover the clause hierarchy — you read `Heading1`...`Heading6` styles rather than trying to regex the numbers out of raw text.

## 2.3 python-docx — what you'll actually use

```python
from docx import Document
doc = Document("38331-i80.docx")

for block in doc.element.body:      # iterate in document order
    ...
for p in doc.paragraphs:
    p.style.name    # "Heading 3", "Normal", ...
    p.text          # plain text of the paragraph
for t in doc.tables:
    t.rows[i].cells[j].text
```

**The trap:** `doc.paragraphs` and `doc.tables` are two *separate* lists. Iterating them separately loses interleaving — you no longer know which clause a table belonged to. You must **walk `doc.element.body` in document order**, checking each element's tag (`w:p` vs `w:tbl`), so a table stays attached to the clause it appeared under. Getting this wrong is the most common 3GPP-RAG parsing bug.

## 2.4 Clause tree construction — the algorithm

```
stack = []                  # holds (level, number, title)
current_clause = None

for element in body_in_document_order:
    if element is a heading (style = "Heading N"):
        level = N
        number, title = split_leading_number(element.text)   # "5.3.5.3", "Reception of..."
        pop stack while stack.top.level >= level
        push (level, number, title)
        current_clause = new Clause(
            id        = number,
            title     = title,
            breadcrumb= " > ".join(f"{lvl.number} {lvl.title}" for lvl in stack[:-1])
        )
    elif element is a paragraph:
        if looks_like_asn1(element): current_clause.add(text, type="asn1")
        else:                        current_clause.add(text, type="prose")
    elif element is a table:
        current_clause.add(table_to_markdown(element), type="table")
```

`looks_like_asn1` heuristics: monospace style (`Courier`, or 3GPP's custom `ASN.1` style), presence of `::=`, or lines matching `^\s*\w+\s+(SEQUENCE|CHOICE|ENUMERATED|INTEGER|BIT STRING)`.

`table_to_markdown`: emit a GFM pipe table, escape internal `|`, and keep the header row — the header row is what makes the table self-describing when it lands in an LLM's context.

## 2.5 Why JSONL as the intermediate format

One JSON object per line. Advantages: streamable (you never load 14 specs into RAM), append-only, `wc -l` gives you a count, and each stage of the pipeline reads one JSONL and writes another. You can inspect stage output with `head -n 3 parsed/38331.jsonl | jq`. Debuggability is the whole point — you *will* need to eyeball chunks.

---

# PART 3 — Chunking

## 3.1 Why chunk at all

Three independent reasons, and people usually only know the first:

1. **Context limits.** You cannot put 14 specs into a prompt. (Weaker argument now — Gemini has 1M context — but see 2 and 3.)
2. **Embedding dilution.** An embedding is a *fixed-size* vector, e.g. 1024 floats. Embedding a 50-page document into 1024 numbers averages away everything specific. The longer the text, the blurrier the vector. Small, focused chunks have sharp vectors.
3. **Precision of the final context.** Even with a 1M window, "lost in the middle" (Liu et al., 2023) shows model accuracy drops sharply for facts buried in the middle of a long context. Feeding 6 precise passages beats feeding 200 pages — **more context is not better context.**

## 3.2 Chunking strategies, weakest to strongest

| Strategy | How | Problem |
|---|---|---|
| **Fixed-size** | Every 512 tokens | Splits mid-sentence, mid-table, mid-requirement. Baseline only. |
| **Recursive character** | Split on `\n\n`, then `\n`, then `.` until under size | LangChain default. Better, but structure-blind — has no idea what a clause is. |
| **Semantic** | Embed sentences, cut where consecutive similarity drops | Expensive, and on technical text the similarity signal is weak and noisy. |
| **Structure-aware** ← **ours** | Use the document's own hierarchy as boundaries | Requires a real parser. That's the work in Part 2 — and it's why our parser matters. |

## 3.3 Our rules, and the reasoning behind each

**Rule 1 — Never split across clause boundaries. One chunk belongs to exactly one clause.**

The reason, and this is the sentence to say in the interview: *3GPP requirements are clause-scoped. A `shall` and the condition that triggers it live in the same clause. Split them and you retrieve an unconditional requirement — the system will confidently tell an engineer something is always mandatory when it is only mandatory in one specific state.* That is not a cosmetic bug, it is a dangerous one.

**Rule 2 — Clause ≤ 1000 tokens → one chunk.** Most clauses fit. Keep them whole.

**Rule 3 — Clause > 1000 tokens → split at paragraph boundaries, ~800 tokens, ~120 overlap.** All children inherit identical clause metadata, so citations stay correct regardless of which child is retrieved.

*Why overlap?* A fact near a split boundary would otherwise have its context severed in both pieces. ~15% overlap means it appears whole in at least one chunk. Cost: index size grows ~15%, and you get near-duplicate hits (handled by dedup at Part 6.5).

**Rule 4 — Clause < 80 tokens → merge upward into parent.** Prevents junk chunks that are just a heading, which match everything weakly and waste top-k slots.

**Rule 5 — Never split a table.** An oversized table becomes its own chunk with the clause header repeated. A half-table is worse than no table: it has the header row and the wrong rows, and it looks authoritative.

## 3.4 Tokens vs characters — know the difference

LLMs and embedding models operate on **tokens**, produced by a subword tokenizer (BPE, WordPiece, or SentencePiece). Roughly:

```
1 token ≈ 4 characters ≈ 0.75 English words
```

But technical text tokenizes *badly*. `RRCReconfiguration` is not one token — it becomes something like `RR` `C` `Recon` `fig` `uration`, 5 tokens for one concept. This matters twice:

- **Budgeting:** measure with the actual tokenizer (`tiktoken`, or the model's own), never `len(text)/4`.
- **Retrieval quality:** those fragments are why dense embeddings smear rare identifiers → Part 5.

## 3.5 Contextual headers (Contextual Retrieval)

Every chunk gets this prepended **before embedding**:

```
[TS 38.331 v18.8.0 | Rel-18]
5 Procedures > 5.3 Connection control > 5.3.5 RRC reconfiguration
§5.3.5.3 Reception of an RRCReconfiguration by the UE
---
<clause text>
```

Why it works, three separate mechanisms:

1. **Retrieval.** A raw chunk reading "the UE shall perform the actions in..." is nearly unretrievable — it has no topical anchor. With the header, the chunk's vector now sits near "RRC reconfiguration" in embedding space.
2. **Disambiguation.** "Initiation" appears as a clause title in dozens of places. The breadcrumb makes each one distinct.
3. **Generation grounding.** The model *sees* the header, so it always knows which spec and clause a passage came from. It cannot accidentally attribute a 23.501 fact to 38.331.

This technique (Anthropic's "Contextual Retrieval", 2024) reduced retrieval failure by ~35% in the original write-up, and ~49% combined with hybrid + rerank. It is the highest ratio of benefit-to-effort in this entire pipeline.

## 3.6 Small-to-big retrieval

Two different jobs, two different sizes:

- **Retrieve** on small chunks → sharp, precise vectors → accurate matching.
- **Generate** on big context → the full parent clause → complete information.

So the flow is: match a small chunk, then look up its `clause_id` and feed the **entire clause** to the LLM. You get the precision of small chunks *and* the completeness of large ones. Also called parent-document retrieval.

---

# PART 4 — Embeddings and vector search

## 4.1 What an embedding is

A neural network that maps text → a fixed-length vector of floats (bge-m3: **1024 dimensions**), trained so that **semantically similar texts land close together** in that 1024-D space.

Training is typically contrastive: given (query, positive passage, negative passages), pull query and positive together, push negatives apart. That objective is *exactly* "make search work", which is why these models beat generic language-model embeddings at retrieval.

## 4.2 Similarity metrics

**Cosine similarity** — the angle between vectors, magnitude-independent:

```
cos(a,b) = (a · b) / (‖a‖ ‖b‖)        range [-1, 1], higher = more similar
```

**Dot product** — `a · b`. Fast, but magnitude-sensitive.

**Key optimisation:** if you **L2-normalise** every vector at index time (`‖v‖ = 1`), then cosine similarity *is* the dot product. So you get cosine semantics at dot-product speed. Always normalise. Also note Euclidean distance on normalised vectors is a monotonic function of cosine, so all three rank identically once normalised.

## 4.3 Bi-encoder vs cross-encoder — the central distinction **[INTERVIEW]**

This is the concept that makes reranking make sense.

**Bi-encoder** (what an embedding model is):

```
query    → [encoder] → vec_q  ─┐
                               ├→ cosine → score
passage  → [encoder] → vec_p  ─┘
```

Query and passage are encoded **independently**. Passage vectors are precomputed once at index time, so search is just a vector comparison. Fast enough for millions of documents. **But the model never sees the query and the passage together** — it must compress the passage into one vector without knowing what will be asked. Information is necessarily lost.

**Cross-encoder** (what a reranker is):

```
[CLS] query [SEP] passage [SEP]  → [transformer, full self-attention] → relevance score
```

Query and passage go in **together**. Every query token attends to every passage token. The model can directly check "does this passage answer *this specific* question?" Much more accurate.

**The cost:** nothing can be precomputed. Scoring N passages needs N forward passes. Scoring a million documents per query is impossible.

**Therefore the standard architecture, and the reason our pipeline has two stages:**

```
1M docs ──[bi-encoder, fast, approximate]──> 20 candidates ──[cross-encoder, slow, accurate]──> 6
```

Recall first, precision second. Cheap model casts a wide net; expensive model picks the winners.

## 4.4 bge-m3 specifically

`BAAI/bge-m3` — "M3" for **M**ulti-lingual, **M**ulti-functionality, **M**ulti-granularity.

- **Multi-functionality:** produces three representations in one forward pass — dense (1024-d), sparse (learned lexical weights, like a neural BM25), and multi-vector (ColBERT-style, one vector per token). You'll use dense; the sparse output is a bonus you can fuse in if time permits.
- **Multi-granularity:** handles up to **8192 tokens** — comfortably more than our 1000-token chunks, so nothing is ever truncated.
- Strong on technical and mixed-language text, and it runs fine on your GPU.

**Critical implementation detail:** BGE models were trained with an **instruction prefix on queries but not on passages**. For retrieval you prepend something like `"Represent this sentence for searching relevant passages: "` to queries only. Getting this wrong silently costs you several points of recall — and it is a classic interview "gotcha".

## 4.5 ANN search and HNSW

Comparing a query against every vector (**exact / flat / brute-force** search) is `O(n)`. Fine for 50k chunks honestly — but production systems use **approximate nearest neighbour (ANN)**, and you should be able to explain it.

**HNSW — Hierarchical Navigable Small World** (what Qdrant uses):

Build a multi-layer graph. The top layer is sparse with long-range links; lower layers get progressively denser. Search starts at the top, greedily walks toward the query, then drops a layer and refines. Like a skip-list for vectors: `O(log n)` instead of `O(n)`.

Parameters you will be asked about:

| Param | Meaning | Trade-off |
|---|---|---|
| `M` | Edges per node | Higher → better recall, more memory |
| `ef_construct` | Candidate list size at build | Higher → better graph, slower indexing |
| `ef_search` | Candidate list size at query | Higher → better recall, slower queries |

"Approximate" means you may miss a true nearest neighbour. At our scale (~50k chunks) recall is ~0.99+ with defaults. **Honest answer for the interview: at this corpus size ANN is not necessary; I use it because it's what scales, and the recall cost is negligible.**

## 4.6 Qdrant concepts

- **Collection** — like a table. Has a fixed vector size and distance metric.
- **Point** — one record: `id` + `vector` + **`payload`** (arbitrary JSON metadata).
- **Payload filtering** — this is the reason to use a real vector DB. `spec_id == "TS 23.501" AND release == "Rel-18"` is applied *during* the graph traversal, not as a post-filter, so you still get a full top-k after filtering. This is what lets a user say "only search the core network specs."
- **Persistence** — survives restarts. Your 40-minute embedding job runs once.
- **Quantization** — scalar/binary compression of vectors to cut memory. Not needed at our scale; know it exists.

---

# PART 5 — Lexical retrieval: BM25

## 5.1 Why you cannot skip this

The one-line argument: **dense retrieval fails on rare exact tokens, and 3GPP is made of rare exact tokens.**

Consider the query `"what is timer t304"`. `t304` tokenizes to something like `t` `30` `4`. There is almost no semantic signal there — the embedding of `t304` is close to the embedding of `t310`, `t311`, `t390`, and every other timer. Dense retrieval returns "a passage about timers." Wrong timer, confidently retrieved.

BM25 does exact term matching with a rarity weight. `t304` is a very rare term, so it gets a huge IDF weight, and the one clause that actually contains it rockets to rank 1.

Now the reverse case: `"how does the UE recover when reconfiguration fails"`. The spec says "upon reconfiguration failure, the UE shall initiate re-establishment." The word "recover" never appears. BM25 scores near zero — this is **vocabulary mismatch**, the classic weakness of lexical search. Dense retrieval handles it easily because "recover" and "re-establishment" are close in embedding space.

**Each method fails exactly where the other succeeds. That is the entire argument for hybrid.** Have both examples ready — a concrete failure case is far more convincing than "hybrid is best practice."

## 5.2 TF-IDF → BM25

**TF-IDF** intuition: a term matters more if it appears often in this document (**TF**) and rarely across the corpus (**IDF**). Problem: TF is linear, so a document mentioning `t304` 50 times scores 50× one mentioning it once — which is wrong, the 10th mention adds almost no information.

**BM25** ("Best Match 25") fixes this with saturation and length normalisation:

```
                    f(qᵢ,D) · (k₁ + 1)
score(D,Q) = Σ IDF(qᵢ) · ────────────────────────────────────
              qᵢ∈Q        f(qᵢ,D) + k₁ · (1 − b + b · |D|/avgdl)

              N − n(qᵢ) + 0.5
IDF(qᵢ) = ln( ─────────────── + 1 )
               n(qᵢ) + 0.5
```

| Symbol | Meaning |
|---|---|
| `f(qᵢ,D)` | Times term `qᵢ` occurs in document `D` |
| `\|D\|` | Length of `D` in terms |
| `avgdl` | Average document length in the corpus |
| `N` | Total documents |
| `n(qᵢ)` | Documents containing `qᵢ` |
| `k₁` | Saturation control, typically **1.2–2.0** |
| `b` | Length-normalisation strength, typically **0.75** |

Read the formula through its two knobs:

- **`k₁` — saturation.** As `f` grows, the fraction asymptotes to `k₁+1` instead of growing linearly. The 2nd occurrence adds a lot; the 20th adds almost nothing. `k₁=0` ignores frequency entirely; higher `k₁` means frequency matters more.
- **`b` — length normalisation.** Long documents contain more terms by chance, so they'd win unfairly. `b=1` fully normalises by length; `b=0` disables it; `0.75` is the standard compromise. **Note: because our chunks are clause-sized and fairly uniform, `b` matters less for us than in general web search — a good thing to say if asked.**

## 5.3 The telecom tokenizer — a real bug you must avoid

Default tokenizers split on punctuation. That destroys you here:

```
"gNB-DU"              → ["gnb", "du"]          ✗ now matches "gNB-CU" too
"5G-GUTI"             → ["5g", "guti"]         ✗
"RRCReconfiguration"  → ["rrcreconfiguration"] ✓ (kept, lowercased)
"TS 38.331"           → ["ts", "38", "331"]    ✗ "38" and "331" match everything
```

Your tokenizer must:

1. Preserve hyphenated identifiers as single tokens (`gnb-du`).
2. Preserve dotted spec numbers (`38.331` as one token).
3. Optionally *also* emit a camelCase-split version (`rrc reconfiguration`) as an **extra** token alongside the original — so both `RRCReconfiguration` and "RRC reconfiguration" match. Belt and braces.
4. Lowercase, and be careful with stemming — `"RRC"` must not stem into something else. Honestly, **skip stemming here**; on identifier-heavy text it does more harm than good.

---

# PART 6 — Fusion, reranking, gating

## 6.1 The problem with combining scores

Dense gives cosine ∈ [-1,1]. BM25 gives an unbounded positive score that depends on corpus statistics. `0.7 + 14.2` is meaningless. Min-max normalising each is possible but fragile — one outlier document rescales everything, and the distributions shift per query.

## 6.2 Reciprocal Rank Fusion — use ranks, not scores

RRF throws away the scores entirely and fuses **ranks**:

```
                       1
RRF(d) =   Σ      ──────────────
         i∈lists   k + rankᵢ(d)

k = 60 (standard)
```

Worked example — document `d` is rank 1 in dense and rank 8 in BM25:

```
RRF(d) = 1/(60+1) + 1/(60+8) = 0.01639 + 0.01471 = 0.03110
```

Why it works:

- **Scale-free.** Ranks are comparable across any retrievers, no calibration needed.
- **`k=60` damps the top.** Without `k`, rank 1 scores 1.0 and rank 2 scores 0.5 — a brutal cliff where one retriever's top hit dominates. With `k=60`, the difference between rank 1 and rank 2 is small, so **agreement across retrievers matters more than being #1 in any single one.** A document ranked 3rd by both beats a document ranked 1st by one and 40th by the other. That consensus behaviour is exactly what you want.
- Trivial to implement, no training, no tuning.

## 6.3 Cross-encoder reranking

Take the fused top-20 and score each with `BAAI/bge-reranker-v2-m3`:

```python
pairs  = [(query, chunk.text) for chunk in candidates]
scores = reranker.predict(pairs)      # 20 forward passes, batched on GPU
top6   = sorted(zip(candidates, scores), reverse=True)[:6]
```

Mechanically (per Part 4.3): full self-attention across query **and** passage together, so the model can verify actual question-answering, not just topical similarity. This is typically the **largest single precision improvement in the whole pipeline** — commonly +10–20 points of nDCG over embedding-only retrieval.

Cost: 20 transformer forward passes. On your GPU, batched, this is roughly 50–150 ms. Completely acceptable. This is why the funnel is 30+30 → 20 → 6 and not 1000 → 6.

## 6.4 Why these specific numbers

| Stage | N | Reasoning |
|---|---|---|
| Dense / BM25 | 30 each | Wide enough that the gold passage is almost certainly in there (high recall). Cheap. |
| After RRF | 20 | Reranking budget. 20 × ~40ms is fine; 200 would not be. |
| After rerank | 6 | Context budget + "lost in the middle". 6 clause-expanded passages ≈ 4–6k tokens. Enough evidence, small enough that the model reads all of it. |

These are starting values. **Tune them on the eval set** — that sentence alone signals engineering maturity.

## 6.5 The relevance gate — control #2 **[INTERVIEW]**

After reranking:

```python
if max(rerank_scores) < TAU:
    return Refusal(
        message="I could not find this in the indexed 3GPP specifications.",
        closest_passages=top3          # transparency: let the user judge
    )
```

**Say this in the interview:** *"If the corpus does not contain the answer, no LLM call happens at all. There is no generation step, therefore there is literally nothing to hallucinate. Most RAG systems always call the LLM and rely on the prompt to make it refuse — that is a soft, probabilistic control. Mine is a hard one."*

**Calibrating τ.** Do not guess it. Run the eval set, collect the max rerank score for every question, and plot two distributions:

- answerable questions → scores should cluster high
- adversarial/out-of-scope questions → scores should cluster low

Pick τ where they separate. Then walk the trade-off explicitly:

- τ too low → out-of-scope questions leak through → hallucinations
- τ too high → answerable questions get refused → useless system

This is a **precision/coverage trade-off**, and being able to draw that curve is a strong signal. Note `bge-reranker-v2-m3` outputs raw logits (roughly −10 to +10); apply a sigmoid if you want a 0–1 number, but calibrate on whichever you use.

**After the gate — small-to-big expansion and dedup.** Expand each surviving chunk to its full parent clause (Part 3.6), then deduplicate: overlapping sibling chunks (Rule 3) frequently expand to the *same* clause. Dedupe by `clause_id` or you will waste 3 of your 6 slots on the same text.

---

# PART 7 — Generation

## 7.1 Decoding parameters

| Param | What it does | Ours |
|---|---|---|
| `temperature` | Scales logits before softmax. →0 = greedy/deterministic; →1 = creative | **0.0–0.1** |
| `top_p` | Nucleus sampling: sample only from the smallest token set with cumulative prob ≥ p | 0.95 (irrelevant at temp 0) |
| `top_k` | Sample only from the k most likely tokens | n/a |

**Temperature 0 is not optional here.** Sampling means picking a lower-probability token, and a lower-probability token in a factual answer is, quite literally, a less likely fact. Creativity and hallucination are the same knob. We also want **reproducibility** — the same question must give the same answer, or your eval numbers are noise.

## 7.2 Prompt architecture

Three blocks, in this order:

```
[SYSTEM]   Role + hard rules + output contract
[CONTEXT]  Numbered passages, each with its citation header
[USER]     The question
```

The rules, and what each one is defending against:

| Rule | Defends against |
|---|---|
| "Answer **only** from CONTEXT; your own telecom knowledge is not admissible" | F5 — parametric override |
| "Every factual sentence ends with `[TS 38.331 §5.3.5.3]`" | Unverifiable claims; enables Part 8.1 |
| "Reproduce `shall`/`should`/`may` exactly" | Normative meaning drift (Part 1.4b) |
| "Reproduce IE names, timers, values verbatim" | Plausible-looking wrong identifiers |
| "If context is insufficient, output `INSUFFICIENT_CONTEXT` and state what's missing" | F7 — answering when it should abstain |
| "Never merge facts across specs/releases without citing both" | Cross-document contamination |

**Why forcing citations reduces hallucination even before you validate them:** to attach `[TS 38.331 §5.3.5.3]` to a sentence, the model must locate that sentence's support in the context. It shifts the task from *generation* to *extraction with attribution*. That constraint alone measurably reduces unsupported claims — and then Part 8.1 makes it airtight by checking.

## 7.3 Context ordering and "lost in the middle"

Liu et al. (2023) showed a **U-shaped** accuracy curve: models attend best to the beginning and end of a long context, and worst to the middle.

Practical consequences:
- Put the highest-reranked passage **first**.
- Consider putting rank 2 **last** (both high-attention positions).
- Keep the total small — 6 passages, not 50. This is a second, independent reason for aggressive reranking.

## 7.4 Structured output

Ask Gemini for JSON conforming to a schema:

```json
{
  "answer": "...",
  "citations": [{"spec_id":"TS 38.331","clause_id":"5.3.5.3","version":"18.8.0","chunk_id":"..."}],
  "sufficient_context": true,
  "unanswered_aspects": []
}
```

Gemini supports `response_mime_type: "application/json"` plus a `response_schema`, which constrains decoding so the output is valid JSON — no regex-scraping prose.

**Why this is architecturally important, not just tidy:** it makes the verification layer *programmatic*. Citations arrive as structured data you can check with a set-membership test, rather than as text you have to parse out with a fragile regex. `sufficient_context: false` becomes a routable signal. **Structured output is what turns "the model said it cited something" into "the system verified the citation."**

---

# PART 8 — Verification: the near-zero layer

This is what separates your submission from every other candidate's LangChain tutorial.

## 8.1 Citation validity — deterministic, free, catches real bugs

```python
retrieved_ids = {c.chunk_id for c in context_chunks}
for cite in response.citations:
    if cite.chunk_id not in retrieved_ids:
        → invalid                     # the model invented a reference
```

No LLM, no embedding, no cost — a set membership test. It catches the specific failure where a model produces a *plausible-looking* clause number that was never in the context. Also verify the `(spec_id, clause_id)` pair actually exists in your chunk store, catching format-correct but nonexistent clauses like `§5.3.5.99`.

On failure: regenerate once with an explicit correction message; if it fails again, downgrade to "partially grounded" and strip the offending sentence.

## 8.2 Groundedness via NLI

**Natural Language Inference** = given a *premise* and a *hypothesis*, classify their relationship:

| Label | Meaning |
|---|---|
| **Entailment** | Premise supports the hypothesis |
| **Contradiction** | Premise refutes it |
| **Neutral** | Premise neither supports nor refutes |

Map it onto RAG: **premise = retrieved context, hypothesis = a sentence from the answer.** Anything not `entailment` is ungrounded.

**Step 1 — claim decomposition.** Split the answer into atomic claims. Sentence splitting is the cheap version; better is one LLM call asking for a list of independent factual claims. Why atomic? A sentence like *"The UE shall initiate re-establishment after T304 expiry, and T304 defaults to 100ms"* contains **two** claims — the first grounded, the second possibly invented. Score the sentence as a whole and the hallucination hides behind the true half.

**Step 2 — entailment check.** For each claim, against the context:

- **Primary: a local NLI cross-encoder.** Same architecture as Part 4.3 — premise and hypothesis go in together. Fast, free, deterministic, runs on your GPU. Look at `cross-encoder/nli-deberta-v3-base` or a purpose-built hallucination-detection model like Vectara's HHEM.
- **Secondary: Flash-Lite as a strict judge**, returning `SUPPORTED | PARTIAL | UNSUPPORTED` per claim with a required quoted span from the context. Demanding the quote is what stops the judge from being lazy.

**Why two:** NLI models are trained on short, general-domain sentence pairs and get shaky on long technical premises. The LLM judge handles nuance but is slower, costs money, and has known biases (self-preference, verbosity bias, position bias). Use NLI as the fast path and the LLM to break ties.

## 8.3 The abstention policy — selective prediction

```
groundedness = supported_claims / total_claims
```

| Score | Action | Badge |
|---|---|---|
| ≥ 0.9 | Return as-is | **Grounded** |
| 0.6 – 0.9 | Strip unsupported sentences, return the rest | **Partially grounded** |
| < 0.6 | Refuse, show closest passages | **Refused** |

The formal framing is **selective prediction**: a model that may output `⊥` (abstain). You trade **coverage** (fraction of questions answered) for **precision** (correctness of answers given). The right operating point depends on the cost of being wrong — and in a telecom standards context, a confident wrong answer costs an engineer hours of debugging, so **precision dominates**. Say exactly that.

Plot the **risk–coverage curve** on your eval set: sweep the threshold, plot error rate against coverage. That single chart in your README is worth a lot.

## 8.4 Auditability as a control

Every answer ships with expandable source snippets showing the exact retrieved text.

This is not just UI polish. Two real effects:

1. **Verification cost drops to near zero for the user.** A citation the user must go look up is a citation the user won't check. A citation with the text one click away gets checked.
2. **It changes what the system can get away with.** A pipeline whose evidence is always visible cannot quietly bluff — and demoing that live, including a refusal, is the most convincing thing you can show an evaluator.

---

# PART 9 — Serving

## 9.1 FastAPI

Built on **Starlette** (ASGI web framework) + **Pydantic** (validation).

- **ASGI vs WSGI:** WSGI (Flask, Django-classic) is synchronous — one request blocks a worker. ASGI is async — while one request waits on the Gemini API, the worker serves others. RAG is I/O-bound (API calls, DB queries), so this matters a lot for concurrency.
- **Pydantic models** define request/response schemas, validate automatically, and **auto-generate OpenAPI docs at `/docs`** — a live, interactive API page you can show an evaluator. Free credibility.
- **`async def` discipline:** use it for I/O (HTTP calls, Qdrant). But your embedding and reranker calls are **CPU/GPU-bound and synchronous** — running them directly in an `async def` blocks the whole event loop. Wrap them with `run_in_threadpool` / `asyncio.to_thread`. This is a real, common bug and a great detail to mention.

Endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Full pipeline, SSE streaming |
| `POST /search` | **Retrieval only** — no generation. Your debugging tool, and a great demo of the internals |
| `GET /health` | Index loaded? Models loaded? Qdrant reachable? |
| `POST /ingest` | Trigger reindex |

**Server-Sent Events (SSE)** for streaming: a long-lived HTTP response where the server pushes `data: ...\n\n` frames. Simpler than WebSockets and one-directional, which is all a chatbot response needs. It cuts *perceived* latency enormously — the user sees tokens at 1s instead of a blank screen for 6s.

**Load models once at startup** (FastAPI lifespan handler), never per request. bge-m3 + reranker take several seconds to load onto the GPU.

## 9.2 Streamlit

The one thing you must understand: **Streamlit re-runs your entire script top-to-bottom on every interaction.** There is no callback/event model by default.

Consequences:

- **`st.session_state`** is the only thing that survives a rerun. Chat history lives there.
- **`@st.cache_resource`** for models/connections (returns the same object), **`@st.cache_data`** for computed data (returns a copy). Without these you reload bge-m3 on every keystroke.
- `st.chat_message` / `st.chat_input` give you a proper chat UI in ~10 lines.
- `st.write_stream` consumes a generator for token-by-token streaming.

**The transparency panel** — build this, it's your demo centrepiece. A sidebar or expander showing, for the current answer:

- each retrieved chunk with its **dense score, BM25 rank, RRF score, and rerank score**
- the citation header of each
- the **groundedness badge** and per-claim verdicts
- whether the relevance gate fired, and the max rerank score vs τ

Most candidates demo a chat box. You demo the machine's reasoning.

---

# PART 10 — Evaluation

"It seems to work" is not a result. This section is what gets you graded well.

## 10.1 Retrieval metrics

**Recall@k** — of the questions, what fraction had the gold clause somewhere in the top-k?

```
Recall@k = (# questions where gold clause ∈ top-k) / (total questions)
```

Measure at k=5 and k=10. **Recall@30 on the pre-rerank candidate set is your ceiling** — if the gold passage isn't in the 30 candidates, no amount of reranking can save you. Track it; it tells you whether to fix retrieval or ranking.

**MRR — Mean Reciprocal Rank**

```
MRR = (1/|Q|) · Σ 1/rankᵢ        rankᵢ = position of the first relevant result
```

Gold at rank 1 → 1.0; rank 2 → 0.5; rank 5 → 0.2. Rewards putting the right thing **first**, which is what reranking is for.

**nDCG@k — Normalised Discounted Cumulative Gain**

```
DCG@k  = Σ  relᵢ / log₂(i+1)
        i=1..k
nDCG@k = DCG@k / IDCG@k        (IDCG = DCG of the ideal ordering)
```

Handles **graded** relevance (a passage can be perfect/partial/irrelevant, not just binary) and discounts by position. The standard IR metric — use it if you have graded labels; MRR is fine if labels are binary.

## 10.2 Answer metrics

| Metric | Question it answers | How |
|---|---|---|
| **Faithfulness / groundedness** | Is every claim supported by the retrieved context? | Claims entailed ÷ total claims |
| **Answer relevance** | Does it actually address the question? | Embedding sim between question and an LLM-generated question reverse-engineered from the answer |
| **Context precision** | Are the retrieved chunks actually useful, and ranked well? | Fraction of retrieved chunks that are relevant, position-weighted |
| **Context recall** | Did retrieval get everything needed? | Fraction of ground-truth answer claims covered by retrieved context |
| **Citation precision** | Are the citations correct? | Valid citations ÷ total citations |

**RAGAS** implements most of these. Use it, but **understand what it computes** — several of its metrics are LLM-judge based, so they carry judge noise and cost. Run them at temperature 0 and be honest about variance.

**Note the crucial distinction — faithfulness ≠ correctness.** An answer can be perfectly faithful to a retrieved passage that was the wrong passage. That is why you measure retrieval *and* generation separately.

## 10.3 The headline metrics

**Hallucination rate** — define it precisely and state your definition:

```
hallucination_rate = (# unsupported claims) / (# total claims across all answers)
```

**Correct-refusal rate** — on the 15 adversarial questions:

```
correct_refusal_rate = (# correctly refused) / (# should-be-refused)
```

And its counterpart, **over-refusal rate** on the answerable set — refusing everything gives a perfect hallucination rate and a useless product. **Always report both.** Reporting the metric that makes you look bad, alongside the one that makes you look good, is the single clearest signal of engineering honesty in a submission.

## 10.4 The gold set — 70 questions

| Type | N | Purpose |
|---|---|---|
| Answerable, single-clause | 40 | Core retrieval + generation quality |
| Adversarial / out-of-scope | 15 | Refusal behaviour |
| Multi-hop, cross-spec | 15 | Harder reasoning, cross-document citation |

**Construction:** sample clauses → have an LLM draft a question whose answer is *in that clause* → **you verify each one by hand**. The clause id becomes ground truth. Hand-verification is non-negotiable; an unverified LLM-generated gold set measures nothing.

**Adversarial categories to include:**
- Non-existent specs — *"What does TS 99.999 specify?"*
- Out-of-domain — *"What does a gNB cost?"*
- Future/unknown — *"Summarise the Rel-25 roadmap"*
- Subjective — *"Is 5G better than 4G?"*
- **Plausible-but-false premises** — *"Explain the 'RRCUltraReconfiguration' procedure."* ← the hardest and most revealing case, because a weak system will cheerfully invent it.

## 10.5 The ablation table

| Configuration | Recall@5 | MRR | Faithfulness | Hallucination rate | Correct refusal | p95 latency |
|---|---|---|---|---|---|---|
| A. Dense only, no rerank | | | | | | |
| B. + hybrid (BM25 + RRF) | | | | | | |
| C. + cross-encoder rerank | | | | | | |
| D. + relevance gate | | | | | | |
| E. + verification layer (**final**) | | | | | | |

**Methodology rules:** change **one variable per row**, same gold set, same seed, temperature 0 everywhere. Report latency alongside quality — every row buys accuracy with time, and pretending otherwise is dishonest.

**What each row proves:**

- A→B: hybrid fixes exact-identifier queries → Recall jumps
- B→C: reranking fixes ordering → MRR jumps more than Recall
- C→D: the gate fixes out-of-scope → correct-refusal jumps, hallucination drops
- D→E: verification catches residual unsupported claims → hallucination approaches zero

**This table is the deliverable.** Five rows, each one a claim you can defend for ten minutes. Add a per-row note of *which specific question* flipped from wrong to right — a concrete example beats a number every time.

---

# PART 11 — Interview drill

Rehearse these out loud. Not reading — saying.

**1. Why is hallucination low in your system?**
Four independent controls, and name them as a stack: version pinning (no cross-release contradiction), the relevance gate (no context → no LLM call → nothing to invent), forced citations validated by deterministic set membership, and NLI groundedness with an abstention policy. Then quote your measured hallucination rate and correct-refusal rate. *Numbers, not adjectives.*

**2. Why hybrid retrieval?**
Give both concrete failure cases: `t304` (dense fails, BM25 saves) and "how does the UE recover from reconfiguration failure" (BM25 fails on vocabulary mismatch, dense saves). Then RRF, and why ranks beat scores.

**3. Bi-encoder vs cross-encoder?**
Independent encoding + precomputation + fast but lossy, versus joint encoding + full cross-attention + accurate but O(n). Hence the two-stage funnel: recall first, precision second.

**4. Why chunk by clause?**
3GPP requirements are clause-scoped. Splitting mid-clause severs a `shall` from its triggering condition, producing a retrieved requirement that reads as unconditional. That is how a RAG system tells an engineer something is always mandatory when it is only mandatory in one state.

**5. How did you pick τ?**
Empirically. Two score distributions on the eval set — answerable versus out-of-scope — pick the separation point, and describe the precision/coverage trade-off in both directions.

**6. What are the limitations?** *(Name them before the interviewer does — this reads as confidence, not weakness.)*
Cross-release comparison questions; figures and diagrams (text-only pipeline, captions only); deep ASN.1 structural reasoning; the corpus is 14 specs not the full 3GPP set; and the LLM-judge components carry judge bias.

**7. How would you scale it?**
Full corpus with per-release collections and metadata filtering; incremental reindex on new spec versions (hash-based change detection so you re-embed only changed clauses); semantic caching of frequent queries; a thumbs-down feedback loop feeding a hard-negatives set to fine-tune the retriever.

**8. What would you do with two more weeks?**
Fine-tune the embedding model on 3GPP query-passage pairs; add a query-decomposition agent for multi-hop questions; build a proper cross-reference graph (specs cite each other constantly) and traverse it during retrieval; expand the gold set to 500 questions with multiple annotators and report inter-annotator agreement.

---

# Reading order for your 3 days

You do not have time to read this linearly and also build. Do this:

**Tonight (Aug 14, before coding):** Part 0 entire, Part 1 entire, Part 3.1–3.3. That's the conceptual spine — roughly 45 minutes. Everything else you read *at the moment you implement it*.

**Then, just-in-time as you build:**

| When you're writing... | Read first |
|---|---|
| `parse_docx.py` | Part 2 |
| `chunk.py` | Part 3 (all) |
| `embedder.py`, `vector_store.py` | Part 4 |
| `bm25_store.py` | Part 5 |
| `hybrid.py`, `reranker.py`, `gate.py` | Part 6 |
| `generator.py`, `prompts.py` | Part 7 |
| `verification/` | Part 8 |
| `api/`, `ui/` | Part 9 |
| `eval/` | Part 10 |

**Sat night, before submitting:** Part 11 out loud, twice.
