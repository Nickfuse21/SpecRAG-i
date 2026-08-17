"""
Central configuration. EVERY tunable number in the system lives here.

Why one file: on Day 3 you will tune thresholds against the eval set and run
ablations. If constants are scattered across ten modules, ablations become
error-prone guesswork. One file = one place to change = reproducible runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW_DIR = DATA / "raw"        # downloaded .zip files
DOCX_DIR = DATA / "docx"      # extracted / converted .docx
PARSED_DIR = DATA / "parsed"  # clause-tree JSONL, one file per spec
CHUNKS_DIR = DATA / "chunks"  # final chunk JSONL
INDEX_DIR = DATA / "index"    # chroma + bm25 persistence
EVAL_DIR = ROOT / "eval"

for _d in (RAW_DIR, DOCX_DIR, PARSED_DIR, CHUNKS_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Corpus  —  CONTROL #1: version pinning
# --------------------------------------------------------------------------
# Mixing releases is a correctness bug, not a nicety: Rel-15 and Rel-18 text
# can flatly contradict each other, retrieval will happily return both, and
# the LLM will silently blend them into one confident wrong answer.
#
# In 3GPP filenames the version is 3 chars, each a base-36-ish digit
# (0-9 then a=10, b=11 ... z=35). The FIRST char is the major version, which
# tracks the Release:  f=15  g=16  h=17  i=18  j=19
TARGET_RELEASE = 18            # Rel-18 ("5G-Advanced")

# (spec number, human title) — spec number drives the FTP path.
SPECS: list[tuple[str, str]] = [
    # --- RAN / air interface ---
    ("38.300", "NR and NG-RAN Overall Description"),
    ("38.331", "NR RRC Protocol Specification"),
    ("38.321", "NR MAC Protocol Specification"),
    ("38.322", "NR RLC Protocol Specification"),
    ("38.323", "NR PDCP Protocol Specification"),
    ("38.401", "NG-RAN Architecture Description"),
    ("38.211", "NR Physical Channels and Modulation"),
    ("38.212", "NR Multiplexing and Channel Coding"),
    ("38.213", "NR Physical Layer Procedures for Control"),
    ("38.214", "NR Physical Layer Procedures for Data"),
    # --- Core network / system ---
    ("23.501", "System Architecture for the 5G System"),
    ("23.502", "Procedures for the 5G System"),
    ("24.501", "NAS Protocol for 5G System"),
    ("33.501", "Security Architecture and Procedures for 5G System"),
    # --- Vocabulary (used to auto-build the acronym glossary) ---
    ("21.905", "Vocabulary for 3GPP Specifications"),
]

# If Day 1 runs long, cut to these six and move on. A small corpus with a
# complete pipeline and real eval numbers beats a large corpus with half a
# system — see the notes, page 73.
SPECS_MINIMAL = ["38.300", "38.331", "23.501", "23.502", "24.501", "21.905"]

FTP_BASE = "https://www.3gpp.org/ftp/Specs/archive"


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# Rule 1 (non-negotiable): one chunk belongs to exactly one clause.
# A `shall` and its triggering condition live in the same clause; split them
# and you retrieve a requirement that reads as unconditional.
CHUNK_MAX_TOKENS = 1000    # above this, split at paragraph boundaries
CHUNK_TARGET_TOKENS = 800  # target size of each split piece
CHUNK_OVERLAP_TOKENS = 120 # ~15% — keeps a fact near a boundary whole somewhere
CHUNK_MIN_TOKENS = 80      # below this, merge upward into the parent clause

TOKENIZER_ENCODING = "cl100k_base"  # tiktoken; only used for length budgeting

# Boilerplate clause titles to drop at parse time. These are near-identical
# across every spec, so they score high similarity against *everything* and
# pollute top-k.
DROP_CLAUSE_TITLES = {
    "foreword", "scope", "references", "definitions of terms, symbols and abbreviations",
    "change history", "introduction",
}


# --------------------------------------------------------------------------
# Embedding / reranking models
# --------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_BATCH = 16          # lower this if you hit CUDA OOM
EMBED_MAX_LEN = 2048      # bge-m3 supports 8192; our chunks are far below

# Half precision. On this box it is not an optimisation, it is what makes the
# run finish: bge-m3 is ~2.27 GB in fp32, and a 4 GB laptop GPU with a desktop
# session already on it has ~2 GB free. The weights do not fit, and Windows
# WDDM does not fail — it silently spills them to system RAM and streams them
# back over PCIe on every forward pass. That is a correct-but-useless 1 chunk/s,
# i.e. ~2.7 hours for this corpus. In fp16 the weights are ~1.14 GB, they stay
# resident, and the same run takes minutes.
#
# The accuracy cost is nil for our purposes: we only ever compare these vectors
# by cosine similarity and normalise them straight afterwards, so fp16 rounding
# lands far below the gap between a relevant and an irrelevant passage.
# `build.py --verify-precision` measures that gap instead of assuming it.
EMBED_FP16 = True         # set False on CPU — fp16 on CPU is slower, not faster

# BGE models were trained with an instruction prefix on QUERIES ONLY.
# Getting this wrong silently costs several points of recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_BATCH = 8
# Same reasoning as EMBED_FP16, same model family and size. This one matters
# for latency rather than for the build: the reranker runs on every live query,
# so it sits directly in the user's response time.
RERANK_FP16 = True

DEVICE: Literal["cuda", "cpu"] = "cuda"   # set to "cpu" if no GPU available


# --------------------------------------------------------------------------
# Retrieval funnel  —  tune these on the eval set, do not trust the defaults
# --------------------------------------------------------------------------
DENSE_TOP_K = 30    # wide net: gold passage must be in here (this is the ceiling)
BM25_TOP_K = 30
RRF_K = 60          # damps the top of each list so cross-retriever agreement wins
FUSED_TOP_K = 20    # reranking budget
FINAL_TOP_K = 6     # context budget + "lost in the middle"

# CONTROL #2 — the relevance gate. Below this, we REFUSE WITHOUT CALLING THE
# LLM. bge-reranker-v2-m3 scores are SIGMOID outputs in [0, 1] (see the note in
# src/retrieval/rerank.py) — not the signed logits this line originally assumed.
# The old placeholder value of 0.0 sat below the entire range, so `best < 0.0`
# was never true and the gate never fired once.
#
# Fitted by eval/calibrate.py, not chosen: it maximises the correct-refusal
# rate on out-of-scope questions subject to a 10% false-refusal budget on
# answerable ones. Re-run that script after any change to the retriever, the
# reranker, or EMBED/RERANK_FP16 — all of them move this number.
RERANK_SCORE_THRESHOLD = 0.90  # set by eval/calibrate.py — see eval/calibration.json


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
# Pinned version names, never the `gemini-flash-latest` aliases. GEN_TEMPERATURE
# is 0 so that eval numbers are reproducible, and an alias that silently
# re-points at a new model underneath would throw that away — the ablation
# table would stop being a comparison between our own arms and start being a
# comparison against whatever Google shipped that week.
#
# 2.5-flash was the original choice and now returns 404 for keys that had not
# already used it ("no longer available to new users"), which is worth knowing
# before a demo: these names expire. Check with `client.models.list()`.
GEN_MODEL = "gemini-3.5-flash"
UTIL_MODEL = "gemini-3.5-flash-lite"   # query rewriting + verification judge

# Transient-failure budget for the hosted model (see src/llm_retry.py). Four
# attempts with a 2s base backs off 2s, 4s, 8s — enough to ride out a 503
# "high demand" spike without stalling a live request for a visible age.
LLM_MAX_ATTEMPTS = 4
LLM_BACKOFF_BASE = 2.0

GEN_TEMPERATURE = 0.0   # not optional: sampling a lower-probability token in a
                        # factual answer means sampling a less likely FACT.
                        # Also required for reproducible eval numbers.
GEN_MAX_OUTPUT_TOKENS = 2048


# --------------------------------------------------------------------------
# Verification  —  CONTROL #4
# --------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
# Verification runs with the embedder and the reranker already resident, so
# this is the model that decides whether all three fit. In fp32 the three
# together reserve ~4.35 GB on a 4 GB card and the driver starts spilling to
# system RAM again; in fp16 they fit with headroom.
NLI_FP16 = True

GROUNDED_THRESHOLD = 0.90    # >= this  -> return as-is, badge "Grounded"
PARTIAL_THRESHOLD = 0.60     # >= this  -> strip unsupported claims, "Partial"
                             # <  this  -> REFUSE + show closest passages

REFUSAL_MESSAGE = (
    "I could not find this in the indexed 3GPP specifications. "
    "The closest passages I found are shown below so you can judge for yourself."
)


# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8000
CHROMA_COLLECTION = "tgpp_chunks"


# --------------------------------------------------------------------------
# Secrets (from .env)
# --------------------------------------------------------------------------
class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    GEMINI_API_KEY: str = ""
    SOFFICE_PATH: str = ""


secrets = Secrets()
