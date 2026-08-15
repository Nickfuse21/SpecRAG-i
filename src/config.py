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
EMBED_BATCH = 8           # lower this if you hit CUDA OOM
EMBED_MAX_LEN = 2048      # bge-m3 supports 8192; our chunks are far below

# BGE models were trained with an instruction prefix on QUERIES ONLY.
# Getting this wrong silently costs several points of recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_BATCH = 8

DEVICE: Literal["cuda", "cpu"] = "cuda"   # set to "cpu" if no GPU available


# --------------------------------------------------------------------------
# Retrieval funnel  —  tune these on the eval set, do not trust the defaults
# --------------------------------------------------------------------------
DENSE_TOP_K = 30    # wide net: gold passage must be in here (this is the ceiling)
BM25_TOP_K = 30
RRF_K = 60          # damps the top of each list so cross-retriever agreement wins
FUSED_TOP_K = 20    # reranking budget
FINAL_TOP_K = 6     # context budget + "lost in the middle"

# CONTROL #2 — the relevance gate. bge-reranker-v2-m3 emits raw logits,
# roughly -10..+10. Below this, we REFUSE WITHOUT CALLING THE LLM.
# Calibrate on Day 3: plot max-score distributions for answerable vs
# out-of-scope questions and cut where they separate.
RERANK_SCORE_THRESHOLD = 0.0   # PLACEHOLDER — must be calibrated


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
GEN_MODEL = "gemini-2.5-flash"
UTIL_MODEL = "gemini-2.5-flash-lite"   # query rewriting + verification judge

GEN_TEMPERATURE = 0.0   # not optional: sampling a lower-probability token in a
                        # factual answer means sampling a less likely FACT.
                        # Also required for reproducible eval numbers.
GEN_MAX_OUTPUT_TOKENS = 2048


# --------------------------------------------------------------------------
# Verification  —  CONTROL #4
# --------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

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
