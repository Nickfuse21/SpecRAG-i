"""
The sparse half of retrieval: BM25 over the same chunks.

Why bother when we have embeddings
----------------------------------
Dense retrieval is good at meaning and bad at identifiers. Ask for
"gNB-DU" and bge-m3 happily returns passages about gNB-CU, because in
embedding space they are nearly the same thing — which is exactly the
behaviour you want for "handover" and "mobility", and exactly the behaviour
that ruins a telecom answer. 3GPP is full of tokens where one character
changes the meaning completely: 38.331 vs 38.321, SRB1 vs SRB3, T310 vs
T311, RRCSetup vs RRCResume.

BM25 has the opposite bias. It cannot generalise at all, which means it
never blurs an identifier. The two failure modes barely overlap, and that
is the whole argument for hybrid retrieval: fuse them and you cover both.

The formula
-----------
For query term q in document d:

                                    tf(q,d) · (k1 + 1)
    score = IDF(q) · ----------------------------------------------
                     tf(q,d) + k1 · (1 - b + b · |d| / avgdl)

Two knobs, and each does one job:

  k1 (=1.5) — term-frequency saturation. Without it, a chunk containing
    "RRCReconfiguration" 40 times would score ~40× one containing it once.
    The 40th mention tells you almost nothing the 2nd didn't. k1 makes the
    curve flatten.

  b (=0.75) — length normalisation. Long chunks contain more of everything,
    so they'd win every query on volume alone. b scales the penalty by how
    far above average the chunk's length is. b=0 disables it, b=1 fully
    normalises; 0.75 is the standard compromise.

  IDF — a term appearing in nearly every chunk ("the", "UE", "shall")
    carries almost no discriminating power, so its weight collapses.

Implemented directly rather than pulled from a library: it is forty lines,
it makes the tokenizer decisions below explicit, and I need to be able to
explain every number in it.

Usage
-----
    python -m src.index.bm25_store --build
    python -m src.index.bm25_store --query "gNB-DU F1 setup procedure"
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CHUNKS_DIR, INDEX_DIR  # noqa: E402

BM25_PATH = INDEX_DIR / "bm25.pkl"

K1 = 1.5
B = 0.75

# A token is an alphanumeric run that may contain internal . - / _
# so these survive intact:
#     38.331      dotted spec number
#     gNB-DU      hyphenated node name
#     SS/PBCH     slashed pair
#     5G-AN       digit-led identifier
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/]*[A-Za-z0-9]|[A-Za-z0-9]")

# camelCase / PascalCase boundary, for RRCReconfiguration -> RRC + Reconfiguration
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Deliberately tiny. The usual English stopword lists delete `shall`, `should`,
# `may` and `must` — the four words TR 21.801 gives legal meaning to. Removing
# them from a 3GPP index would be like removing the verbs.
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are",
    "be", "been", "as", "at", "by", "on", "it", "its", "this", "that",
    "with", "from", "which", "if", "then", "else", "was", "were",
}


def tokenize(text: str) -> list[str]:
    """
    One surface form goes in, several searchable forms come out.

    `RRCReconfiguration` is emitted as itself AND as `rrc` + `reconfiguration`,
    so a user typing "RRC reconfiguration" as two words still matches, while
    someone pasting the exact identifier gets the exact-match boost. Same for
    `gNB-DU` -> `gnb-du` + `gnb` + `du`.

    The whole token is always kept, so precision on identifiers is never
    traded away for this recall.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        if low not in _STOP:
            out.append(low)

        extra: list[str] = []

        # separator split: gNB-DU -> gNB, DU ; SS/PBCH -> SS, PBCH
        parts = [p for p in re.split(r"[-_/]", raw) if len(p) >= 2]
        if len(parts) > 1:
            extra.extend(parts)

        # camel split, min 3 chars — this keeps RRC and Reconfiguration but
        # drops the "g" and "NB" that gNB would otherwise shed as noise.
        for p in parts or [raw]:
            pieces = [x for x in _CAMEL.split(p) if len(x) >= 3]
            if len(pieces) > 1:
                extra.extend(pieces)

        for p in extra:
            pl = p.lower()
            if pl != low and pl not in _STOP:
                out.append(pl)
    return out


class BM25Store:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        # term -> list of (doc index, term frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

    # -- build ------------------------------------------------------------
    def build(self, chunks: list[dict]) -> None:
        self.ids = [c["chunk_id"] for c in chunks]
        self.doc_len = []
        self.postings = defaultdict(list)

        for i, c in enumerate(chunks):
            # Index embed_text, not text: the contextual header carries the
            # spec number and clause id, which are exactly the tokens a
            # keyword search for "38.331 clause 5.3.5.3" needs to hit.
            toks = tokenize(c["embed_text"])
            self.doc_len.append(len(toks))
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, n in tf.items():
                self.postings[t].append((i, n))

        n_docs = len(chunks)
        self.avgdl = sum(self.doc_len) / max(n_docs, 1)
        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative
        # for terms appearing in more than half the corpus.
        self.idf = {
            t: math.log(1 + (n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    # -- query ------------------------------------------------------------
    def search(self, query: str, k: int = 30) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc, tf in postings:
                dl = self.doc_len[doc]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[doc] += idf * (tf * (K1 + 1)) / denom

        top = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.ids[d], s) for d, s in top]

    # -- persistence ------------------------------------------------------
    def save(self, path: Path = BM25_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "doc_len": self.doc_len,
                    "avgdl": self.avgdl,
                    "postings": dict(self.postings),
                    "idf": self.idf,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Store":
        if not path.exists():
            raise SystemExit(f"No BM25 index at {path}. Run: python -m src.index.bm25_store --build")
        with path.open("rb") as f:
            d = pickle.load(f)
        s = cls()
        s.ids, s.doc_len, s.avgdl = d["ids"], d["doc_len"], d["avgdl"]
        s.postings = defaultdict(list, d["postings"])
        s.idf = d["idf"]
        return s


def load_chunks() -> list[dict]:
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No chunks in {CHUNKS_DIR}. Run: python -m src.ingest.chunk")
    return [json.loads(line) for f in files for line in f.open(encoding="utf-8")]


def main() -> int:
    ap = argparse.ArgumentParser(description="BM25 keyword index")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query", default=None)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    if args.build:
        chunks = load_chunks()
        store = BM25Store()
        store.build(chunks)
        store.save()
        size = BM25_PATH.stat().st_size / 1e6
        print(f"indexed {len(chunks):,} chunks, {len(store.postings):,} unique terms")
        print(f"avg doc length {store.avgdl:.0f} tokens -> {BM25_PATH} ({size:.1f} MB)")

        # Show that the tokenizer does what the docstring claims.
        print("\ntokenizer check:")
        for probe in ("RRCReconfiguration", "gNB-DU", "38.331", "SS/PBCH"):
            print(f"  {probe:<22} -> {tokenize(probe)}")

    if args.query:
        store = BM25Store.load()
        by_id = {c["chunk_id"]: c for c in load_chunks()}
        print(f"\nquery: {args.query}\n")
        for cid, score in store.search(args.query, k=args.k):
            c = by_id.get(cid, {})
            print(f"  {score:7.3f}  {c.get('citation','?')} — {c.get('clause_title','')[:56]}")

    if not args.build and not args.query:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
