"""
The dense half of retrieval: a persisted ChromaDB collection.

Why Chroma and not Qdrant: Qdrant is the better engine, but on Windows it
wants Docker, and a broken Docker install on day 2 of a 3-day build is a
project-ending risk for zero marks gained. Chroma is `pip install` and
embeds in-process. The cost of that choice is contained by the interface
below — everything downstream talks to `VectorStore`, never to Chroma
directly, so swapping engines later is one file, not a refactor.

Two Chroma-specific things worth knowing:

  - Metadata values must be scalars (str / int / float / bool). Lists are
    rejected. So list-valued fields are joined into comma strings on the way
    in and split on the way out — the round-trip lives here so no caller has
    to remember it.

  - We set the space to cosine explicitly. Our vectors are already L2
    normalised, so l2 and cosine rank identically, but the returned distance
    is only interpretable as `1 - cosine` if we say so. Scores we cannot
    read are scores we cannot threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CHROMA_COLLECTION, INDEX_DIR  # noqa: E402

CHROMA_PATH = INDEX_DIR / "chroma"

# Fields that are lists in the chunk record and strings in Chroma.
_LIST_FIELDS = ("block_types", "merged_clause_ids")

# Everything we want to be able to filter or display on. `text` is stored as
# the document body, not as metadata, so it is not repeated here.
_META_FIELDS = (
    "spec", "spec_id", "spec_title", "version", "release",
    "clause_id", "clause_title", "breadcrumb", "level",
    "part", "n_parts", "has_normative", "citation",
    "token_count", "order",
)


def to_metadata(chunk: dict) -> dict[str, Any]:
    md: dict[str, Any] = {k: chunk[k] for k in _META_FIELDS}
    for f in _LIST_FIELDS:
        md[f] = ",".join(chunk.get(f) or [])
    return md


def from_metadata(md: dict[str, Any]) -> dict[str, Any]:
    out = dict(md)
    for f in _LIST_FIELDS:
        out[f] = [x for x in str(md.get(f, "")).split(",") if x]
    return out


class VectorStore:
    """The only thing in this project that knows Chroma exists."""

    def __init__(self, path: Path = CHROMA_PATH, collection: str = CHROMA_COLLECTION):
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.name = collection
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    # -- writing ----------------------------------------------------------
    def reset(self) -> None:
        """Drop and recreate. Re-indexing into a live collection silently
        mixes old and new vectors, which is a debugging nightmare."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.col.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c["text"] for c in chunks],
            metadatas=[to_metadata(c) for c in chunks],
        )

    # -- reading ----------------------------------------------------------
    def count(self) -> int:
        return self.col.count()

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 30,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Returns hits sorted best-first, each with a `score` in [0, 1] where
        higher is more similar (Chroma hands back a distance, so we flip it).
        """
        res = self.col.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, md, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hit = from_metadata(md)
            hit["chunk_id"] = cid
            hit["text"] = doc
            hit["score"] = 1.0 - float(dist)
            hits.append(hit)
        return hits

    def get(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch by id — used for small-to-big expansion and citation checks."""
        # Chroma raises on an empty id list rather than returning nothing.
        # "Fetch none" is a legitimate request from every caller here, and it
        # happens whenever the fused list adds no ids the dense search did not
        # already return — which is every query once BM25 is switched off.
        if not chunk_ids:
            return []
        res = self.col.get(ids=chunk_ids, include=["documents", "metadatas"])
        out = []
        for cid, doc, md in zip(res["ids"], res["documents"], res["metadatas"]):
            hit = from_metadata(md)
            hit["chunk_id"] = cid
            hit["text"] = doc
            out.append(hit)
        return out
