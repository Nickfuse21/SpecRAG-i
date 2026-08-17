"""
Step 5: embed every chunk and write it into the vector store.

This is the longest-running script in the project — it is the only place the
GPU does real work for minutes at a time. Two habits make that survivable:

  --limit N   index a small slice first. Proving the plumbing on 200 chunks
              takes 20 seconds; discovering a metadata bug after 40 minutes
              of embedding takes 40 minutes.

  --reset     always rebuild from empty when the chunker output changed.
              Adding into a populated collection leaves the old vectors in
              place, and stale chunks that no longer exist in data/chunks
              will keep being retrieved and cited. That failure looks exactly
              like a hallucination and is very hard to trace.

Usage
-----
    python -m src.index.build --limit 200 --reset     # smoke test
    python -m src.index.build --reset                 # the real run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import CHUNKS_DIR  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.vector_store import VectorStore  # noqa: E402

# Chroma rejects very large single add() calls; this also keeps peak memory
# flat instead of holding every vector in RAM at once.
WRITE_BATCH = 512


def load_chunks(limit: int | None) -> list[dict]:
    files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No chunks in {CHUNKS_DIR}. Run: python -m src.ingest.chunk")
    chunks: list[dict] = []
    for f in files:
        for line in f.open(encoding="utf-8"):
            chunks.append(json.loads(line))
            if limit and len(chunks) >= limit:
                return chunks
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the dense vector index")
    ap.add_argument("--limit", type=int, default=None, help="index only the first N chunks")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    chunks = load_chunks(args.limit)
    print(f"{len(chunks):,} chunks to index")

    store = VectorStore()
    if args.reset:
        store.reset()
        print("collection reset")
    elif store.count():
        print(f"! collection already holds {store.count():,} vectors.")
        print("  Use --reset unless you are deliberately appending.")
        return 1

    emb = Embedder()
    t0 = time.time()

    for i in range(0, len(chunks), WRITE_BATCH):
        batch = chunks[i:i + WRITE_BATCH]
        # We embed `embed_text` (header + body) but Chroma stores `text`.
        # The header helps the retriever find it; the model downstream reads
        # the clean body plus structured metadata.
        vectors = emb.embed_documents([c["embed_text"] for c in batch], show=False)
        store.add(batch, vectors)

        done = min(i + WRITE_BATCH, len(chunks))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (len(chunks) - done) / max(rate, 1e-9)
        print(f"  {done:>6,}/{len(chunks):,}  {rate:5.1f} chunks/s  eta {eta/60:4.1f} min")

    print(f"\ndone in {(time.time() - t0)/60:.1f} min — collection holds {store.count():,} vectors")

    # A real query through the real path, so a broken index fails here rather
    # than three modules downstream.
    q = "What does the UE do when it receives an RRCReconfiguration message?"
    hits = store.search(emb.embed_queries([q])[0], k=5)
    print(f"\nsanity query: {q}")
    for h in hits:
        print(f"  {h['score']:.4f}  {h['citation']} — {h['clause_title'][:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
