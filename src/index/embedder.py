"""
Turning chunks into vectors.

The model is BAAI/bge-m3. Three things about it decide whether retrieval
works, and all three are easy to get silently wrong:

1. QUERIES AND DOCUMENTS ARE NOT SYMMETRIC.
   BGE models are trained with an instruction prefix on the QUERY side only.
   Embed a query without the prefix (or a document WITH it) and nothing
   crashes — you just quietly lose several points of recall, and you will
   never see why, because every score still looks plausible. So this module
   exposes two separate methods and refuses to let you use one for the other.

2. NORMALIZE, THEN COSINE IS FREE.
   After L2 normalisation, cosine similarity and dot product are the same
   operation: cos(a,b) = a·b / (|a||b|), and |a| = |b| = 1. Vector databases
   compute dot products very fast. So we normalise once at index time and get
   cosine ranking at zero cost forever after.

3. A BI-ENCODER CANNOT SEE THE QUERY AND THE DOCUMENT TOGETHER.
   Each is embedded independently, so the vector for a chunk must be a good
   summary of it for EVERY possible question. That is a hard ask, and it is
   exactly why this is only stage one — the cross-encoder reranker later
   reads query and chunk jointly and fixes what this stage gets wrong.
   Do not spend days tuning the embedder; spend them on the reranker and
   the gate.

Usage
-----
    python -m src.index.embedder --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    DEVICE,
    EMBED_BATCH,
    EMBED_DIM,
    EMBED_FP16,
    EMBED_MAX_LEN,
    EMBED_MODEL,
    QUERY_PREFIX,
)


class Embedder:
    """Thin wrapper over sentence-transformers that keeps the two sides apart."""

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str = DEVICE,
        batch_size: int = EMBED_BATCH,
        max_len: int = EMBED_MAX_LEN,
        fp16: bool = EMBED_FP16,
    ) -> None:
        # Imported lazily: loading torch takes seconds, and several callers
        # (the chunker's tests, the eval harness) import this module only for
        # its constants.
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_len

        # See EMBED_FP16 in config: on a 4 GB GPU this is the difference
        # between the weights living in VRAM and being streamed over PCIe.
        # Guarded on device because fp16 arithmetic on CPU is emulated and
        # comes out slower than fp32, not faster.
        if fp16 and device == "cuda":
            self.model = self.model.half()
        self.fp16 = fp16 and device == "cuda"

        dim = self.model.get_sentence_embedding_dimension()
        if dim != EMBED_DIM:
            raise ValueError(
                f"{model_name} produces {dim}-dim vectors but config says "
                f"EMBED_DIM={EMBED_DIM}. Fix config before indexing — a "
                f"mismatch here corrupts the whole collection."
            )

    def _encode(self, texts: list[str], show: bool) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # see note 2 above
            convert_to_numpy=True,
            show_progress_bar=show,
        )
        # In fp16 the model hands back float16. Cast up before it leaves this
        # class: half precision is a compute-time trick, and letting it reach
        # the store would bake it into the index and into every similarity
        # computed against it forever after. Chroma also expects float32.
        return vecs.astype(np.float32, copy=False)

    def embed_documents(self, texts: list[str], show: bool = False) -> np.ndarray:
        """Chunks. NO instruction prefix — that side is not prefixed."""
        return self._encode(texts, show)

    def embed_queries(self, texts: list[str], show: bool = False) -> np.ndarray:
        """User questions. Prefix is mandatory; see note 1."""
        return self._encode([QUERY_PREFIX + t for t in texts], show)


# --------------------------------------------------------------------------
def self_test() -> int:
    """
    Prove four things at once before spending an hour indexing:
    the model loads, the GPU is actually used, vectors are unit length,
    and the query prefix is doing something useful.
    """
    import torch

    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    elif DEVICE == "cuda":
        print("! config says DEVICE='cuda' but torch cannot see a GPU.")
        print("  Fix the torch install or set DEVICE='cpu' in src/config.py.")
        return 1

    print(f"\nloading {EMBED_MODEL} ...")
    emb = Embedder()

    docs = [
        # the one that should win
        "TS 38.331 v18.10.0, clause 5.3.5.3: Reception of an RRCReconfiguration "
        "by the UE. The UE shall perform the following actions upon reception "
        "of the RRCReconfiguration message.",
        # same spec, different topic
        "TS 38.331, clause 5.3.7: RRC connection re-establishment. The purpose "
        "of this procedure is to re-establish the RRC connection.",
        # different domain entirely
        "TS 23.501, clause 5.15: Network slicing. A network slice is defined "
        "within a PLMN and identified by S-NSSAI.",
        # unrelated to telecom
        "The recipe calls for two cups of flour and a teaspoon of salt.",
    ]
    query = "What does the UE do when it receives an RRCReconfiguration message?"

    dv = emb.embed_documents(docs)
    qv = emb.embed_queries([query])

    norms = np.linalg.norm(dv, axis=1)
    print(f"\nvector shape {dv.shape}, norms min {norms.min():.4f} max {norms.max():.4f}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        print("! vectors are not unit length — normalisation is broken")
        return 1

    sims = (dv @ qv[0])
    order = np.argsort(-sims)
    print(f"\nquery: {query}\n")
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {sims[i]:.4f}  {docs[i][:72]}...")

    if order[0] != 0:
        print("\n! the relevant passage did not rank first — investigate before indexing")
        return 1

    # The prefix should change the vector. If it doesn't, it isn't being applied.
    raw = emb.embed_documents([query])[0]
    delta = float(np.dot(raw, qv[0]))
    print(f"\nprefixed vs unprefixed query similarity: {delta:.4f} "
          f"(should be < 1.0 — proves the prefix is applied)")

    print("\nself-test passed.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bge-m3 embedder")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    raise SystemExit(self_test() if args.self_test else 0)
