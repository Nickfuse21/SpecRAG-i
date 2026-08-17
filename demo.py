"""
Demo run-through: the four hallucination controls, each shown firing.

Run this, not a single happy-path query. One good answer proves the pipeline is
connected; it proves nothing about the part of the brief that is actually being
assessed, which is what the system does when it should NOT answer. So the
script walks four questions chosen so that each one stops at a different place:

  1. answerable            -> answers, cites a real clause, verifies grounded
  2. pinpoint lookup       -> the user named a clause, so it is fetched, not
                              ranked; the gate is deliberately bypassed here
                              because "here is the clause you asked for" is the
                              right answer even at a poor similarity score
  3. obviously off-domain  -> REFUSED AT THE GATE, before the LLM is called
  4. plausible fiction     -> passes the gate (it reuses real 3GPP vocabulary,
                              so retrieval legitimately finds related text) and
                              has to be caught downstream by citation validity
                              and groundedness instead

Question 4 is the one worth watching. It is the residual case the gate cannot
catch by construction, and it is where controls #3 and #4 earn their place.

Usage
-----
    python demo.py
    python demo.py --query "your own question"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import RERANK_SCORE_THRESHOLD, TARGET_RELEASE  # noqa: E402

DEMO = [
    ("answerable",
     "What does the UE do when it receives an RRCReconfiguration message?"),
    ("pinpoint lookup",
     "What does clause 5.3.3.7 of TS 38.331 say?"),
    ("out of scope",
     "What is the best pizza dough recipe for a home oven?"),
    ("plausible fiction",
     "What is the purpose of the RRCQuantumResume message?"),
]

RULE = "=" * 78


def show(label: str, question: str, retriever, generator, verifier) -> None:
    from src.verification.groundedness import apply

    print(f"\n{RULE}\n[{label}]  {question}\n{RULE}")

    t0 = time.time()
    res = retriever.retrieve(question)

    if res.hits:
        print(f"\n  retrieved {len(res.hits)} passages, "
              f"best rerank score {res.top_score:.3f} "
              f"(gate threshold {RERANK_SCORE_THRESHOLD})")
        for i, h in enumerate(res.hits[:3], 1):
            print(f"    [{i}] {h.get('rerank_score', 0):.3f}  {h['citation']} — "
                  f"{h.get('clause_title', '')[:48]}")

    # ---- CONTROL #2 ----
    if res.refused:
        print(f"\n  REFUSED AT THE GATE — {res.reason}")
        print("  The LLM was never called, so it had no opportunity to invent")
        print("  anything. The closest passages are still shown above so the")
        print("  user can judge for themselves.")
        print(f"\n  {time.time() - t0:.1f}s")
        return

    ans = generator.generate(question, res.hits)

    # ---- CONTROL #3 ----
    if ans.invalid_citations:
        print(f"\n  dropped fabricated passage numbers: {ans.invalid_citations}")

    if ans.refused:
        print(f"\n  REFUSED AT GENERATION — {ans.refusal_reason}")
        if ans.missing:
            print(f"  missing: {ans.missing}")
        print(f"\n  {time.time() - t0:.1f}s")
        return

    # ---- CONTROL #4 ----
    verdict = verifier.verify(ans)
    ans = apply(ans, verdict)

    print(f"\n  groundedness {verdict.score:.2f} -> {verdict.label.upper()}")
    for cv in verdict.claims:
        flag = "  CONTRADICTED" if cv.contradicted else ""
        print(f"    {cv.score:.2f}{flag}  {cv.text[:66]}")

    if ans.refused:
        print(f"\n  REFUSED AFTER VERIFICATION — {ans.refusal_reason}")
        print(f"\n  {time.time() - t0:.1f}s")
        return

    print(f"\n  ANSWER:\n")
    for line in ans.text.split("\n"):
        print(f"    {line}")

    print("\n  claims and their sources:")
    for c in ans.claims:
        force = f" [{c.normative}]" if c.normative != "none" else ""
        print(f"    - {c.text[:70]}{force}")
        for cite in c.citations:
            print(f"        {cite}")

    print(f"\n  {time.time() - t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo the four hallucination controls")
    ap.add_argument("--query", default=None, help="run one question of your own instead")
    args = ap.parse_args()

    from src.generation.answer import Generator
    from src.retrieval.pipeline import Retriever
    from src.verification.groundedness import Verifier

    print(f"loading models (Rel-{TARGET_RELEASE} corpus) ...")
    t0 = time.time()
    retriever = Retriever()
    generator = Generator()
    # Both judges here, not just the NLI one the API uses: the demo is where
    # the verification story is being told, so it should run the full version.
    verifier = Verifier(judge="both")
    print(f"ready in {time.time() - t0:.0f}s — "
          f"{retriever.store.count():,} passages indexed")

    items = [("custom", args.query)] if args.query else DEMO
    for label, q in items:
        show(label, q, retriever, generator, verifier)

    print(f"\n{RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
