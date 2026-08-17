"""
The ablation table — the main evidence for "quality and effectiveness".

The claim this project makes is that four specific controls reduce
hallucination. A working demo does not support that claim: a demo shows the
system answering, not that any individual piece is carrying its weight. So we
turn each control off and measure what breaks.

Two tables, because the controls act at two different places
------------------------------------------------------------
RETRIEVAL (answerable questions only). Nothing here can refuse; these arms
differ only in how they rank. If hybrid search and reranking do not move
Recall/MRR/nDCG, they are cost with no benefit and should be cut.

    A  dense only          bge-m3 vector search, no query understanding
    B  + hybrid            + BM25 + RRF fusion + spec/clause extraction
    C  + rerank            + bge-reranker-v2-m3 cross-encoder

ANSWERS (all questions). These arms retrieve identically — arm C's retrieval —
and differ in what they refuse to do with the result.

    C  no gate, no verify   answers whatever retrieval returned
    D  + gate               refuses before the LLM when the best score is weak
    E  + gate + verify      also refuses after the LLM when claims are unsupported

Metrics, and why these
----------------------
hallucination rate  On an UNANSWERABLE question, any non-refused answer is a
                    hallucination by construction — the corpus cannot support
                    it, so there is nothing to grade against and no human
                    labelling needed. This is the headline number.

coverage            Share of ANSWERABLE questions the system actually answers.
                    Reported next to hallucination rate because a system that
                    refuses everything scores a perfect 0% hallucination rate,
                    and any table that hides that is dishonest.

gold citation rate  Of the answerable questions we answered, how often at
                    least one cited passage came from the gold clause. Catches
                    the case where the system answers fluently from the wrong
                    part of the corpus — right shape, wrong source.

faithfulness        Mean groundedness score over answered questions (arm E
                    only, since it is the verifier's own output). Measured with
                    the NLI judge by default — see run_answers() for why that
                    is the honest default rather than the stronger one.

Usage
-----
    python -m eval.run_eval --retrieval          # no API calls, GPU only
    python -m eval.run_eval --answers            # calls Gemini
    python -m eval.run_eval --all
    python -m eval.run_eval --answers --limit 20 # quick pass while iterating
    python -m eval.run_eval --answers --judge both   # needs API quota; slow
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EVAL_DIR, FINAL_TOP_K, FUSED_TOP_K  # noqa: E402

GOLD = Path(__file__).resolve().parent / "gold.jsonl"


def load_gold() -> list[dict]:
    return [json.loads(l) for l in GOLD.open(encoding="utf-8") if l.strip()]


# --------------------------------------------------------------------------
# relevance
# --------------------------------------------------------------------------
def is_relevant(hit: dict, gold: list[dict]) -> bool:
    """
    Does this chunk actually contain the answer's clause?

    Three cases, and the third is the one a naive `==` gets wrong:

      exact        chunk's clause_id IS the gold clause.
      descendant   gold "5.3.5" matches chunk "5.3.5.3" — a sub-clause lives
                   inside the clause the question was about.
      merged-up    Rule 4 folds a tiny clause into its ancestor, so a chunk
                   labelled "5.3.5" can physically contain 5.3.5.1's text.
                   That counts only if the gold clause is in merged_clause_ids
                   — we check the record rather than assuming, because a bare
                   ancestor match would credit any chunk anywhere above the
                   gold clause in the tree and quietly inflate recall.
    """
    for g in gold:
        if hit.get("spec") != g["spec"]:
            continue
        cid = hit.get("clause_id", "")
        if cid == g["clause"] or cid.startswith(g["clause"] + "."):
            return True
        if g["clause"] in (hit.get("merged_clause_ids") or []):
            return True
    return False


def rank_metrics(hits: list[dict], gold: list[dict], k: int) -> dict:
    rel = [1 if is_relevant(h, gold) else 0 for h in hits[:k]]
    first = next((i for i, r in enumerate(rel) if r), None)

    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    # Binary relevance, so the ideal ranking is every relevant chunk first.
    # Capped at the number actually found: normalising against a count of
    # gold CLAUSES would compare against passages that may not exist as
    # separate chunks, making a perfect ranking score below 1.0.
    ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rel), k))) or 1.0

    return {
        "recall": 1.0 if first is not None else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg": dcg / ideal,
    }


# --------------------------------------------------------------------------
# retrieval ablation
# --------------------------------------------------------------------------
RETRIEVAL_ARMS = [
    ("A  dense only", dict(load_reranker=False, use_bm25=False,
                           use_gate=False, use_query_understanding=False)),
    ("B  + hybrid",   dict(load_reranker=False, use_bm25=True,
                           use_gate=False, use_query_understanding=True)),
    ("C  + rerank",   dict(load_reranker=True,  use_bm25=True,
                           use_gate=False, use_query_understanding=True)),
]


def run_retrieval(rows: list[dict]) -> list[dict]:
    from src.retrieval.pipeline import Retriever

    answerable = [r for r in rows if r["answerable"]]
    print(f"retrieval ablation over {len(answerable)} answerable questions\n")

    out = []
    for name, cfg in RETRIEVAL_ARMS:
        r = Retriever(**cfg)
        t0 = time.time()
        deep, shallow = [], []
        for row in answerable:
            res = r.retrieve(row["question"], top_k=FUSED_TOP_K)
            deep.append(rank_metrics(res.hits, row["gold"], FUSED_TOP_K))
            shallow.append(rank_metrics(res.hits, row["gold"], FINAL_TOP_K))

        n = len(answerable)
        rec = {
            "arm": name,
            f"recall@{FINAL_TOP_K}": sum(m["recall"] for m in shallow) / n,
            f"recall@{FUSED_TOP_K}": sum(m["recall"] for m in deep) / n,
            f"mrr@{FUSED_TOP_K}": sum(m["mrr"] for m in deep) / n,
            f"ndcg@{FINAL_TOP_K}": sum(m["ndcg"] for m in shallow) / n,
            "sec_per_query": (time.time() - t0) / n,
        }
        out.append(rec)
        print(f"  {name:<16} recall@{FINAL_TOP_K} {rec[f'recall@{FINAL_TOP_K}']:.3f}   "
              f"recall@{FUSED_TOP_K} {rec[f'recall@{FUSED_TOP_K}']:.3f}   "
              f"mrr {rec[f'mrr@{FUSED_TOP_K}']:.3f}   "
              f"ndcg@{FINAL_TOP_K} {rec[f'ndcg@{FINAL_TOP_K}']:.3f}   "
              f"{rec['sec_per_query']:.2f}s/q")
        # These arms genuinely need different objects (one has no reranker at
        # all), so they are rebuilt — but `del` alone frees nothing on the GPU.
        # torch's caching allocator holds the blocks, so arm C would load the
        # cross-encoder into whatever is left after A and B's models are still
        # nominally resident, on a card with no room to spare.
        del r
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# answer ablation
# --------------------------------------------------------------------------
ANSWER_ARMS = [
    ("C  no gate, no verify", dict(gate=False, verify=False)),
    ("D  + gate",             dict(gate=True,  verify=False)),
    ("E  + gate + verify",    dict(gate=True,  verify=True)),
]


def run_answers(rows: list[dict], limit: int | None, judge: str = "nli",
                only: set[str] | None = None) -> list[dict]:
    """
    `judge` defaults to "nli", not "both", for two reasons.

    The measurement reason: `src/api/main.py` builds `Verifier(judge="nli")`, so
    NLI-only is what the served system actually does. Evaluating with a stronger
    verifier than production runs would report a hallucination rate nobody gets.

    The practical reason: the LLM judge costs one API call per claim per cited
    passage, which is hundreds of calls across 77 questions x 3 arms. On a
    free-tier key that returns 429 RESOURCE_EXHAUSTED partway through, and
    because a judge that could not run scores 0.0, the failures do not show up
    as errors — they show up as claims that look ungrounded. That silently
    understates coverage and faithfulness, which is worse than not measuring it.

    Use `--judge both` deliberately, on a key with quota, and expect it to be
    slow. `src/demo.py` uses both, since it makes a handful of calls.
    """
    from src.generation.answer import Generator
    from src.retrieval.pipeline import Retriever
    from src.verification.groundedness import Verifier, apply

    if limit:
        ans = [r for r in rows if r["answerable"]][: max(1, limit // 2)]
        una = [r for r in rows if not r["answerable"]][: max(1, limit // 2)]
        rows = ans + una

    n_ans = sum(1 for r in rows if r["answerable"])
    n_una = len(rows) - n_ans
    print(f"\nanswer ablation over {n_ans} answerable + {n_una} unanswerable\n")

    gen = Generator()
    verifier = Verifier(judge=judge)
    out, details = [], []

    # One Retriever for all three arms, with the gate toggled between them.
    # These arms retrieve identically by design — only what they refuse
    # differs — so rebuilding it per arm would reload bge-m3 and the
    # cross-encoder three times. On a 4 GB GPU that is not merely wasteful:
    # with the NLI judge also resident, the second copy is what pushes the
    # driver into spilling to system RAM.
    r = Retriever(use_gate=True)

    for name, cfg in ANSWER_ARMS:
        # `only` exists because arm E is the expensive one: arms C and D make a
        # single generation call per question, E adds a judge call per claim per
        # cited passage. Re-running all three to fix E wastes ~30 minutes of
        # quota on two arms whose numbers did not change.
        if only and name.split()[0] not in only:
            continue
        r.use_gate = cfg["gate"]
        stats = {"answered_unanswerable": 0, "answered_answerable": 0,
                 "gold_cited": 0, "faith": [], "refused_by_gate": 0,
                 "refused_by_gen": 0, "refused_by_verify": 0}
        t0 = time.time()

        for row in rows:
            res = r.retrieve(row["question"])
            refused, why = res.refused, "gate"

            answer = None
            if not refused:
                answer = gen.generate(row["question"], res.hits)
                refused, why = answer.refused, "generation"

                if not refused and cfg["verify"]:
                    v = verifier.verify(answer)
                    answer = apply(answer, v)
                    refused, why = answer.refused, "verification"
                    if not refused:
                        stats["faith"].append(v.score)

            if refused:
                stats[{"gate": "refused_by_gate", "generation": "refused_by_gen",
                       "verification": "refused_by_verify"}[why]] += 1
            elif row["answerable"]:
                stats["answered_answerable"] += 1
                cited = {c for cl in answer.claims for c in cl.passages}
                if any(is_relevant(res.hits[p - 1], row["gold"])
                       for p in cited if 1 <= p <= len(res.hits)):
                    stats["gold_cited"] += 1
            else:
                stats["answered_unanswerable"] += 1

            details.append({"arm": name, "qid": row["qid"],
                            "answerable": row["answerable"],
                            "refused": refused, "stage": why if refused else ""})

        rec = {
            "arm": name,
            "hallucination_rate": stats["answered_unanswerable"] / n_una if n_una else 0.0,
            "coverage": stats["answered_answerable"] / n_ans if n_ans else 0.0,
            "gold_citation_rate": (stats["gold_cited"] / stats["answered_answerable"]
                                   if stats["answered_answerable"] else 0.0),
            "faithfulness": (sum(stats["faith"]) / len(stats["faith"])
                             if stats["faith"] else None),
            "refused_by_gate": stats["refused_by_gate"],
            "refused_by_generation": stats["refused_by_gen"],
            "refused_by_verification": stats["refused_by_verify"],
            "sec_per_query": (time.time() - t0) / len(rows),
        }
        out.append(rec)
        f = f"{rec['faithfulness']:.3f}" if rec["faithfulness"] is not None else "  -  "
        print(f"  {name:<22} halluc {rec['hallucination_rate']:.3f}   "
              f"coverage {rec['coverage']:.3f}   "
              f"gold-cite {rec['gold_citation_rate']:.3f}   "
              f"faith {f}   {rec['sec_per_query']:.1f}s/q")

    (EVAL_DIR / "answer_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
def markdown_tables(retrieval: list[dict], answers: list[dict]) -> str:
    md = []
    if retrieval:
        md.append("### Retrieval (55 answerable questions)\n")
        md.append(f"| Arm | Recall@{FINAL_TOP_K} | Recall@{FUSED_TOP_K} | "
                  f"MRR@{FUSED_TOP_K} | nDCG@{FINAL_TOP_K} | s/query |")
        md.append("|---|---|---|---|---|---|")
        for r in retrieval:
            md.append(f"| {r['arm'].strip()} | {r[f'recall@{FINAL_TOP_K}']:.3f} | "
                      f"{r[f'recall@{FUSED_TOP_K}']:.3f} | {r[f'mrr@{FUSED_TOP_K}']:.3f} | "
                      f"{r[f'ndcg@{FINAL_TOP_K}']:.3f} | {r['sec_per_query']:.2f} |")
        md.append("")
    if answers:
        md.append("### Answers (55 answerable + 22 unanswerable)\n")
        md.append("| Arm | Hallucination rate | Coverage | Gold-citation rate | Faithfulness |")
        md.append("|---|---|---|---|---|")
        for r in answers:
            f = f"{r['faithfulness']:.3f}" if r["faithfulness"] is not None else "—"
            md.append(f"| {r['arm'].strip()} | {r['hallucination_rate']:.1%} | "
                      f"{r['coverage']:.1%} | {r['gold_citation_rate']:.1%} | {f} |")
    return "\n".join(md)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the ablation study")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="answers arm only: use a subset while iterating")
    ap.add_argument("--arms", default=None,
                    help="answers arm only: comma-separated subset, e.g. --arms E")
    ap.add_argument("--judge", default="nli", choices=["nli", "llm", "both"],
                    help="verification judge for arm E; default nli, which is what the API serves")
    args = ap.parse_args()

    if not (args.retrieval or args.answers or args.all):
        ap.error("pick at least one of --retrieval / --answers / --all")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_gold()

    retrieval = run_retrieval(rows) if (args.retrieval or args.all) else []
    only = {a.strip().upper() for a in args.arms.split(",")} if args.arms else None
    answers = (run_answers(rows, args.limit, args.judge, only)
               if (args.answers or args.all) else [])

    report = {"retrieval": retrieval, "answers": answers}
    (EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = markdown_tables(retrieval, answers)
    (EVAL_DIR / "results.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nwrote {EVAL_DIR/'results.json'} and results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
