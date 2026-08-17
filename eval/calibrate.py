"""
Fit RERANK_SCORE_THRESHOLD — CONTROL #2's only free parameter.

Why this file has to exist
--------------------------
The gate is the strongest of the four hallucination controls, because it is
the only one that refuses BEFORE the model is called: a model that is never
asked cannot invent anything. But it is strong in proportion to one number,
and `RERANK_SCORE_THRESHOLD = 0.0` in config.py was a placeholder — a round
number someone picked because it looked like a natural midpoint for a logit.

A threshold that was guessed is indefensible in both directions. Too low and
the gate never fires, so the control is decorative. Too high and it refuses
questions the corpus genuinely answers, which users experience as the system
being broken and which trains them to stop trusting the refusals that matter.

So we measure. Score every question in the gold set through the real reranker
with the gate DISABLED, split the resulting max-scores by whether the corpus
can actually answer the question, and look at where the two distributions
separate.

What "correct" means here
-------------------------
This is not a symmetric trade-off and it must not be fitted as one. The brief
ranks minimal hallucination first, so a false answer costs more than a false
refusal. But "refuse everything" scores perfectly on hallucination and is
useless, so we cannot simply maximise refusals either.

The rule used below, stated up front so the number is auditable:

    among all thresholds whose FALSE-REFUSAL rate on answerable questions is
    at most MAX_FALSE_REFUSAL, take the one with the highest CORRECT-REFUSAL
    rate on unanswerable questions; break ties toward the LOWER threshold.

That is a constrained optimisation, not an F-score: it fixes the cost we are
willing to pay in usability and then buys as much safety as that budget
allows. Youden's J (which weights both errors equally) is reported alongside
it for reference, because it is the conventional choice and a reader should be
able to see how far our asymmetric rule moved the answer.

Usage
-----
    python -m eval.calibrate                 # fit and write eval/calibration.json
    python -m eval.calibrate --max-false-refusal 0.05
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EVAL_DIR  # noqa: E402

GOLD = Path(__file__).resolve().parent / "gold.jsonl"

# The usability budget. 10% means: we accept that one answerable question in
# ten gets refused, in exchange for the strongest gate that constraint allows.
# Raise it and the gate gets stricter and more annoying; lower it and the gate
# gets permissive. This is a product decision, so it is a named constant and a
# CLI flag rather than a number buried in a comparison.
MAX_FALSE_REFUSAL = 0.10


def load_gold() -> list[dict]:
    return [json.loads(l) for l in GOLD.open(encoding="utf-8") if l.strip()]


def collect_scores(rows: list[dict]) -> list[dict]:
    """
    Run every question through the real retriever with the gate off.

    `use_gate=False` and nothing else: the point is to observe the score the
    served pipeline would have produced, so every other stage — query
    understanding, hybrid retrieval, fusion, the reranker, fp16 — has to be
    exactly what production runs. A threshold fitted against a different
    pipeline is a threshold for a system nobody is running.
    """
    from src.retrieval.pipeline import Retriever

    r = Retriever(use_gate=False)
    out = []
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        res = r.retrieve(row["question"])
        top = res.hits[0]["rerank_score"] if res.hits else float("-inf")
        out.append({
            "qid": row["qid"],
            "question": row["question"],
            "answerable": row["answerable"],
            "category": row.get("category", "answerable"),
            "top_score": float(top),
            "top_citation": res.hits[0]["citation"] if res.hits else "",
            "top_clause_title": res.hits[0].get("clause_title", "") if res.hits else "",
        })
        rate = i / max(time.time() - t0, 1e-9)
        print(f"  {i:>3}/{len(rows)}  {rate:4.1f} q/s  {row['qid']}  "
              f"{out[-1]['top_score']:+7.2f}  {row['question'][:52]}")

    return out


def sweep(scores: list[dict], max_false_refusal: float) -> dict:
    """
    Walk every candidate cut and score it. Candidates are the observed scores
    themselves — no other value can change the outcome, so a fixed grid would
    only add resolution we cannot actually resolve.
    """
    yes = [s["top_score"] for s in scores if s["answerable"]]
    no = [s["top_score"] for s in scores if not s["answerable"]]
    if not yes or not no:
        raise SystemExit("gold set needs both answerable and unanswerable questions")

    cands = sorted({round(s["top_score"], 4) for s in scores})
    rows = []
    for t in cands:
        # `< t` is refusal, matching pipeline.py exactly. An off-by-one on the
        # comparison operator here would silently shift the fitted number.
        false_refusals = sum(1 for v in yes if v < t)
        correct_refusals = sum(1 for v in no if v < t)
        rows.append({
            "threshold": t,
            "false_refusal_rate": false_refusals / len(yes),
            "correct_refusal_rate": correct_refusals / len(no),
            "answered_unanswerable": len(no) - correct_refusals,
            "refused_answerable": false_refusals,
            "youden_j": (correct_refusals / len(no)) - (false_refusals / len(yes)),
        })

    feasible = [r for r in rows if r["false_refusal_rate"] <= max_false_refusal]
    if feasible:
        best = max(feasible, key=lambda r: (r["correct_refusal_rate"], -r["threshold"]))
    else:
        # Cannot satisfy the budget: the distributions overlap too much to buy
        # any safety at that price. Say so rather than quietly returning the
        # least-bad option, because it means the gate cannot be tuned into
        # usefulness and the honest fix is upstream in retrieval.
        best = None

    best_j = max(rows, key=lambda r: (r["youden_j"], -r["threshold"]))

    return {
        "n_answerable": len(yes),
        "n_unanswerable": len(no),
        "answerable_score_min": min(yes),
        "answerable_score_p05": sorted(yes)[max(0, int(len(yes) * 0.05) - 1)],
        "answerable_score_median": sorted(yes)[len(yes) // 2],
        "unanswerable_score_max": max(no),
        "unanswerable_score_p95": sorted(no)[min(len(no) - 1, int(len(no) * 0.95))],
        "unanswerable_score_median": sorted(no)[len(no) // 2],
        "max_false_refusal_budget": max_false_refusal,
        "chosen": best,
        "youden_optimum": best_j,
        "sweep": rows,
    }


def histogram(scores: list[dict], bins: int = 24) -> None:
    """
    A terminal histogram, because the separation is the whole argument.

    A single fitted number is easy to disagree with and impossible to sanity
    check. Two visibly disjoint (or visibly overlapping) distributions tell you
    immediately whether a threshold can work at all.

    The range is taken from the data rather than hardcoded. A fixed axis is how
    the original 0.0 threshold survived: assume the scores are signed logits in
    -10..+10, and a sigmoid's worth of mass piled up in [0, 1] looks like a
    plausible left-hand cluster instead of the whole distribution.
    """
    vals = [s["top_score"] for s in scores if s["top_score"] != float("-inf")]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.02 or 0.01
    lo, hi = lo - pad, hi + pad
    width = (hi - lo) / bins
    print(f"\n{'score':>8}  {'answerable':<26} {'unanswerable':<26}")
    for b in range(bins):
        a, z = lo + b * width, lo + (b + 1) * width
        y = sum(1 for s in scores if s["answerable"] and a <= s["top_score"] < z)
        n = sum(1 for s in scores if not s["answerable"] and a <= s["top_score"] < z)
        if not y and not n:
            continue
        print(f"{a:>8.1f}  {'#' * y:<26} {'.' * n:<26} {y:>3} / {n:<3}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit the relevance-gate threshold")
    ap.add_argument("--max-false-refusal", type=float, default=MAX_FALSE_REFUSAL,
                    help="usability budget: max share of answerable questions we accept refusing")
    ap.add_argument("--scores", default=None,
                    help="reuse a previous scores file instead of re-running retrieval")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = EVAL_DIR / "gate_scores.json"

    if args.scores:
        scores = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        print(f"reusing {len(scores)} scores from {args.scores}")
    else:
        rows = load_gold()
        print(f"scoring {len(rows)} questions through the real pipeline (gate off)\n")
        scores = collect_scores(rows)
        scores_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"\nwrote {scores_path}")

    report = sweep(scores, args.max_false_refusal)
    histogram(scores)

    print("\n" + "=" * 74)
    print("distributions")
    print("=" * 74)
    print(f"  answerable   ({report['n_answerable']:>2})  "
          f"min {report['answerable_score_min']:+.2f}   "
          f"p05 {report['answerable_score_p05']:+.2f}   "
          f"median {report['answerable_score_median']:+.2f}")
    print(f"  unanswerable ({report['n_unanswerable']:>2})  "
          f"max {report['unanswerable_score_max']:+.2f}   "
          f"p95 {report['unanswerable_score_p95']:+.2f}   "
          f"median {report['unanswerable_score_median']:+.2f}")

    j = report["youden_optimum"]
    print(f"\n  Youden's J optimum (equal weight, for reference): "
          f"{j['threshold']:+.2f}  "
          f"correct-refusal {j['correct_refusal_rate']:.0%}  "
          f"false-refusal {j['false_refusal_rate']:.0%}")

    print("\n" + "=" * 74)
    if report["chosen"] is None:
        print(f"NO THRESHOLD satisfies a {args.max_false_refusal:.0%} false-refusal budget.")
        print("The two distributions overlap too much. Do not paper over this with a")
        print("number — it means retrieval, not the gate, is what needs work.")
        return 1

    c = report["chosen"]
    print(f"CHOSEN  RERANK_SCORE_THRESHOLD = {c['threshold']:+.2f}")
    print("=" * 74)
    print(f"  refuses {c['correct_refusal_rate']:.0%} of unanswerable questions "
          f"({report['n_unanswerable'] - c['answered_unanswerable']}/{report['n_unanswerable']})")
    print(f"  refuses {c['false_refusal_rate']:.0%} of answerable questions "
          f"({c['refused_answerable']}/{report['n_answerable']}) — the price paid")
    print(f"  {c['answered_unanswerable']} unanswerable question(s) still reach the LLM; "
          f"controls #3 and #4 are what catch those")

    out = EVAL_DIR / "calibration.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"\nNow set RERANK_SCORE_THRESHOLD = {c['threshold']:+.2f} in src/config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
