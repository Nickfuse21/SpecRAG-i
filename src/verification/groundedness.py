"""
CONTROL #4: does the answer actually follow from the passages?

Citation validity (Control #3, in the generation module) proved that every
citation points at a real passage we retrieved. It did NOT prove the passage
says what the claim says. A model can cite passage [2] correctly and still
describe it wrongly — that is the most dangerous residual failure in a RAG
system, because every surface signal looks right: the answer is fluent, the
citation resolves, the source is real, and the content is wrong.

So we check entailment directly: for each claim, does its cited passage
ENTAIL it, CONTRADICT it, or neither?

Two judges, because they fail differently
-----------------------------------------
NLI model (cross-encoder/nli-deberta-v3-base)
    Fast, free, deterministic, runs on the GPU we already have. Trained on
    SNLI/MNLI, which are short everyday sentence pairs — so a 900-token
    clause of 3GPP legalese is well outside its training distribution and it
    gets brittle on long premises. Mitigated by scoring the claim against
    each cited passage separately and taking the best, rather than against
    one concatenated blob.

LLM judge (Gemini Flash-Lite, temperature 0)
    Handles long technical premises and conditional logic far better, and
    understands that "may" and "shall" are not interchangeable. Costs an API
    call per claim and is only near-deterministic.

Using both and taking the MINIMUM is deliberate. This is a safety check, and
for a safety check disagreement should mean caution, not a coin flip. A
claim only counts as grounded if neither judge doubts it.

The abstention policy — per claim, never on the average
------------------------------------------------------
    every claim >= GROUNDED_THRESHOLD (0.90)  -> "grounded", return as-is
    at least one claim >= PARTIAL_THRESHOLD (0.60)
                                              -> "partial": drop the claims
                                                 below the bar, serve the rest,
                                                 and say which were dropped
    no claim clears the bar, or any claim is CONTRADICTED
                                              -> REFUSE

An answer where four claims are solid and one is invented is not "wrong" — it
is mostly right with a poisoned sentence in it, and the honest move is to serve
the four and say the fifth could not be verified. Throwing away good grounded
content because of one bad claim trains users to route around the system.

The first implementation gated that partial path on the MEAN claim score, which
defeated it: entailment scores here are sharply bimodal (p25 0.05, median 0.89),
so the mean lands in the empty middle and describes no claim that exists. It
measured 18% coverage — refusing four answerable questions in five, including
answers with nine well-supported claims. See verify() for the numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    GROUNDED_THRESHOLD,
    NLI_FP16,
    NLI_MODEL,
    PARTIAL_THRESHOLD,
    REFUSAL_MESSAGE,
    DEVICE,
    UTIL_MODEL,
    secrets,
)

# sentence-transformers ships this model with a fixed label order.
CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2

JUDGE_PROMPT = """\
You are checking whether a CLAIM is fully supported by a PASSAGE from a 3GPP
specification.

Answer with a single number between 0 and 1:
  1.0  the passage states the claim, including any conditions and the same
       normative force (shall / should / may)
  0.5  the passage is related and does not contradict the claim, but does not
       state it
  0.0  the passage does not support the claim, or contradicts it, or the
       claim changes the normative force (e.g. passage says "may", claim
       says "must"), or the claim drops a condition the passage attaches

Be strict. If you are unsure, score low. Output only the number.

PASSAGE:
{passage}

CLAIM:
{claim}
"""


@dataclass
class ClaimVerdict:
    text: str
    score: float
    contradicted: bool
    detail: dict


@dataclass
class Verdict:
    label: str                    # "grounded" | "partial" | "refused"
    score: float
    claims: list[ClaimVerdict]
    kept: list[int]               # indices of claims that survived
    dropped: list[int]


class NLIJudge:
    def __init__(self, model_name: str = NLI_MODEL, device: str = DEVICE,
                 fp16: bool = NLI_FP16):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        # .half() on the CrossEncoder container, not on a submodule — see the
        # note in src/retrieval/rerank.py for what rebinding `.model` breaks.
        if fp16 and device == "cuda":
            self.model.half()

    def score(self, passage: str, claim: str) -> tuple[float, bool]:
        import numpy as np
        # Cast to float32 before the softmax: exp() on fp16 logits overflows to
        # inf around 11, and this head's logits comfortably exceed that, which
        # would turn a confident entailment into nan and score it as ungrounded.
        logits = np.asarray(self.model.predict([(passage, claim)])[0], dtype=np.float32)
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        return float(probs[ENTAILMENT]), bool(probs.argmax() == CONTRADICTION)


class LLMJudge:
    def __init__(self, model: str = UTIL_MODEL):
        from google import genai
        if not secrets.GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is empty — cannot use the LLM judge.")
        self.client = genai.Client(api_key=secrets.GEMINI_API_KEY)
        self.model = model

    def score(self, passage: str, claim: str) -> tuple[float, bool]:
        from google.genai import types
        from src.llm_retry import with_retry

        try:
            resp = with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=JUDGE_PROMPT.format(passage=passage[:12000], claim=claim),
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=8),
                ),
                what=f"judge ({self.model})",
            )
        except Exception:                                          # noqa: BLE001
            # A judge that could not run has not established anything. Score 0
            # and let the minimum-across-judges rule do the rest: an unverified
            # claim gets dropped or refused, never quietly passed through.
            return 0.0, False

        try:
            s = float((resp.text or "").strip().split()[0])
        except (ValueError, IndexError):
            s = 0.0     # unreadable judge output is treated as "not supported"
        return max(0.0, min(1.0, s)), s == 0.0


class Verifier:
    def __init__(self, judge: str = "both"):
        self.nli = NLIJudge() if judge in ("nli", "both") else None
        self.llm = LLMJudge() if judge in ("llm", "both") else None
        if not self.nli and not self.llm:
            raise ValueError("judge must be one of: nli, llm, both")

    def _score_claim(self, claim_text: str, passages: list[str]) -> ClaimVerdict:
        """
        Score against each cited passage separately and keep the best.

        Concatenating the passages into one premise would let a claim look
        supported because the words appear SOMEWHERE across three unrelated
        clauses — which is exactly the stitched-together hallucination this
        check exists to catch.
        """
        if not passages:
            return ClaimVerdict(claim_text, 0.0, False, {"reason": "no cited passage"})

        best, contradicted, detail = 0.0, False, {}
        for name, judge in (("nli", self.nli), ("llm", self.llm)):
            if judge is None:
                continue
            scores, contras = [], []
            for p in passages:
                s, c = judge.score(p, claim_text)
                scores.append(s)
                contras.append(c)
            detail[name] = round(max(scores), 4)
            contradicted = contradicted or all(contras)

        # minimum across judges — disagreement means caution
        best = min(detail.values()) if detail else 0.0
        return ClaimVerdict(claim_text, best, contradicted, detail)

    def verify(self, answer) -> Verdict:
        """`answer` is a src.generation.answer.Answer."""
        hits = answer.hits
        verdicts: list[ClaimVerdict] = []

        for c in answer.claims:
            passages = [hits[p - 1]["text"] for p in c.passages if 1 <= p <= len(hits)]
            verdicts.append(self._score_claim(c.text, passages))

        if not verdicts:
            return Verdict("refused", 0.0, [], [], [])

        scores = [v.score for v in verdicts]
        mean = sum(scores) / len(scores)

        # A single contradicted claim is disqualifying on its own. Averages
        # hide it: four solid claims plus one that the source directly
        # contradicts still averages above 0.8.
        if any(v.contradicted for v in verdicts):
            return Verdict("refused", mean, verdicts, [],
                           list(range(len(verdicts))))

        # The decision is PER CLAIM, not on the mean.
        #
        # The first version of this gated the whole answer on the mean, and it
        # measured 18% coverage on the eval set — it refused four answerable
        # questions in five. The reason is that entailment scores here are
        # sharply bimodal (measured: p25 = 0.05, median = 0.89, p75 = 0.99).
        # A claim is either clearly stated by its passage or clearly not; almost
        # nothing lands in between. So the MEAN of those scores falls in the
        # empty middle and describes no claim that exists. One real answer scored
        # claims [0.87, 0, 0, 0, 0.98, 0, 0, 0, 0.81, 0.3, 0.97, 0.92, 0.98,
        # 0.99, 0.99, 0.93] -> mean 0.55 -> refused, discarding nine
        # well-supported claims because six others scored zero.
        #
        # That directly contradicted this module's own stated policy: serve the
        # claims that hold, drop the ones that do not, and say so. The mean gate
        # meant the partial path could only fire when the answer was already
        # mostly fine, which is exactly when it was least needed.
        #
        # This is not a weakening. Every claim that reaches the user still has to
        # clear PARTIAL_THRESHOLD on its own — the same bar as before. What
        # changed is that failing claims no longer take passing ones down with
        # them. Refusal is now reserved for the cases that genuinely warrant it:
        # a contradiction, or nothing surviving at all.
        kept = [i for i, v in enumerate(verdicts) if v.score >= PARTIAL_THRESHOLD]
        dropped = [i for i in range(len(verdicts)) if i not in kept]

        if not kept:
            return Verdict("refused", mean, verdicts, [], list(range(len(verdicts))))

        # `min`, not `mean`, for the grounded badge: "grounded" is a claim about
        # every sentence in the answer, so the weakest one decides it. A mean
        # would let one unsupported claim hide behind several strong ones, which
        # is the failure this whole module exists to prevent.
        kept_scores = [scores[i] for i in kept]
        label = "grounded" if (not dropped and min(kept_scores) >= GROUNDED_THRESHOLD) \
            else "partial"

        # Report the score over what is actually served, not over claims we just
        # threw away — otherwise a "partial" answer carries the score of the
        # version the user never sees.
        return Verdict(label, sum(kept_scores) / len(kept_scores),
                       verdicts, kept, dropped)


def apply(answer, verdict: Verdict):
    """Rewrite the answer in line with the verdict. Mutates and returns it."""
    if verdict.label == "refused":
        answer.refused = True
        answer.refusal_reason = (
            f"groundedness {verdict.score:.2f} below {PARTIAL_THRESHOLD}"
            if not any(v.contradicted for v in verdict.claims)
            else "a claim is contradicted by its own cited passage"
        )
        answer.text = REFUSAL_MESSAGE
        answer.claims = []
    elif verdict.label == "partial":
        answer.claims = [answer.claims[i] for i in verdict.kept]
    return answer


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieve, answer, then verify")
    ap.add_argument("--query", required=True)
    ap.add_argument("--judge", default="both", choices=["nli", "llm", "both"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from src.generation.answer import Generator
    from src.retrieval.pipeline import Retriever

    res = Retriever().retrieve(args.query)
    if res.refused:
        print(f"REFUSED at the gate — {res.reason}")
        return 0

    ans = Generator().generate(args.query, res.hits)
    if ans.refused:
        print(f"REFUSED at generation — {ans.refusal_reason}")
        return 0

    v = Verifier(judge=args.judge).verify(ans)
    ans = apply(ans, v)

    if args.json:
        out = ans.to_dict()
        out["verification"] = {
            "label": v.label,
            "score": round(v.score, 4),
            "claims": [{"text": c.text, "score": c.score,
                        "contradicted": c.contradicted, "detail": c.detail}
                       for c in v.claims],
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    print(f"\nQ: {args.query}\n")
    print(f"[{v.label.upper()}  groundedness {v.score:.2f}]\n")
    print(ans.text)
    print("\nper-claim:")
    for cv in v.claims:
        flag = " CONTRADICTED" if cv.contradicted else ""
        print(f"  {cv.score:.2f}{flag}  {cv.text[:88]}   {cv.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
