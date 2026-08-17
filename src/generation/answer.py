"""
Generation: turn retrieved chunks into an answer that cannot cite what it
was not given.

The prompt contract
-------------------
Everything here is built on one idea: the model's job is not to know things.
It is to READ the passages it was handed and report what they say. Every
design choice follows from that.

  temperature = 0
      Not a style preference. In a factual answer, sampling a
      lower-probability token means sampling a less likely FACT. There is no
      creativity worth having here, and eval numbers have to be reproducible
      run to run or the ablation table means nothing.

  numbered passages, not a wall of text
      Each chunk is presented as [1], [2], ... and the model must cite by
      that number. It never writes a citation STRING. The string is rendered
      afterwards from our own metadata. This is the difference between "the
      model was asked to cite accurately" and "the model is structurally
      incapable of inventing a citation" — it can only emit an integer, and
      an integer that is out of range is caught by a range check, not by
      hoping.

  structured JSON output
      Free text forces us to regex an answer apart to find its claims and
      citations. A schema gives us claims already separated, each with its
      own support list — which is exactly the shape the verification stage
      needs. Parsing prose to reconstruct structure the model already had is
      a self-inflicted wound.

  an explicit "answerable" field
      The model is given permission to say the passages do not contain the
      answer, and a place to say it. Without that, "I don't know" competes
      with a fluent guess in the same output slot, and fluency wins. With
      it, refusing is just filling in a field.

The normative-language rule in the prompt is 3GPP-specific and matters more
than it looks: `shall` is a requirement, `should` is a recommendation, `may`
is a permission. A summary that turns a `may` into a `shall` has invented a
requirement — technically a hallucination even though every word came from
the source.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    GEN_MAX_OUTPUT_TOKENS,
    GEN_MODEL,
    GEN_TEMPERATURE,
    REFUSAL_MESSAGE,
    secrets,
)

SYSTEM_PROMPT = """\
You answer questions about 3GPP telecommunications specifications.

You will be given a QUESTION and a numbered list of PASSAGES taken verbatim
from 3GPP specifications. Answer ONLY from those passages.

Rules, in order of importance:

1. Never state anything that is not supported by the passages. If the
   passages do not contain the answer, set "answerable" to false and explain
   what is missing. This is always an acceptable outcome and is strongly
   preferred over a plausible guess.

2. Every claim you make must cite the passage numbers it comes from, using
   the integers given. Do not write specification names, versions or clause
   numbers as citations — cite the integer only.

3. Preserve normative force exactly. In 3GPP:
     "shall"  = a requirement       "shall not" = a prohibition
     "should" = a recommendation    "may"       = a permission
   Never upgrade or downgrade these. If a passage says the UE "may" do
   something, do not write that it "must".

4. Preserve conditions. Requirements in 3GPP are almost always conditional
   ("if X, the UE shall Y"). Never state the action without its condition.

5. If passages disagree, say so and cite both. Do not silently pick one.

6. Do not add background knowledge, context, or explanation from your own
   training. If it is not in the passages, it does not go in the answer.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {
            "type": "boolean",
            "description": "true only if the passages contain enough to answer",
        },
        "answer": {
            "type": "string",
            "description": "The full answer in prose. Empty if not answerable.",
        },
        "claims": {
            "type": "array",
            "description": "The answer broken into individually checkable statements.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "passages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Passage numbers supporting this claim",
                    },
                    "normative": {
                        "type": "string",
                        "enum": ["shall", "shall not", "should", "may", "none"],
                        "description": "Normative force of this claim, if any",
                    },
                },
                "required": ["text", "passages", "normative"],
            },
        },
        "missing": {
            "type": "string",
            "description": "If not answerable, what information would be needed.",
        },
    },
    "required": ["answerable", "answer", "claims", "missing"],
}


def build_prompt(question: str, hits: list[dict]) -> str:
    parts = [f"QUESTION: {question}\n", "PASSAGES:"]
    for i, h in enumerate(hits, 1):
        parts.append(
            f"\n[{i}] {h['citation']}"
            f"\n    section: {h.get('breadcrumb', '')}"
            f"\n    title:   {h.get('clause_title', '')}"
            f"\n{h['text']}"
        )
    return "\n".join(parts)


@dataclass
class Claim:
    text: str
    passages: list[int]
    normative: str = "none"
    citations: list[str] = field(default_factory=list)


@dataclass
class Answer:
    answerable: bool
    text: str
    claims: list[Claim]
    missing: str = ""
    refused: bool = False
    refusal_reason: str = ""
    hits: list[dict] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answerable": self.answerable,
            "answer": self.text,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "missing": self.missing,
            "claims": [
                {"text": c.text, "passages": c.passages,
                 "normative": c.normative, "citations": c.citations}
                for c in self.claims
            ],
            "sources": [
                {"n": i, "citation": h["citation"], "chunk_id": h["chunk_id"],
                 "breadcrumb": h.get("breadcrumb", "")}
                for i, h in enumerate(self.hits, 1)
            ],
            "invalid_citations": self.invalid_citations,
        }


class Generator:
    def __init__(self, model: str = GEN_MODEL):
        from google import genai

        if not secrets.GEMINI_API_KEY:
            raise SystemExit(
                "GEMINI_API_KEY is empty. Copy .env.example to .env and paste your key."
            )
        self.client = genai.Client(api_key=secrets.GEMINI_API_KEY)
        self.model = model

    def generate(self, question: str, hits: list[dict]) -> Answer:
        from google.genai import types

        if not hits:
            return Answer(False, REFUSAL_MESSAGE, [], refused=True,
                          refusal_reason="no passages retrieved")

        from src.llm_retry import with_retry

        try:
            resp = with_retry(
                lambda: self.client.models.generate_content(
                    model=self.model,
                    contents=build_prompt(question, hits),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=GEN_TEMPERATURE,
                        max_output_tokens=GEN_MAX_OUTPUT_TOKENS,
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                    ),
                ),
                what=f"generate ({self.model})",
            )
        except Exception as exc:                                   # noqa: BLE001
            # The model is unreachable after retries. Refuse — an answer we
            # could not generate is not an answer we may guess at, and a stack
            # trace reaching a user is strictly worse than a refusal.
            return Answer(False, REFUSAL_MESSAGE, [], refused=True,
                          refusal_reason=f"generation unavailable: {str(exc)[:120]}",
                          hits=hits)

        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            # Schema-constrained output should make this unreachable. If it
            # ever fires, refusing is the only safe move — we have no way to
            # tell which parts of a malformed answer were grounded.
            return Answer(False, REFUSAL_MESSAGE, [], refused=True,
                          refusal_reason="model returned unparseable output",
                          hits=hits)

        return self._validate(data, hits)

    # ----------------------------------------------------------------
    def _validate(self, data: dict, hits: list[dict]) -> Answer:
        """
        CONTROL #3: citation validity.

        Two checks, and the second is the one people forget:

          a) every cited passage number is in range. An out-of-range number
             is a fabricated source.
          b) every claim cites at least one passage. An uncited claim is an
             unsupported assertion wearing the same font as a supported one.

        Citation STRINGS are built here from our metadata, never taken from
        the model. That is what makes a fabricated citation impossible
        rather than merely unlikely.
        """
        n = len(hits)
        claims: list[Claim] = []
        invalid: list[int] = []

        for raw in data.get("claims") or []:
            nums = [p for p in (raw.get("passages") or []) if isinstance(p, int)]
            good = [p for p in nums if 1 <= p <= n]
            invalid.extend(p for p in nums if not (1 <= p <= n))
            claims.append(Claim(
                text=str(raw.get("text", "")).strip(),
                passages=good,
                normative=str(raw.get("normative", "none")),
                citations=[hits[p - 1]["citation"] for p in good],
            ))

        answerable = bool(data.get("answerable"))
        uncited = [c for c in claims if not c.passages]

        ans = Answer(
            answerable=answerable,
            text=str(data.get("answer", "")).strip(),
            claims=claims,
            missing=str(data.get("missing", "")).strip(),
            hits=hits,
            invalid_citations=sorted(set(invalid)),
        )

        if not answerable:
            ans.refused = True
            ans.refusal_reason = "model reported the passages do not contain the answer"
            ans.text = REFUSAL_MESSAGE
        elif claims and len(uncited) == len(claims):
            ans.refused = True
            ans.refusal_reason = "no claim cited any passage"
            ans.text = REFUSAL_MESSAGE

        return ans


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieve then answer")
    ap.add_argument("--query", required=True)
    ap.add_argument("--json", action="store_true", help="dump the full structured answer")
    args = ap.parse_args()

    from src.retrieval.pipeline import Retriever

    r = Retriever()
    res = r.retrieve(args.query)

    if res.refused:
        print(f"\nREFUSED at the gate — {res.reason}")
        print(f"\n{REFUSAL_MESSAGE}\n")
        for i, h in enumerate(res.hits[:3], 1):
            print(f"  [{i}] {h['citation']} — {h.get('clause_title','')[:60]}")
        return 0

    ans = Generator().generate(args.query, res.hits)

    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"\nQ: {args.query}\n")
    if ans.refused:
        print(f"REFUSED — {ans.refusal_reason}")
        if ans.missing:
            print(f"missing: {ans.missing}")
        return 0

    print(ans.text)
    print("\nclaims:")
    for c in ans.claims:
        force = f" [{c.normative}]" if c.normative != "none" else ""
        print(f"  - {c.text}{force}")
        for cite in c.citations:
            print(f"      {cite}")
    if ans.invalid_citations:
        print(f"\n! fabricated passage numbers dropped: {ans.invalid_citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
