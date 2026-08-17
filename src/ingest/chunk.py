"""
Step 4 of ingestion: turn the clause tree into retrieval-ready chunks.

The parser gave us clauses. A clause is the right SEMANTIC unit but the wrong
RETRIEVAL unit: clause 5.3.5.3 of 38.331 is ~4,000 tokens (too big to put in a
prompt six times over), while clause 5.3.5.1 "General" is 30 tokens (too small
to carry any retrieval signal). Chunking fixes both ends.

The five rules, and why each one exists
---------------------------------------

RULE 1 — A chunk never spans two clauses (except merge-up, see Rule 4).
    3GPP requirements are conditional: "if X, the UE shall Y". The condition
    and the `shall` live in the same clause. Split them across chunks and you
    can retrieve a requirement that reads as unconditional — the single most
    dangerous hallucination this system can produce, because the answer looks
    perfectly citable and is flatly wrong.

RULE 2 — Clause <= CHUNK_MAX_TOKENS becomes exactly one chunk.
    No splitting when splitting isn't needed. Whole-clause chunks are the
    best case: complete context, unambiguous citation.

RULE 3 — Clause > CHUNK_MAX_TOKENS is split at paragraph boundaries, with
    overlap. Target ~CHUNK_TARGET_TOKENS per piece, carrying ~CHUNK_OVERLAP
    tokens of the previous piece's tail into the next. Overlap exists so a
    fact sitting on a boundary is complete in at least one chunk.

RULE 4 — Clause < CHUNK_MIN_TOKENS merges UP into its nearest ancestor.
    Never sideways into a sibling, never down into a child. Direction matters
    for citation honesty: clause 5.3.5.1 genuinely lives inside 5.3.5, so a
    chunk labelled "5.3.5" that contains 5.3.5.1's text is still a TRUE
    citation, just a less specific one. Merging a parent's text down into a
    child (or across to a sibling) would produce a citation that is more
    specific than the truth — i.e. a fabricated one.

RULE 5 — Tables and ASN.1 blocks are atomic. Never split.
    Half a table is worse than no table: the column headers end up in one
    chunk and the values in another, and the model confidently pairs the
    wrong ones. Same for ASN.1 — a truncated definition is unparseable.
    If a single table exceeds the max, we emit it oversized on purpose and
    report it, rather than mangling it.

The contextual header
---------------------
Before embedding, each chunk gets a header prepended:

    TS 38.331 v18.10.0 (Rel-18) - NR RRC Protocol Specification
    5 Procedures > 5.3 Connection control > 5.3.5 RRC reconfiguration
    Clause 5.3.5.3: Reception of an RRCReconfiguration by the UE [part 2/3]

    <the actual text>

This is Anthropic's "Contextual Retrieval" idea. Raw clause text often says
"the UE shall apply the received configuration" with no clue which spec,
which procedure, or which release it belongs to. The embedding of that
sentence alone is nearly identical across four different specs. The header
puts the disambiguating words INTO the vector, which is the only place the
retriever can see them.

Note we store `text` and `embed_text` separately: the header helps retrieval,
but we hand the model the clean text plus structured metadata, so it never
has to parse a header to know what it is reading.

Usage
-----
    python -m src.ingest.chunk                      # all parsed specs
    python -m src.ingest.chunk --file 38331-ia0.jsonl
    python -m src.ingest.chunk --sample 20          # eyeball 20 chunks
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (  # noqa: E402
    CHUNKS_DIR,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    PARSED_DIR,
    SPECS,
)
from src.ingest.parse_docx import count_tokens, _ENC  # noqa: E402

SPEC_TITLES = {num: title for num, title in SPECS}

# TR 21.801 gives these words a precise legal meaning. A chunk containing one
# is a requirement, not commentary — useful later for filtering and for the
# "is this answer normative?" badge in the UI.
_NORMATIVE = re.compile(r"\b(shall|shall not|must|should|should not|may|need not)\b", re.I)


# --------------------------------------------------------------------------
# atomic units
# --------------------------------------------------------------------------
@dataclass
class Unit:
    """The smallest thing we are willing to move around. Never split further."""
    kind: str          # "prose" | "table" | "asn1"
    text: str
    tokens: int
    atomic: bool       # True for table/asn1 — Rule 5


def split_table(md: str, limit: int) -> list[str]:
    """
    Split an over-long markdown table at ROW boundaries, repeating the header.

    This is a refinement of Rule 5, not a violation of it. Rule 5 exists to
    stop headers and values getting separated. Cutting between rows and
    copying the header into every piece preserves exactly that pairing, so
    each piece is still a valid, self-describing table.

    It matters because 3GPP "field descriptions" tables are enormous — the
    one in ServingCellConfig is ~7k tokens on its own. Six of those would be
    42k tokens of context for one answer. And they are row-independent by
    construction: each row documents one field, standalone.
    """
    lines = md.split("\n")
    if len(lines) < 3:
        return [md]
    header, rows = lines[:2], lines[2:]
    head_tok = count_tokens("\n".join(header))

    pieces, cur, cur_tok = [], [], head_tok
    for row in rows:
        t = count_tokens(row)
        if cur and cur_tok + t > limit:
            pieces.append("\n".join(header + cur))
            cur, cur_tok = [], head_tok
        cur.append(row)
        cur_tok += t
    if cur:
        pieces.append("\n".join(header + cur))
    return pieces


# A definition starts at column 0 with an identifier and contains "::=".
# Covers both type assignments  (ServingCellConfig ::= SEQUENCE {)
# and value assignments        (maxBandComb  INTEGER ::= 65536).
# Comment lines start with "--", so they can never match — they stay attached
# to the definition above them, which is where they belong.
_ASN1_DEF = re.compile(r"^[A-Za-z][A-Za-z0-9\-]*[^\n]*::=")


def split_asn1(text: str, limit: int) -> list[str]:
    """
    Split an over-long ASN.1 block at TOP-LEVEL DEFINITION boundaries.

    A line like `ServingCellConfig ::= SEQUENCE {` at column 0 starts a new
    definition; everything indented under it belongs to that definition. So
    cutting only at column-0 `::=` lines never truncates a definition — each
    piece stays syntactically complete, which is the property that made ASN.1
    atomic in the first place.
    """
    lines = text.split("\n")
    defs: list[list[str]] = []
    for line in lines:
        if _ASN1_DEF.match(line) or not defs:
            defs.append([line])
        else:
            defs[-1].append(line)

    pieces, cur, cur_tok = [], [], 0
    for d in defs:
        block = "\n".join(d)
        t = count_tokens(block)
        if cur and cur_tok + t > limit:
            pieces.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(block)
        cur_tok += t
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def split_by_lines(text: str, limit: int) -> list[str]:
    """
    Last-resort splitter for an atomic block that its own structural splitter
    could not get under the limit — a single ASN.1 definition thousands of
    tokens long, or one table row with an enormous cell.

    Why bother instead of just letting it through oversized: the embedder
    truncates at EMBED_MAX_LEN. An 13k-token chunk does not fail loudly, it
    gets SILENTLY represented by its first ~2k tokens. Everything after that
    is in the corpus but unreachable — invisible to retrieval while looking
    perfectly indexed. A clean cut we can see beats silent truncation we
    cannot.
    """
    out, cur, tok = [], [], 0
    for line in text.split("\n"):
        t = count_tokens(line)
        if cur and tok + t > limit:
            out.append("\n".join(cur))
            cur, tok = [], 0
        if t > limit:                       # one monstrous line
            ids = _ENC.encode(line, disallowed_special=())
            for i in range(0, len(ids), limit):
                out.append(_ENC.decode(ids[i:i + limit]))
            continue
        cur.append(line)
        tok += t
    if cur:
        out.append("\n".join(cur))
    return out


def blocks_to_units(blocks: list[dict], stats: dict) -> list[Unit]:
    """
    Flatten a clause's blocks into atomic units.

    Prose blocks were joined with "\\n" by the parser (one line per source
    paragraph), so splitting on "\\n" recovers the original paragraphs — which
    is exactly the boundary we want to split at. Tables and ASN.1 stay whole
    unless they exceed the max, in which case they are cut at their own safe
    internal boundaries (rows / definitions), and only if that still isn't
    enough, at line boundaries.
    """
    units: list[Unit] = []
    for b in blocks:
        kind, text = b["type"], b["text"]

        if kind in ("table", "asn1"):
            tok = count_tokens(text)
            if tok <= CHUNK_MAX_TOKENS:
                units.append(Unit(kind, text, tok, atomic=True))
                continue
            splitter = split_table if kind == "table" else split_asn1
            for piece in splitter(text, CHUNK_TARGET_TOKENS):
                pt = count_tokens(piece)
                if pt <= CHUNK_MAX_TOKENS:
                    units.append(Unit(kind, piece, pt, atomic=True))
                    continue
                stats["forced_atomic_splits"] += 1
                for sub in split_by_lines(piece, CHUNK_TARGET_TOKENS):
                    units.append(Unit(kind, sub, count_tokens(sub), atomic=True))
        else:
            for para in text.split("\n"):
                para = para.strip()
                if para:
                    units.append(Unit("prose", para, count_tokens(para), atomic=False))
    return units


def hard_split(unit: Unit, limit: int) -> list[Unit]:
    """
    Last resort: a single non-atomic unit that alone exceeds the hard limit.

    Splitting mid-sentence is bad, so we count how often this fires and
    report it. In 3GPP this should be near-zero — a "paragraph" here is one
    numbered bullet line, which is rarely more than a few hundred tokens.
    """
    ids = _ENC.encode(unit.text, disallowed_special=())
    out = []
    for i in range(0, len(ids), limit):
        piece = _ENC.decode(ids[i:i + limit])
        out.append(Unit(unit.kind, piece, count_tokens(piece), atomic=False))
    return out


# --------------------------------------------------------------------------
# packing units into pieces  (Rule 3)
# --------------------------------------------------------------------------
_NEST = re.compile(r"^(\d+)>")
# Lines that CONTINUE the current list rather than starting a new one.
# Getting this set wrong is expensive in both directions: treat a real new
# stem as a continuation and a later line inherits a condition it was never
# under (a fabricated condition — worse than none); treat a continuation as a
# new stem and the list tree resets, orphaning everything below it.
#
# 23.501 / 24.501 lean heavily on lettered and numbered bullets nested inside
# the `N>` tree, so those must count as continuations.
#
# `NOTE\s*\d*` rather than `NOTE\b`: 38.331 contains both "NOTE 1:" and
# "NOTE2:" (no space). `NOTE\b` silently fails on the second — there is no word
# boundary between "E" and "2" — so that one line was read as a new stem and
# reset the whole tree, orphaning the nine nested lines below it.
_CONTINUATION = re.compile(
    r"^("
    r"NOTE\s*\d*\b"                 # NOTE:, NOTE 1:, and NOTE2: (no space)
    r"|Editor's [Nn]ote\b"
    r"|[-–—•*][\s\t]"               # dash / bullet items
    r"|\(?[a-zA-Z][)\.][\s\t]"      # a)  (b)  c.
    r"|\(?\d+[)][\s\t]"             # 1)  (2)
    r"|\(?[ivxIVX]+[)][\s\t]"       # i)  (iv)
    r")"
)


def ancestor_map(flat: list[Unit]) -> list[list[int]]:
    """
    For each unit, the indices of the lines that syntactically enclose it.

    3GPP procedural text is a decision tree written with depth markers:

        1>  if the RRCReconfiguration includes the masterCellGroup:
        2>      perform the cell group configuration procedure;
        3>          release the RLC entity;

    If a split lands such that a chunk contains `3> release the RLC entity`
    without the `1>` and `2>` above it, that chunk states an unconditional
    instruction to release an RLC entity. It is not unconditional — it fires
    only inside two nested `if`s. Retrieve it alone and the model answers
    confidently, wrongly, with a citation that checks out. That is the exact
    failure Rule 1 exists to prevent, one level down.

    So we precompute, for every line, the chain of shallower lines currently
    open above it. Any chunk that includes a line then also includes its
    chain, verbatim. Verbatim matters: we never synthesise connective text,
    because anything we invent here is indistinguishable from spec text to
    every stage downstream.
    """
    anc: list[list[int]] = []
    open_at: dict[int, int] = {}     # depth -> index of the line holding it

    # Depth of the next `N>` line at or after each index, or None if the list
    # is finished. Precomputed in one backward pass so the lookahead below is
    # O(1) per line instead of rescanning the clause each time.
    nxt: list[int | None] = [None] * (len(flat) + 1)
    for i in range(len(flat) - 1, -1, -1):
        m = _NEST.match(flat[i].text)
        nxt[i] = int(m.group(1)) if m else nxt[i + 1]

    for i, u in enumerate(flat):
        m = _NEST.match(u.text)

        if m:
            d = int(m.group(1))
            anc.append([open_at[dd] for dd in range(1, d) if dd in open_at])
            open_at[d] = i
            for deeper in [x for x in open_at if x > d]:
                del open_at[deeper]
            continue

        # Not a depth marker. Two very different cases, and getting them
        # confused is what produced orphans in the first draft:
        #
        #   NOTE 1: ...            <- sits INSIDE the open list
        #   -  is smaller than ... <- a dash sub-bullet, also INSIDE
        #   SFN mod T = FLOOR(gapOffset/10);   <- a formula, also INSIDE
        #   Upon receiving SIB1 the UE shall:  <- a NEW stem, list ends here
        #
        # Tables and ASN.1 are always "inside" too.
        #
        # Matching new stems by their wording is a losing game — the formula
        # lines in 38.331 clause 5.5.2.9 look nothing like a NOTE or a bullet,
        # and any keyword list we invent will miss the next variant. So we
        # decide STRUCTURALLY instead, on a fact the format guarantees: a new
        # list always opens at depth 1. If the next `N>` line after this prose
        # is `1>`, the list genuinely restarted and the old conditions must
        # die. If it is `2>` or deeper, that line is still inside the list this
        # prose interrupted, so the tree has to survive — clearing it here is
        # precisely what stranded those lines with no condition at all.
        #
        # The failure directions are not symmetric, which is why the rule is
        # written this way round. Clearing when we should not have invents
        # nothing but LOSES a condition, turning a conditional `shall` into an
        # unconditional one — the exact hallucination this file exists to
        # prevent. Keeping the tree when we should not have would be the worse
        # error (a fabricated condition), but it cannot happen here: a real new
        # stem is followed by `1>`, and the `if m:` branch above already
        # overwrites depth 1 and drops everything deeper, which erases the old
        # tree just as thoroughly as clearing it.
        if u.kind == "prose" and not _CONTINUATION.match(u.text):
            following = nxt[i + 1]
            # None => no nested line follows, so nothing can be orphaned either
            # way; close the list, as the old code did.
            if following is None or following == 1:
                open_at.clear()
                anc.append([])
                continue

        anc.append([open_at[dd] for dd in sorted(open_at)])
    return anc


def pack(units: list[Unit], stats: dict) -> list[list[Unit]]:
    """
    Greedy pack into pieces of ~CHUNK_TARGET_TOKENS, then re-attach context.

    "Greedy" is right here rather than clever: clause text is sequential
    argument, so keeping source order and cutting at the latest legal
    boundary preserves more meaning than any reordering scheme.
    """
    flat: list[Unit] = []
    for u in units:
        if not u.atomic and u.tokens > CHUNK_MAX_TOKENS:
            stats["hard_splits"] += 1
            flat.extend(hard_split(u, CHUNK_TARGET_TOKENS))
        else:
            flat.append(u)

    anc = ancestor_map(flat)

    # pass 1 — decide the cut points
    ranges: list[tuple[int, int]] = []
    start, tok = 0, 0
    for i, u in enumerate(flat):
        if i > start and tok + u.tokens > CHUNK_TARGET_TOKENS:
            ranges.append((start, i))
            start, tok = i, 0
        tok += u.tokens
        if u.atomic and u.tokens > CHUNK_MAX_TOKENS:
            stats["oversized_atomic"] += 1
    ranges.append((start, len(flat)))

    # pass 2 — body + overlap tail + every missing enclosing condition
    pieces: list[list[Unit]] = []
    for n, (s, e) in enumerate(ranges):
        if n == 0:
            pieces.append(flat[s:e])
            continue

        # The overlap tail is part of the chunk, so it needs its conditions
        # too — carrying back three trailing lines that happen to sit at
        # depth 3 would re-introduce exactly the orphan we are preventing.
        # So ancestry is resolved over tail AND body together.
        tl = _overlap_len(flat, ranges[n - 1][0], s)
        included = set(range(s - tl, e))
        needed: set[int] = set()
        for i in included:
            needed.update(j for j in anc[i] if j not in included)

        if needed:
            stats["condition_prefixed"] += 1
        prefix = sorted(needed | set(range(s - tl, s)))
        pieces.append([flat[i] for i in prefix] + flat[s:e])

    return pieces


def _overlap_len(flat: list[Unit], prev_start: int, cut: int) -> int:
    """
    How many trailing units of the previous piece to repeat in the next one.

    Only non-atomic (prose) units are carried: duplicating a whole table into
    the next chunk wastes context budget and creates two near-identical
    vectors that both surface for the same query, crowding out genuinely
    different results.
    """
    n, total = 0, 0
    for i in range(cut - 1, prev_start - 1, -1):
        u = flat[i]
        if u.atomic or total + u.tokens > CHUNK_OVERLAP_TOKENS:
            break
        n += 1
        total += u.tokens
    return n


# --------------------------------------------------------------------------
# merge-up  (Rule 4)
# --------------------------------------------------------------------------
@dataclass
class Group:
    """One clause's worth of content, before it is split into final chunks."""
    clause: dict
    units: list[Unit]
    merged_ids: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(u.tokens for u in self.units)


def build_groups(clauses: list[dict], stats: dict) -> list[Group]:
    """
    Walk clauses in document order, merging under-sized ones into the nearest
    ancestor. The ancestor stack mirrors the one in the parser: pop until the
    top has a strictly smaller level, and whatever is left on top is the
    parent.
    """
    groups: list[Group] = []
    stack: list[Group] = []

    for c in clauses:
        units = blocks_to_units(c["blocks"], stats)
        if not units:
            continue

        while stack and stack[-1].clause["level"] >= c["level"]:
            stack.pop()

        tokens = sum(u.tokens for u in units)
        ancestor = stack[-1] if stack else None

        if (
            tokens < CHUNK_MIN_TOKENS
            and ancestor is not None
            and ancestor.tokens + tokens <= CHUNK_MAX_TOKENS
        ):
            # Rule 4: fold up. The ancestor's clause_id stays the citation,
            # which remains true because this clause is inside it.
            ancestor.units.extend(units)
            ancestor.merged_ids.append(c["clause_id"])
            stats["merged_up"] += 1
            continue

        g = Group(clause=c, units=units)
        groups.append(g)
        stack.append(g)

    return groups


# --------------------------------------------------------------------------
# emitting chunks
# --------------------------------------------------------------------------
def context_header(c: dict, part: int, n_parts: int) -> str:
    """The disambiguating preamble that goes into the embedding."""
    title = SPEC_TITLES.get(c["spec"], "")
    lines = [f"{c['spec_id']} v{c['version']} (Rel-{c['release']})" + (f" - {title}" if title else "")]
    if c["breadcrumb"]:
        lines.append(c["breadcrumb"])
    tail = f" [part {part}/{n_parts}]" if n_parts > 1 else ""
    # For un-numbered headings (ASN.1 IE names) the id IS the title — don't
    # print it twice, the repetition would be double-weighted in the embedding.
    name = c["clause_id"] if c["clause_id"] == c["clause_title"] \
        else f"{c['clause_id']}: {c['clause_title']}"
    lines.append(f"Clause {name}{tail}")
    return "\n".join(lines)


def emit(group: Group, stats: dict) -> list[dict]:
    c = group.clause

    if group.tokens <= CHUNK_MAX_TOKENS:
        pieces = [group.units]          # Rule 2
    else:
        pieces = pack(group.units, stats)   # Rule 3
        stats["split_clauses"] += 1

    out = []
    n = len(pieces)
    for i, units in enumerate(pieces, start=1):
        text = "\n\n".join(u.text for u in units)
        header = context_header(c, i, n)
        embed_text = f"{header}\n\n{text}"
        chunk_id = f"{c['clause_uid']}::{i}" if n > 1 else c["clause_uid"]
        out.append({
            "chunk_id": chunk_id,
            "spec": c["spec"],
            "spec_id": c["spec_id"],
            "spec_title": SPEC_TITLES.get(c["spec"], ""),
            "version": c["version"],
            "release": c["release"],
            "clause_id": c["clause_id"],
            "clause_title": c["clause_title"],
            "breadcrumb": c["breadcrumb"],
            "level": c["level"],
            "part": i,
            "n_parts": n,
            "merged_clause_ids": group.merged_ids,
            "block_types": sorted({u.kind for u in units}),
            "has_normative": bool(_NORMATIVE.search(text)),
            # Precomputed so the generation layer renders citations from
            # METADATA rather than letting the model write them. A model
            # cannot fabricate a citation it is not allowed to compose.
            "citation": f"{c['spec_id']} v{c['version']}, clause {c['clause_id']}",
            "text": text,
            "embed_text": embed_text,
            "token_count": count_tokens(text),
            "order": c["order"],
        })
    return out


# --------------------------------------------------------------------------
def chunk_file(path: Path, stats: dict) -> list[dict]:
    clauses = [json.loads(line) for line in path.open(encoding="utf-8")]
    groups = build_groups(clauses, stats)
    chunks: list[dict] = []
    for g in groups:
        chunks.extend(emit(g, stats))
    return chunks


def find_orphans(chunks: list[dict]) -> list[tuple[str, int, str]]:
    """
    The single most important correctness check in this file.

    Walk every chunk and find nested lines whose enclosing depth never
    appears above them inside that same chunk. Each one is a requirement
    that reads as unconditional but isn't — a hallucination waiting to be
    retrieved, with a citation that will pass every downstream check.

    Returns (chunk_id, depth, line) so a failure can be diagnosed instead of
    guessed at. This must be empty. If it isn't, do not build the index.
    """
    bad: list[tuple[str, int, str]] = []
    for c in chunks:
        seen: set[int] = set()
        for line in c["text"].split("\n"):
            s = line.strip()
            m = _NEST.match(s)
            if not m:
                continue
            d = int(m.group(1))
            # No shallower line at all above this one => the condition context
            # is genuinely gone. If SOME shallower line is present but not
            # exactly d-1, the source itself skipped a level (3GPP does this
            # occasionally) and the context we can recover is already there.
            if d > 1 and not any(x < d for x in seen):
                bad.append((c["chunk_id"], d, s[:100]))
            seen.add(d)
    return bad


def find_level_skips(chunks: list[dict]) -> int:
    """Depth jumps that come from the source document, not from our splitting."""
    n = 0
    for c in chunks:
        seen: set[int] = set()
        for line in c["text"].split("\n"):
            m = _NEST.match(line.strip())
            if not m:
                continue
            d = int(m.group(1))
            if d > 1 and (d - 1) not in seen and any(x < d for x in seen):
                n += 1
            seen.add(d)
    return n


def print_sample(chunks: list[dict], n: int) -> None:
    """Evenly spaced, not random — so a re-run shows you the same ones."""
    if not chunks:
        return
    step = max(1, len(chunks) // n)
    print("\n" + "=" * 78)
    print(f"SAMPLE — eyeball these. Every one should be self-contained and citable.")
    print("=" * 78)
    for c in chunks[::step][:n]:
        print(f"\n--- {c['chunk_id']}  ({c['token_count']} tok, "
              f"{'+'.join(c['block_types'])}"
              f"{', normative' if c['has_normative'] else ''}) ---")
        print(c["embed_text"][:900])
        if len(c["embed_text"]) > 900:
            print("  ... [truncated for display]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunk parsed clause trees")
    ap.add_argument("--file", action="append", default=None,
                    help="only this parsed file (repeatable), e.g. --file 38331-ia0.jsonl")
    ap.add_argument("--sample", type=int, default=0,
                    help="print N chunks after writing, for manual inspection")
    args = ap.parse_args()

    files = [PARSED_DIR / f for f in args.file] if args.file else sorted(PARSED_DIR.glob("*.jsonl"))
    if not files:
        print(f"No parsed files in {PARSED_DIR}. Run: python -m src.ingest.parse_docx")
        return 1

    stats = {"merged_up": 0, "split_clauses": 0, "hard_splits": 0,
             "oversized_atomic": 0, "condition_prefixed": 0,
             "forced_atomic_splits": 0}
    all_chunks: list[dict] = []

    for path in files:
        if not path.exists():
            print(f"! missing: {path.name}")
            continue
        chunks = chunk_file(path, stats)
        out = CHUNKS_DIR / path.name
        with out.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        toks = sum(c["token_count"] for c in chunks)
        avg = toks // len(chunks) if chunks else 0
        print(f"{path.stem:<14} -> {len(chunks):>5} chunks, {toks:>9,} tokens, avg {avg}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks produced.")
        return 1

    sizes = sorted(c["token_count"] for c in all_chunks)
    print("\n" + "-" * 60)
    print(f"TOTAL            {len(all_chunks):>5} chunks, "
          f"{sum(sizes):>9,} tokens")
    print(f"size  min {sizes[0]}  p50 {sizes[len(sizes)//2]}  "
          f"p95 {sizes[int(len(sizes)*0.95)]}  max {sizes[-1]}")
    print(f"merged up (Rule 4)      {stats['merged_up']}")
    print(f"clauses split (Rule 3)  {stats['split_clauses']}")
    print(f"condition-prefixed      {stats['condition_prefixed']}  "
          f"(pieces that regained their enclosing 1>/2> conditions)")
    print(f"oversized tables/ASN.1  {stats['oversized_atomic']}  "
          f"(kept whole on purpose - Rule 5)")
    if stats["forced_atomic_splits"]:
        print(f"forced line-splits      {stats['forced_atomic_splits']}  "
              f"(giant table row / ASN.1 definition - would otherwise be "
              f"silently truncated at embed time)")
    if stats["hard_splits"]:
        print(f"! mid-paragraph splits  {stats['hard_splits']}  "
              f"(should be ~0 - tell me if this is large)")

    print("\nchecks")
    ok = True

    ids = [c["chunk_id"] for c in all_chunks]
    if len(ids) != len(set(ids)):
        print(f"  FAIL  duplicate chunk_ids: {len(ids) - len(set(ids))}")
        ok = False
    else:
        print("  ok    chunk_id unique")

    orphans = find_orphans(all_chunks)
    if orphans:
        print(f"  FAIL  {len(orphans)} nested lines lost their enclosing condition")
        by_chunk: dict[str, list] = {}
        for cid, d, line in orphans:
            by_chunk.setdefault(cid, []).append((d, line))
        for cid, items in list(by_chunk.items())[:12]:
            print(f"        {cid}")
            for d, line in items[:3]:
                print(f"          depth {d}: {line}")
        ok = False
    else:
        print("  ok    every 'shall' still carries its condition")

    skips = find_level_skips(all_chunks)
    if skips:
        print(f"  note  {skips} depth jumps inherited from the source text "
              f"(not caused by chunking)")

    over = [c for c in all_chunks if c["token_count"] > CHUNK_MAX_TOKENS]
    if over:
        worst = max(over, key=lambda c: c["token_count"])
        print(f"  warn  {len(over)} chunks over {CHUNK_MAX_TOKENS} tokens "
              f"(worst {worst['token_count']}: {worst['chunk_id']})")
    else:
        print(f"  ok    no chunk exceeds {CHUNK_MAX_TOKENS} tokens")

    empty = sum(1 for c in all_chunks if not c["text"].strip())
    print(f"  {'ok   ' if not empty else 'FAIL '} {empty} empty chunks")
    ok = ok and not empty

    if not ok:
        print("\nDo NOT build the index on this output. Send me the failures.")
        return 1

    if args.sample:
        print_sample(all_chunks, args.sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
