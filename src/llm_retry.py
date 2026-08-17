"""
Retry wrapper for Gemini calls.

Why this is not optional
------------------------
The hosted model is the only part of this system we do not run ourselves, and
it fails in ways nothing else here does: 503 UNAVAILABLE ("high demand"), 429
rate limits, and the occasional 500. These are transient and uncorrelated with
our input — the same call succeeds seconds later.

Without a retry, a single one of them anywhere in an eval run kills the whole
run, and there are hundreds of calls in one: 77 questions x 3 arms, plus a
judge call per claim. Losing 20 minutes of work to a blip that resolves itself
in two seconds is not an acceptable failure mode, and neither is a live query
returning a stack trace to a user because a datacentre was busy.

What is deliberately NOT retried
--------------------------------
Only transport-level failures. A 400 (bad request), 403 (bad key) or 404
(model retired — see the note on GEN_MODEL in config.py) is a bug in our code
or config, and retrying it just turns a fast, clear error into a slow, clear
error. We check the status code and re-raise those immediately.

The caller decides what failure means
-------------------------------------
This helper raises after the last attempt rather than returning a sentinel.
Both call sites treat a dead LLM as a REFUSAL, which is the only safe reading:
an answer we could not generate and a claim we could not verify are both
"not established", and this system's entire posture is that unestablished
means refuse. Swallowing the error here and returning something neutral would
let that decision drift into a utility module where nobody would look for it.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from src.config import LLM_BACKOFF_BASE, LLM_MAX_ATTEMPTS

T = TypeVar("T")

# Transient by nature: the request was fine, the service was not.
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _status_of(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    # google-genai puts the code in the message when it is not an attribute.
    head = str(exc)[:4].strip()
    return int(head) if head.isdigit() else None


def with_retry(fn: Callable[[], T], what: str = "LLM call") -> T:
    """
    Call `fn`, retrying transient API failures with exponential backoff.

    Jitter is not decoration: an eval run issues these calls in a tight loop,
    so a fixed backoff would line every retry up on the same schedule and
    hammer the service in synchronised waves at exactly the moment it is
    already telling us it is overloaded.
    """
    last: Exception | None = None

    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:                                   # noqa: BLE001
            status = _status_of(exc)
            if status is not None and status not in _RETRY_STATUS:
                raise
            last = exc
            if attempt == LLM_MAX_ATTEMPTS - 1:
                break
            delay = LLM_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"    ! {what} failed ({status or type(exc).__name__}); "
                  f"retry {attempt + 1}/{LLM_MAX_ATTEMPTS - 1} in {delay:.1f}s")
            time.sleep(delay)

    raise last if last else RuntimeError(f"{what} failed with no exception")
