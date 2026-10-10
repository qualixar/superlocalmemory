# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What became of the answer check on one recall, and how long it may take.

The answer check is a SIGNAL about the results, never part of retrieving them.
Every rule in this module follows from that:

* Skipping the check costs a verdict (``answer_confidence`` / ``abstained``),
  never a result. The memories, their order from retrieval and ranking, and
  their scores are identical whether the check ran or not.
* So the check may be skipped to keep a recall inside its ceiling, and when it
  is, the response says so (``answer_check_status``) instead of looking like a
  recall nobody judged on purpose. Two runs of one question that disagree on
  the verdict can then be told apart from two runs that disagree on results.

The ceiling is the owner's recall budget: 3.0 s, retrieval and answer check
together (raised from 2.0 s in 4.1.19 so the check gets room to run). It is a
ceiling, not a target, and nothing here makes retrieval itself faster or shorter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# -- what happened, on every response ------------------------------------------

#: A verdict was produced and applied to this recall.
STATUS_JUDGED = "judged"
#: No answer check runs for this engine (the configured state, not a failure).
STATUS_OFF = "off"
#: A check is on but was deliberately not asked for this recall: a system recall
#: (daemon warm-up), nothing to judge, a shown memory that may not leave the
#: machine, or too little of the recall's time budget left.
STATUS_SKIPPED = "skipped"
#: The on-device check was answering another recall and did not free up within
#: this recall's budget.
STATUS_BUSY = "busy"
#: The on-device check is still loading its model. Loading takes tens of seconds,
#: far beyond any recall budget, so a recall never waits for it.
STATUS_WARMING = "warming"
#: The check was on and could not answer: it timed out, failed, returned
#: something malformed, has no key, or cannot run in this process.
STATUS_UNAVAILABLE = "unavailable"

ANSWER_CHECK_STATUSES = frozenset({
    STATUS_JUDGED, STATUS_OFF, STATUS_SKIPPED, STATUS_BUSY, STATUS_WARMING,
    STATUS_UNAVAILABLE,
})

# -- why a recall was skipped (4.1.20) -----------------------------------------------
#: A closed set, never a free-form string: the Answer Check history stores it.
DETAIL_NONE = ""
#: A recall that is not a question (``skip_answer_check()``) or background work.
DETAIL_NOT_A_QUESTION = "not_a_question"
DETAIL_NO_RESULTS = "no_results"
#: The online check would have read a memory another profile owns.
DETAIL_OTHER_PROFILE = "other_profile_memory"
#: Too little of the recall's time budget was left to ask.
DETAIL_BUDGET = "budget"
#: Judged: the verdict is the one the same judge gave a moment ago for the same
#: question over the same memories (``core.answer_check_memo``), not a new ask.
DETAIL_REUSED = "reused"
#: Every result is a picture with no words of its own: a text check has nothing to read.
DETAIL_MEDIA_UNJUDGED = "media_unjudged"

ANSWER_CHECK_DETAILS = frozenset({
    DETAIL_NONE, DETAIL_NOT_A_QUESTION, DETAIL_NO_RESULTS, DETAIL_OTHER_PROFILE,
    DETAIL_BUDGET, DETAIL_REUSED, DETAIL_MEDIA_UNJUDGED,
})

#: One plain sentence per way the check can end without a verdict. A recall
#: that was not checked must never read like one that was: ``abstained`` is
#: False on both, so the note (and ``answer_check_ran``) is what tells them apart.
_NOT_CHECKED = "Answer check did not run"
_SKIP_NOTES = {
    DETAIL_BUDGET: (f"{_NOT_CHECKED}: retrieval used the recall's time budget, so "
                    "there was no time left to ask. The results are complete; only "
                    "the verdict is missing."),
    DETAIL_NOT_A_QUESTION: f"{_NOT_CHECKED}: this recall loads context, it is not a question.",
    DETAIL_NO_RESULTS: f"{_NOT_CHECKED}: nothing was found to check.",
    DETAIL_OTHER_PROFILE: (f"{_NOT_CHECKED}: the online check would have read a memory "
                           "another profile owns."),
    DETAIL_MEDIA_UNJUDGED: (f"{_NOT_CHECKED}: the results are pictures with no words, "
                            "so there was no text to check."),
}

# -- what a caller asks for, per recall ------------------------------------------

#: Every recall a person or an agent asks for: the check, and the reorder too
#: when it is switched on.
REQUEST_FULL = "full"
#: The check only, never the reorder. The bounded-loop gate needs the verdict to
#: refuse an insufficient match, but reordering three throwaway results buys
#: nothing and adds a second calibration to a decision that needs one.
REQUEST_NO_REORDER = "no_reorder"
#: No check at all — recalls that are not questions (the daemon warm-up,
#: context loading) — is NOT a request value: it is the one shared marker,
#: ``core.answer_check_scope.skip_answer_check()``, so there is one mechanism.

ANSWER_CHECK_REQUESTS = frozenset({REQUEST_FULL, REQUEST_NO_REORDER})

# -- the time budget ---------------------------------------------------------------

#: The recall ceiling (the owner's quality rule). Measured from the start of the
#: recall pipeline.
RECALL_CEILING_S = 3.0
#: Kept back for what runs after the check (score contract, markers, working
#: memory): in-process work of well under a millisecond, with headroom.
POST_JUDGE_RESERVE_S = 0.05
#: Below this, the check is not asked. A hosted round trip or a local judgement
#: of three memories rarely completes in a quarter of a second, so asking would
#: add that wait to the recall for a verdict that would almost always time out.
#: COST: a recall whose retrieval already used more than ~2.7 s is reported
#: unjudged (status "skipped") — its results are unchanged.
JUDGE_FLOOR_S = 0.25


@dataclass(frozen=True)
class JudgeOutcome:
    """A verdict (or None) and the status that explains it."""

    verdict: Any = None
    status: str = STATUS_UNAVAILABLE
    #: 4.1.20: why a skipped recall was skipped (``DETAIL_*``). ``compare=False``
    #: so two outcomes that differ only in their explanation still compare equal.
    detail: str = field(default=DETAIL_NONE, compare=False)


@dataclass(frozen=True, slots=True)
class AnswerCheckTrace:
    """Timing and provenance of one recall's answer check (4.1.20).

    In-process only: it is never put on an MCP or HTTP recall envelope. All
    times are milliseconds measured from the start of the recall pipeline —
    the same clock ``RECALL_CEILING_S`` bounds.
    """

    detail: str              # one of ANSWER_CHECK_DETAILS
    backend: str             # "laya" | "jev" | ""
    threshold: float | None
    reordered: bool          # the hosted check reordered the results
    retrieval_ms: float      # pipeline start -> just before the check
    judge_ms: float          # the check's own wall time (0 when not asked)
    total_ms: float          # pipeline start -> the response is returned


def answer_check_note(status: object, detail: object = DETAIL_NONE) -> str:
    """The sentence a person reads when the check gave no verdict; "" otherwise.

    "" for ``judged`` (the verdict speaks) and ``off`` (nothing is configured,
    so nothing is missing).
    """
    if status == STATUS_SKIPPED:
        return _SKIP_NOTES.get(detail, f"{_NOT_CHECKED}.")  # type: ignore[arg-type]
    if status == STATUS_BUSY:
        return (f"{_NOT_CHECKED}: the on-device check was answering another recall "
                "and did not free up in time.")
    if status == STATUS_WARMING:
        return f"{_NOT_CHECKED}: the on-device check is still loading its model."
    if status == STATUS_UNAVAILABLE:
        return f"{_NOT_CHECKED}: the check did not answer in time or could not run."
    return ""


def judge_deadline(recall_started: float, *, now: float | None = None) -> float | None:
    """The monotonic instant the check must answer by, or None to skip it.

    ``recall_started`` is ``time.monotonic()`` at the start of the recall.
    """
    current = time.monotonic() if now is None else now
    deadline = recall_started + RECALL_CEILING_S - POST_JUDGE_RESERVE_S
    if deadline - current < JUDGE_FLOOR_S:
        return None
    return deadline


def effective_deadline(deadline: float | None, timeout_s: float) -> float:
    """The earlier of the recall's deadline and the backend's own timeout."""
    own = time.monotonic() + max(0.0, float(timeout_s))
    return own if deadline is None else min(deadline, own)


def seconds_left(deadline: float) -> float:
    return deadline - time.monotonic()


def normalize_request(value: object) -> str:
    """A caller's per-recall request, validated. Raises ValueError on garbage:
    this is an internal API, and a typo must not silently run the full check."""
    if value is None or value == "":
        return REQUEST_FULL
    if isinstance(value, str) and value in ANSWER_CHECK_REQUESTS:
        return value
    raise ValueError(
        f"answer_check must be one of {sorted(ANSWER_CHECK_REQUESTS)}, got {value!r}"
    )


__all__ = [
    "ANSWER_CHECK_DETAILS",
    "ANSWER_CHECK_REQUESTS",
    "ANSWER_CHECK_STATUSES",
    "AnswerCheckTrace",
    "DETAIL_BUDGET",
    "DETAIL_MEDIA_UNJUDGED",
    "DETAIL_NONE",
    "DETAIL_NOT_A_QUESTION",
    "DETAIL_NO_RESULTS",
    "DETAIL_OTHER_PROFILE",
    "DETAIL_REUSED",
    "JUDGE_FLOOR_S",
    "JudgeOutcome",
    "POST_JUDGE_RESERVE_S",
    "RECALL_CEILING_S",
    "REQUEST_FULL",
    "REQUEST_NO_REORDER",
    "STATUS_BUSY",
    "STATUS_JUDGED",
    "STATUS_OFF",
    "STATUS_SKIPPED",
    "STATUS_UNAVAILABLE",
    "STATUS_WARMING",
    "answer_check_note",
    "effective_deadline",
    "judge_deadline",
    "normalize_request",
    "seconds_left",
]
