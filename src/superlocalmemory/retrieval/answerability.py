# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Whether a recall's answer was CHECKED, in one word every surface shares.

WHY
---
``abstained`` is False both on a recall the answer check judged sufficient and
on a recall nobody checked at all (the check was off, still loading, busy,
unavailable, or out of time). A caller -- an agent, a hook, a bounded-loop gate
-- reading ``abstained=false`` as "the memory answers this" was wrong on every
unchecked recall. Seen on a real store: a question about something never stored
came back with candidates, the check skipped for lack of time, and
``abstained=false``. That is not a judged false acceptance; it is an unjudged
answer whose old field reads like a verified one.

THE FIELDS
----------
``answerability`` is exactly one of:

* ``supported``   -- the check ran and judged the shown memories sufficient;
* ``unsupported`` -- the check ran and judged them insufficient;
* ``unjudged``    -- the check did not produce a verdict for this recall.

``answerability_reason`` says why, from a closed set:
``judged_fresh`` / ``judged_from_memo`` (a supported or unsupported verdict,
asked now or reused from an identical earlier check), or, for ``unjudged``:
``disabled``, ``warming``, ``unavailable``, ``busy``, ``budget_exhausted``,
``no_results``, ``not_a_question``, ``other_profile``.

An empty result set is ``unjudged`` / ``no_results``: nothing was found to
check, which is not evidence that no answer exists (a channel may have been
down -- see ``channel_status``). ``abstained`` keeps its old meaning for old
callers, and the old ``answer_check_*`` fields are unchanged.

Pure function of fields every response already carries, so the daemon, the
MCP tools (including the fallback for an older daemon), the CLI, hooks, the
dashboard and the bounded-loop gate all derive the same word the same way.
"""

from __future__ import annotations

from typing import Any, Mapping

from superlocalmemory.retrieval import answer_check_status as acs

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNJUDGED = "unjudged"
ANSWERABILITY = frozenset({SUPPORTED, UNSUPPORTED, UNJUDGED})

REASON_JUDGED_FRESH = "judged_fresh"
REASON_JUDGED_FROM_MEMO = "judged_from_memo"
REASON_DISABLED = "disabled"
REASON_WARMING = "warming"
REASON_UNAVAILABLE = "unavailable"
REASON_BUSY = "busy"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_NO_RESULTS = "no_results"
REASON_NOT_A_QUESTION = "not_a_question"
REASON_OTHER_PROFILE = "other_profile"
REASON_MEDIA_UNJUDGED = "media_unjudged"

REASONS = frozenset({
    REASON_JUDGED_FRESH, REASON_JUDGED_FROM_MEMO, REASON_DISABLED, REASON_WARMING,
    REASON_UNAVAILABLE, REASON_BUSY, REASON_BUDGET_EXHAUSTED, REASON_NO_RESULTS,
    REASON_NOT_A_QUESTION, REASON_OTHER_PROFILE, REASON_MEDIA_UNJUDGED,
})

_UNJUDGED_BY_STATUS = {
    acs.STATUS_OFF: REASON_DISABLED,
    acs.STATUS_WARMING: REASON_WARMING,
    acs.STATUS_BUSY: REASON_BUSY,
    acs.STATUS_UNAVAILABLE: REASON_UNAVAILABLE,
}
_UNJUDGED_BY_SKIP = {
    acs.DETAIL_BUDGET: REASON_BUDGET_EXHAUSTED,
    acs.DETAIL_NO_RESULTS: REASON_NO_RESULTS,
    acs.DETAIL_NOT_A_QUESTION: REASON_NOT_A_QUESTION,
    acs.DETAIL_OTHER_PROFILE: REASON_OTHER_PROFILE,
    acs.DETAIL_MEDIA_UNJUDGED: REASON_MEDIA_UNJUDGED,
}


def answerability(status: object, detail: object, *, abstained: object,
                  result_count: int) -> tuple[str, str]:
    """``(answerability, reason)`` for one recall. Never raises.

    A ``judged`` status counts only with results to judge; anything not
    recognised is ``unjudged`` -- the safe reading, since only a verdict can
    make an answer ``supported``.
    """
    if result_count <= 0:
        return UNJUDGED, REASON_NO_RESULTS
    if status == acs.STATUS_JUDGED:
        reason = REASON_JUDGED_FROM_MEMO if detail == acs.DETAIL_REUSED else REASON_JUDGED_FRESH
        return (UNSUPPORTED if abstained is True else SUPPORTED), reason
    if status == acs.STATUS_SKIPPED:
        return UNJUDGED, _UNJUDGED_BY_SKIP.get(detail, REASON_UNAVAILABLE)  # type: ignore[arg-type]
    return UNJUDGED, _UNJUDGED_BY_STATUS.get(status, REASON_UNAVAILABLE)  # type: ignore[arg-type]


def of_response(response: Any) -> tuple[str, str]:
    """``answerability`` for a RecallResponse-like object."""
    status = getattr(response, "answer_check_status", None)
    if status is None:  # an object from before the status existed: same tell
        calibrated = getattr(response, "calibration_status", None) not in (
            None, "", "uncalibrated")
        status = acs.STATUS_JUDGED if calibrated else acs.STATUS_SKIPPED
    return answerability(
        status, getattr(response, "answer_check_detail", acs.DETAIL_NONE),
        abstained=bool(getattr(response, "abstained", False)),
        result_count=len(getattr(response, "results", None) or ()),
    )


def of_envelope(envelope: Mapping[str, Any]) -> tuple[str, str]:
    """``answerability`` for a recall envelope (dict). Trusts a well-formed
    value the daemon already put there; derives it for an older daemon."""
    given, why = envelope.get("answerability"), envelope.get("answerability_reason")
    if given in ANSWERABILITY and why in REASONS:
        return given, why  # type: ignore[return-value]
    results = envelope.get("results")
    count = envelope.get("result_count")
    if not isinstance(count, int):
        # Metadata without its results (a summary): emptiness is unknown, so
        # it is not invented -- only the check's own status decides.
        count = len(results) if isinstance(results, list) else 1
    status = envelope.get("answer_check_status")
    if status is None:
        # A daemon older than 4.1.18 sends no check status. Its score contract
        # is calibrated only when a verdict was applied, so that is the tell.
        calibrated = envelope.get("calibration_status") not in (None, "", "uncalibrated")
        status = acs.STATUS_JUDGED if calibrated else acs.STATUS_SKIPPED
    return answerability(
        status, envelope.get("answer_check_reason", acs.DETAIL_NONE),
        abstained=envelope.get("abstained") is True, result_count=count,
    )


def is_supported(value: Any) -> bool:
    """True only for a checked, sufficient answer (a gate's pass condition)."""
    if isinstance(value, Mapping):
        return of_envelope(value)[0] == SUPPORTED
    return of_response(value)[0] == SUPPORTED


__all__ = [
    "ANSWERABILITY", "REASONS", "SUPPORTED", "UNJUDGED", "UNSUPPORTED",
    "answerability", "is_supported", "of_envelope", "of_response",
]
