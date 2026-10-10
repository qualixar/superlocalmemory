# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The answer-check stage of one recall: asked at most once, inside the recall's
time budget, and always reported.

The check is a signal about the results, never part of retrieving them, so
every way this stage can end — judged, skipped, busy, warming, unavailable,
off — leaves the results exactly as retrieval and ranking produced them, apart
from the one opt-in exception: the hosted check's reorder, which the person
switched on and which this stage applies before anything downstream reads the
order.

Who may ask (``retrieval.answer_check_status`` has the full rules):

* Recalls that are not questions run inside ``skip_answer_check()``
  (``core.answer_check_scope``): the daemon warm-up, context loading, the
  per-prompt hook. Nothing is judged, sent or billed. Best-effort background
  work (health probe, materialiser) is skipped too, as before.
* ``REQUEST_NO_REORDER`` — the bounded-loop gate: it needs the verdict, not
  the reorder.
* ``REQUEST_FULL`` — every other recall a person or an agent asked for.

The judge is read off the engine once: a switch may replace it at any moment,
and one read means this recall uses one judge from start to end.
"""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.core import answer_check_memo
from superlocalmemory.core.answer_check_scope import answer_check_skipped
from superlocalmemory.retrieval import media_rerank
from superlocalmemory.retrieval.answer_check_status import (
    ANSWER_CHECK_DETAILS,
    ANSWER_CHECK_STATUSES,
    DETAIL_BUDGET,
    DETAIL_MEDIA_UNJUDGED,
    DETAIL_NONE,
    DETAIL_NO_RESULTS,
    DETAIL_NOT_A_QUESTION,
    DETAIL_OTHER_PROFILE,
    DETAIL_REUSED,
    REQUEST_FULL,
    STATUS_JUDGED,
    STATUS_OFF,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    AnswerCheckTrace,
    JudgeOutcome,
    judge_deadline,
)

logger = logging.getLogger(__name__)

_JUDGE_ATTR = "_sufficiency_judge"


def run_answer_check(retrieval_engine: Any, query: str, response: Any, *,
                     request: str = REQUEST_FULL,
                     recall_started: float | None = None,
                     profile_id: str | None = None) -> JudgeOutcome:
    """Ask the engine's answer check about ``response``, once, and say what happened.

    ``recall_started`` is ``time.monotonic()`` at the start of the recall. When
    given, the check must answer inside what is left of the recall ceiling, and
    is not asked at all when less than the floor is left. Skipping it costs the
    verdict, never a result. Without it (direct callers), each backend's own
    timeout applies, as before.

    ``profile_id`` is the profile the recall runs for. The online check is not
    asked when what it would read includes another profile's memory (a shared
    or global result): consent to send is this install's, but in a team that
    memory may be someone else's.
    """
    from superlocalmemory.core.recall_gate import is_background_work

    judge = getattr(retrieval_engine, _JUDGE_ATTR, None)
    if judge is None:
        return JudgeOutcome(None, STATUS_OFF)
    if answer_check_skipped() or is_background_work():
        return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_NOT_A_QUESTION)
    if not response.results:
        return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_NO_RESULTS)
    if media_rerank.all_unreadable(response.results):
        return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_MEDIA_UNJUDGED)
    if _would_send_another_profiles_memory(judge, response, profile_id, request):
        logger.debug("Answer check skipped: the online check would read another "
                     "profile's memory")
        return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_OTHER_PROFILE)
    if request == REQUEST_FULL and reorders(judge):
        deadline = _deadline_or_skip(recall_started)
        if deadline is _SKIP:
            # 4.1.22, the reorder branch's budget recovery, explicitly: the
            # hosted check is never memoised (it re-reads consent and its key
            # before every request) and never finished later (that would bill
            # for a verdict nobody waited for, and a late order could reorder
            # a list the caller already read). So the recall returns exactly as
            # retrieval ranked it, nothing is sent, nothing is queued, and the
            # response says unjudged / budget_exhausted.
            return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_BUDGET)
        return rerank_and_judge(judge, query, response, deadline)
    # 4.1.20: the same question over the same memories gets the same verdict,
    # whatever retrieval cost this time (core.answer_check_memo; on-device only).
    documents = _top_documents(judge, response)
    binding, changed = _binding(retrieval_engine, response, profile_id, len(documents))
    remembered = answer_check_memo.lookup(judge, query, documents,
                                          binding=binding, changed=changed)
    if remembered is not None:
        return JudgeOutcome(remembered, STATUS_JUDGED, DETAIL_REUSED)
    deadline = _deadline_or_skip(recall_started)
    if deadline is _SKIP:
        # Results are unchanged; finish the check off the recall's clock so the
        # next run of this question is judged instead of skipped again.
        answer_check_memo.finish_later(judge, query, documents, binding=binding)
        return JudgeOutcome(None, STATUS_SKIPPED, DETAIL_BUDGET)
    # A live recall comes first: a check being finished later yields the worker
    # to it, and if that check was this very question, its verdict is reused
    # the moment this recall gets the worker instead of being asked again.
    with answer_check_memo.live_check(judge):
        outcome = _plain_check(judge, query, documents, deadline,
                               reuse=lambda: answer_check_memo.lookup(
                                   judge, query, documents, binding=binding, changed=changed))
    if outcome.status == STATUS_JUDGED and outcome.detail != DETAIL_REUSED:
        answer_check_memo.store(judge, query, documents, outcome.verdict, binding=binding)
    return outcome


def _binding(retrieval_engine: Any, response: Any, profile_id: str | None,
             count: int) -> tuple[Any, Any]:
    """What a remembered verdict is bound to, and how to tell it went stale.

    Never raises: without a readable change log the binding still carries the
    profile, ids and kinds, and ``changed`` answers "cannot say" (True), so a
    remembered verdict is not reused on a store that cannot prove it current.
    """
    from superlocalmemory.storage import fact_search_changes as changes

    facts = [getattr(r, "fact", None) for r in response.results[:count]]
    ids = tuple(str(getattr(f, "fact_id", "") or "") for f in facts)
    kinds = tuple(str(getattr(f, "memory_kind", "") or "") for f in facts)
    db = getattr(retrieval_engine, "_db", None)
    try:
        seq = changes.log_bounds(db)[0] if db is not None else 0
    except Exception:  # noqa: BLE001
        seq = -1
    binding = answer_check_memo.Binding(str(profile_id or ""), ids, kinds, seq)

    if db is None:  # a stand-in engine with no store: nothing to validate against
        return binding, None

    def changed(since: int, fact_ids: tuple[str, ...]) -> bool:
        if since < 0:
            return True
        head, oldest = changes.log_bounds(db)
        if head < since or (oldest is not None and oldest > since + 1 and head > since):
            return True  # restored store, or the log no longer reaches back
        return bool(changes.changed_fact_ids_among(db, since, fact_ids))

    return binding, changed


_SKIP = object()


def _deadline_or_skip(recall_started: float | None) -> Any:
    """The check's deadline, None for "the backend's own timeout", or ``_SKIP``."""
    if recall_started is None:
        return None
    deadline = judge_deadline(recall_started)
    if deadline is None:
        logger.debug("Answer check skipped: too little of the recall budget left")
        return _SKIP
    return deadline


def _top_documents(judge: Any, response: Any) -> list[Any]:
    """What the plain check reads: the top ``top_k`` memories' text. Never raises."""
    from superlocalmemory.retrieval.judge_recipe import JudgeDocument, document_from_fact
    from superlocalmemory.retrieval.sufficiency import DEFAULT_TOP_K

    top_k = getattr(judge, "top_k", DEFAULT_TOP_K)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        top_k = DEFAULT_TOP_K
    docs = [document_from_fact(getattr(r, "fact", None)) for r in response.results[:top_k]]
    # Saving labels are not part of what the memory says.
    return [d if "[" not in d.content else JudgeDocument(content=media_rerank.strip_labels(d.content))
            for d in docs]


def _would_send_another_profiles_memory(judge: Any, response: Any,
                                       profile_id: str | None, request: str) -> bool:
    """Whether the online check would read a memory another profile owns.

    Only the hosted backend sends anything; only the memories it would read
    count — the top three, or the top ``rerank_k`` when it reorders.
    """
    if profile_id is None or getattr(judge, "backend", None) != "jev":
        return False
    if request == REQUEST_FULL and reorders(judge):
        span = getattr(judge, "rerank_k", 0)
    else:
        span = getattr(judge, "top_k", 3)
    if not isinstance(span, int) or isinstance(span, bool) or span < 1:
        span = len(response.results)
    return any(getattr(getattr(r, "fact", None), "profile_id", profile_id) != profile_id
               for r in response.results[:span])


def reorders(judge: Any) -> bool:
    """Only the hosted check, and only when it was built to reorder.

    ``is True`` and the backend check are deliberate: a test double, or a
    local judge that happens to grow the same attribute, never reorders.
    """
    return (getattr(judge, "backend", None) == "jev"
            and getattr(judge, "rerank_enabled", False) is True
            and callable(getattr(judge, "rerank_and_judge", None)))


def genuine_verdict(verdict: Any) -> Any:
    """``verdict`` if it is a real SufficiencyVerdict, else None."""
    from superlocalmemory.retrieval.sufficiency import SufficiencyVerdict

    return verdict if isinstance(verdict, SufficiencyVerdict) else None


_genuine = genuine_verdict


def _settled(verdict: Any, status: Any) -> JudgeOutcome:
    """Only a genuine verdict counts, and only a known status is reported.

    A judge that misbehaves can never feed the contract something it did not
    decide, nor put an unknown word on the wire.
    """
    verdict = _genuine(verdict)
    if verdict is not None:
        return JudgeOutcome(verdict, STATUS_JUDGED)
    if status not in ANSWER_CHECK_STATUSES or status == STATUS_JUDGED:
        status = STATUS_UNAVAILABLE
    return JudgeOutcome(None, status)


def _plain_check(judge: Any, query: str, documents: list[Any],
                 deadline: float | None, *, reuse: Any = None) -> JudgeOutcome:
    """``reuse`` reaches only a judge that declares ``reuses_after_lock``: it is
    asked again once the judge has its worker, before anything is sent."""
    try:
        assess = getattr(judge, "assess", None)
        if callable(assess):
            if reuse is not None and getattr(judge, "reuses_after_lock", False) is True:
                outcome = assess(query, documents, deadline=deadline, reuse=reuse)
            else:
                outcome = assess(query, documents, deadline=deadline)
            if not isinstance(outcome, JudgeOutcome):
                return JudgeOutcome(None, STATUS_UNAVAILABLE)
            settled = _settled(outcome.verdict, outcome.status)
            if settled.verdict is not None and outcome.detail == DETAIL_REUSED:
                return JudgeOutcome(settled.verdict, STATUS_JUDGED, DETAIL_REUSED)
            return settled
        return _settled(judge.judge(query, documents), STATUS_UNAVAILABLE)
    except Exception as exc:  # noqa: BLE001 — a judge never breaks a recall
        # The type only: a message could quote the memory text it was judging.
        logger.warning("Sufficiency judge failed; reporting the recall unjudged (%s)",
                       type(exc).__name__)
        return JudgeOutcome(None, STATUS_UNAVAILABLE)


def rerank_and_judge(judge: Any, query: str, response: Any,
                     deadline: float | None) -> JudgeOutcome:
    """Let the hosted check reorder the top results, then judge the new top three.

    One request (``JevSufficiencyJudge.rerank_and_judge``). The new order is
    applied here, before the score contract assigns ``rank_position`` and
    before markers and working memory see the list, so everything downstream
    describes the order the caller receives. Every failure leaves the results
    exactly as they were — the same objects, in the same order.

    Precedence: this runs after every other ordering pass, including the
    exact-lexical guard. Someone who turned it on asked Jev, which reads each
    memory, to choose the order of the top ``rerank_k``; a memory containing
    the question's words is one of the memories it reads.
    """
    from superlocalmemory.retrieval.jev_rerank import STATUS_LISTWISE, RerankVerdict
    from superlocalmemory.retrieval.judge_recipe import document_from_fact

    rerank_k = getattr(judge, "rerank_k", 0)
    if isinstance(rerank_k, bool) or not isinstance(rerank_k, int) or rerank_k < 1:
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    try:
        documents = [document_from_fact(r.fact) for r in response.results[:rerank_k]]
        if deadline is None:
            outcome = judge.rerank_and_judge(query, documents)
        else:
            outcome = judge.rerank_and_judge(query, documents, deadline=deadline)
    except Exception as exc:  # noqa: BLE001 — a judge never breaks a recall
        logger.warning("Answer check could not reorder; reporting the recall as it was (%s)",
                       type(exc).__name__)
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    if not isinstance(outcome, RerankVerdict):
        return JudgeOutcome(None, STATUS_UNAVAILABLE)
    if outcome.order is not None:
        if not apply_order(response, outcome.order):
            # The verdict describes a top three nobody will see.
            logger.warning("Answer check returned an unusable order; ignoring it")
            return JudgeOutcome(None, STATUS_UNAVAILABLE)
        response.local_reranker_status = response.reranker_status
        response.reranker_applied = True
        response.reranker_status = STATUS_LISTWISE
    return _settled(outcome.verdict, getattr(outcome, "status", "") or STATUS_UNAVAILABLE)


def apply_order(response: Any, order: tuple[int, ...]) -> bool:
    """Reorder the first ``len(order)`` results; the rest keep their places."""
    from superlocalmemory.retrieval.jev_rerank import is_permutation

    results = list(response.results)
    if not is_permutation(order, len(results)):
        return False
    block = results[: len(order)]
    response.results = [block[i] for i in order] + results[len(order):]
    return True


def build_trace(retrieval_engine: Any, outcome: JudgeOutcome, response: Any, *,
                recall_started: float, judge_started: float, judge_ended: float,
                ended: float) -> AnswerCheckTrace | None:
    """How long the check took and who ran it, for the Answer Check history.

    Pure: attribute reads and arithmetic, no I/O. Times are monotonic seconds
    from the same clock as ``recall_started``. Never raises: a recall must not
    fail because its timing could not be described (None instead).
    """
    try:
        return _trace(retrieval_engine, outcome, response, recall_started,
                      judge_started, judge_ended, ended)
    except Exception as exc:  # noqa: BLE001 — telemetry never breaks a recall
        logger.debug("answer-check trace skipped: %s", type(exc).__name__)
        return None


def _trace(retrieval_engine: Any, outcome: JudgeOutcome, response: Any,
           recall_started: float, judge_started: float, judge_ended: float,
           ended: float) -> AnswerCheckTrace:
    from superlocalmemory.retrieval.jev_rerank import STATUS_LISTWISE

    verdict = outcome.verdict
    if verdict is not None:
        backend = getattr(verdict, "backend", "")
        threshold = getattr(verdict, "threshold", None)
    else:
        backend = getattr(getattr(retrieval_engine, _JUDGE_ATTR, None), "backend", "")
        threshold = None
    detail = outcome.detail if outcome.detail in ANSWER_CHECK_DETAILS else DETAIL_NONE
    return AnswerCheckTrace(
        detail=detail,
        backend=backend if isinstance(backend, str) else "",
        threshold=threshold if isinstance(threshold, (int, float)) else None,
        reordered=getattr(response, "reranker_status", "") == STATUS_LISTWISE,
        retrieval_ms=round((judge_started - recall_started) * 1000.0, 1),
        judge_ms=round((judge_ended - judge_started) * 1000.0, 1),
        total_ms=round((ended - recall_started) * 1000.0, 1),
    )


__all__ = ["apply_order", "build_trace", "genuine_verdict", "reorders",
           "rerank_and_judge", "run_answer_check"]
