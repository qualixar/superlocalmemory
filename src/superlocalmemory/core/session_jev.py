# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Jev chooses the order of the memories a session starts with.

Every recall a person asks for can already be reordered by Jev
(``retrieval.jev_rerank``). A session start could not: it loads context rather
than answering a question, so it skips the answer check
(``core.answer_check_scope``), and Jev's reordering rides in the same request.
The SessionStart hook's own path (``slm session-context``) never reached a
judge at all. So the memories an agent starts with were ordered by importance
and recency alone, while the person had chosen Jev to judge relevance.

This asks Jev for the ORDER only, once per session start:

* only when the person turned Jev reordering on AND accepted that more
  memories leave the machine (``judge_selection.jev_rerank_k`` > 0); for
  everyone else nothing is sent and nothing changes;
* under a hard budget (``BUDGET_S``): late, failed or malformed means the
  original order, never a slower or broken start;
* the request's answer check is discarded. A session start is not a question,
  so it may never be told "that isn't in SuperLocalMemory"; nothing is hidden,
  added or rewritten, only reordered;
* a credential-only memory is never sent (the judge keeps it in place).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

#: Seconds a session start may wait for Jev, whatever the recall timeout says.
BUDGET_S = 1.5

T = TypeVar("T")


def _default_judge(retrieval_config: Any, timeout_s: float) -> Any | None:
    from superlocalmemory.core import judge_selection

    return judge_selection.session_rank_judge(retrieval_config, timeout_s)


def jev_order(query: str, items: Sequence[T], content: Callable[[T], str],
              retrieval_config: Any, *, budget_s: float = BUDGET_S,
              judge_factory: Callable[[Any, float], Any] | None = None) -> list[T]:
    """``items`` in Jev's order when it may and does answer in time, else as given."""
    items = list(items)
    if len(items) < 2 or not (query or "").strip() or retrieval_config is None:
        return items
    judge = None
    try:
        judge = (judge_factory or _default_judge)(retrieval_config, budget_s)
        if judge is None:
            return items
        from superlocalmemory.retrieval.judge_recipe import JudgeDocument

        docs = [JudgeDocument(content=str(content(item) or "")) for item in items]
        outcome = judge.rerank_and_judge(query, docs, deadline=time.monotonic() + budget_s)
        return _apply(items, getattr(outcome, "order", None))
    except Exception as exc:  # noqa: BLE001 - a session start never fails on its judge
        logger.debug("session start: Jev order unavailable (%s)", type(exc).__name__)
        return items
    finally:
        _shutdown(judge)


def _apply(items: list[T], order: Any) -> list[T]:
    if not order:
        return items
    order = tuple(order)
    from superlocalmemory.retrieval.jev_rerank import is_permutation

    if len(order) > len(items) or not is_permutation(order, len(order)):
        return items
    return [*(items[i] for i in order), *items[len(order):]]


def _shutdown(judge: Any) -> None:
    stop = getattr(judge, "shutdown", None)
    if callable(stop):
        try:
            stop()
        except Exception:  # noqa: BLE001
            pass


__all__ = ["BUDGET_S", "jev_order"]
