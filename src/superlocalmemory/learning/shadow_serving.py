# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Pick the model that ranks one recall while a candidate is under test.

Before 4.1.25 nothing served a candidate ranker: a retrained model was
persisted with ``is_candidate = 1`` and every recall was still ranked by the
active model, so the "A/B test" compared the active model with itself. This module is the serve-time half of
the fix. For each recall it

1. asks the shadow router whether a candidate is under test,
2. routes the recall's ``query_id`` to an arm,
3. loads the candidate's verified booster when the arm is ``candidate``, and
4. persists the assignment (``shadow_assignments``) only once the model that
   will actually rank the recall is known — so a candidate that fails to load
   is never credited with an answer it did not produce.

Two reads and one small INSERT per recall, and only while a test runs.
Never raises.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["serving_model"]

_CANDIDATE_CACHE: dict[tuple[str, int], Any] = {}
_CACHE_LOCK = threading.Lock()


def _active_model_id(learning_db: Path, profile_id: str) -> int | None:
    try:
        conn = sqlite3.connect(f"{learning_db.resolve().as_uri()}?mode=ro",
                               uri=True, timeout=0.5)
        try:
            row = conn.execute(
                "SELECT id FROM learning_model_state "
                "WHERE profile_id = ? AND is_active = 1 LIMIT 1",
                (profile_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


def _load_candidate(learning_db: Path, profile_id: str, cid: int) -> Any:
    key = (str(learning_db), int(cid))
    with _CACHE_LOCK:
        if key in _CANDIDATE_CACHE:
            return _CANDIDATE_CACHE[key]
    from superlocalmemory.learning.model_cache import _parse_row

    try:
        conn = sqlite3.connect(f"{learning_db.resolve().as_uri()}?mode=ro",
                               uri=True, timeout=0.5)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT state_bytes, bytes_sha256, feature_names, trained_at "
                "FROM learning_model_state WHERE id = ? AND profile_id = ?",
                (int(cid), profile_id),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    model = _parse_row(profile_id, dict(row)) if row is not None else None
    with _CACHE_LOCK:
        if len(_CANDIDATE_CACHE) > 8:
            _CANDIDATE_CACHE.clear()
        _CANDIDATE_CACHE[key] = model
    return model


def serving_model(learning_db: Path | str, profile_id: str, query_id: str,
                  active_model: Any) -> Any:
    """The model that should rank this recall. ``active_model`` if no test."""
    if not query_id:
        return active_model
    try:
        from superlocalmemory.core.shadow_router import (
            ARM_CANDIDATE,
            get_shadow_router,
        )

        path = Path(learning_db)
        router = get_shadow_router(learning_db=str(path), profile_id=profile_id)
        cid = router.refresh_candidate()
        if cid is None:
            return active_model
        chosen = active_model
        if router.route_query(query_id) == ARM_CANDIDATE:
            chosen = _load_candidate(path, profile_id, cid)
            if chosen is None:
                return active_model  # unservable: record nothing
        assignment = router.assign(
            query_id, active_model_id=_active_model_id(path, profile_id))
        if assignment is None:
            return active_model
        if assignment.arm != ARM_CANDIDATE:
            return active_model
        if chosen is active_model:  # a replay first assigned to the candidate
            chosen = _load_candidate(path, profile_id, cid) or active_model
        return chosen
    except Exception as exc:  # pragma: no cover — never break a recall
        logger.debug("shadow serving skipped: %s", exc)
        return active_model
