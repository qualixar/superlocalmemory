# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Retry, by itself, failed saves that an older version could never clear (#325).

Before 4.2.30 two kinds of failed save stayed in the failed list forever: an
embedding that landed a moment after the save, and a fact archived after the
save. The engine now clears both on retry (``core/engine_ingestion.py``), but the
only retry was ``slm ops resolve <id> --action retry``, one save at a time, and
one reporter had more than 700 of them. A person who does not use a terminal
would never clear them.

So the pending materializer calls :func:`step` on its normal cycle. It is
bounded on every axis:

* at most :data:`BATCH` saves per run, and a run at most every
  :data:`INTERVAL_S` seconds;
* only dead letters whose error is one of :data:`RETRYABLE` (a save that failed
  for another reason is left for a person to look at);
* each save is tried once per SLM version (recorded in the data folder), so a
  save that still fails is not retried on every cycle; the next upgrade, which
  may fix its cause, tries it again;
* only the running profile's saves, by the engine admitted for that profile;
* nothing while the kill switch is on.

Each retry goes through ``ops_remediation._action_retry``, the same path as the
command: the dead letter is removed only when the save left FAILED.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Error texts of failures a retry on this version can clear.
RETRYABLE = (
    "incomplete derivation stages",
    "external projection facts are missing",
    "checkpointed relational facts are missing",
)
BATCH = 20
INTERVAL_S = 300.0
STATE_NAME = "stuck-save-redrive.json"

_lock = threading.Lock()
_last_run = 0.0


def _version() -> str:
    from superlocalmemory import __version__

    return __version__


def _load(state_file: Path, version: str) -> set[str]:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    if not isinstance(data, dict) or data.get("version") != version:
        return set()  # a new version tries every stuck save once more
    return {str(i) for i in data.get("tried", []) if isinstance(i, str)}


def _save(state_file: Path, version: str, tried: set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_name(state_file.name + ".tmp")
    tmp.write_text(json.dumps({"version": version, "tried": sorted(tried)}), encoding="utf-8")
    os.replace(tmp, state_file)


def _candidates(db: Any, profile_id: str, tried: set[str]) -> list[str]:
    """Oldest first: dead letters of this profile with a retryable error, not yet tried."""
    where = " OR ".join("error LIKE ?" for _ in RETRYABLE)
    rows = db.execute(
        f"SELECT original_op_id FROM dead_letter_operations "
        f"WHERE COALESCE(profile_id, 'default')=? AND ({where}) "
        f"ORDER BY dead_lettered_at ASC, id ASC",
        (profile_id, *(f"%{text}%" for text in RETRYABLE)))
    out: list[str] = []
    for row in rows:
        op_id = str(dict(row)["original_op_id"])
        if op_id not in tried and op_id not in out:
            out.append(op_id)
        if len(out) >= BATCH:
            break
    return out


def run(engine: Any, state_file: Path, *, version: str | None = None) -> dict:
    """One bounded redrive; returns ``{"retried": n, "cleared": n}``. Never raises."""
    from superlocalmemory.core.ops_remediation import _action_retry

    version = version or _version()
    db = getattr(engine, "_db", None)
    profile_id = str(getattr(engine, "_profile_id", "") or "default")
    if db is None:
        return {"retried": 0, "cleared": 0}
    tried = _load(state_file, version)
    try:
        todo = _candidates(db, profile_id, tried)
    except Exception as exc:  # noqa: BLE001 -- a store without the dead-letter table
        logger.debug("stuck-save redrive: nothing to read (%s)", exc)
        return {"retried": 0, "cleared": 0}
    cleared = 0
    for op_id in todo:
        try:
            result = _action_retry(db.db_path, engine, op_id, {profile_id})
        except Exception as exc:  # noqa: BLE001 -- one save never stops the rest
            result = {"success": False, "reason": type(exc).__name__}
        tried.add(op_id)
        cleared += 1 if result.get("success") else 0
    if todo:
        try:
            _save(state_file, version, tried)
        except OSError as exc:
            logger.warning("stuck-save redrive: could not record progress: %s", exc)
        logger.info("Stuck-save redrive: %d retried, %d cleared", len(todo), cleared)
    return {"retried": len(todo), "cleared": cleared}


def step(engine: Any, *, now: float | None = None) -> dict | None:
    """Run :func:`run` when due and allowed; None when skipped. Never raises."""
    global _last_run
    now = time.monotonic() if now is None else now
    if not _lock.acquire(blocking=False):
        return None
    try:
        if _last_run and now - _last_run < INTERVAL_S:
            return None
        _last_run = now
        from superlocalmemory.core.feature_flags import kill_switch_on
        from superlocalmemory.infra.data_root import canonical_data_root, state_path

        if kill_switch_on(canonical_data_root()):
            return None
        return run(engine, state_path(STATE_NAME))
    except Exception as exc:  # noqa: BLE001 -- background repair is fail-open
        logger.debug("stuck-save redrive skipped: %s", exc)
        return None
    finally:
        _lock.release()
