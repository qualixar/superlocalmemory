# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The one import flow every door calls: preview, run, status, cancel, undo (WS-9, D-19, D-49).

CLI, HTTP and the dashboard all end up here, so they behave the same. The pack is on by default
(flag ``portability.import``; a person's own off or the kill switch stops it): every call says which switch is off and
how to turn it on. An import is a decision (D-49): any caller may start one, and it is recorded
in the decision log with who and through which channel, as kind ``import``, and its undo is the
rollback of the whole run. One import runs at a time per profile.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from superlocalmemory.core import decisions
from superlocalmemory.portability import constants as C
from superlocalmemory.portability import rollback as _rollback
from superlocalmemory.portability import runner, runs, summary
from superlocalmemory.portability.jobs import JOBS
from superlocalmemory.portability.model import ImportFormatError, Parsed
from superlocalmemory.portability.readers import Doc
from superlocalmemory.portability.registry import parse_any

logger = logging.getLogger(__name__)

OFF_WORDS = ("Importing from other apps is turned off. Turn it on with "
             "'slm flags set portability.import on' (or the Import card on the dashboard).")
KILLED_WORDS = ("Importing is stopped because the SuperLocalMemory kill switch is on. "
                "Turn the kill switch off first ('slm flags kill off').")


class ImportOff(PermissionError):
    """The pack is off; the message names the switch."""


class ImportRefused(ValueError):
    """The request cannot be done now; the message is plain words."""


def flag_view(home: Any = None) -> dict:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    flags = FeatureFlags(home or canonical_data_root())
    return {"flag": C.FLAG, "state": flags.state(C.FLAG).value, "enabled": flags.enabled(C.FLAG),
            "kill_switch": flags.killed}


def require_on() -> None:
    view = flag_view()
    if view["kill_switch"]:
        raise ImportOff(KILLED_WORDS)
    if not view["enabled"]:
        raise ImportOff(OFF_WORDS)


def limits() -> dict:
    return {"max_upload_mb": C.MAX_UPLOAD_BYTES // (1024 * 1024),
            "disputes_per_run": C.DISPUTES_PER_RUN, "judge_calls_per_run": C.JUDGE_CALLS_PER_RUN}


def _read_conn(engine: Any):
    from superlocalmemory.truth.invariants import open_read_only

    return open_read_only(str(engine._db.db_path))


def preview(engine: Any, profile_id: str, docs: list[Doc], fmt: str | None) -> tuple[Parsed, dict]:
    """What a file holds, and how much of it is new. Writes nothing."""
    require_on()
    parsed = parse_any(docs, fmt)
    conn = _read_conn(engine)
    try:
        todo, left_out = runner.plan(conn, profile_id, engine._config, parsed)
    finally:
        conn.close()
    view = summary.describe(parsed, left_out)
    view.update(would_save=len(todo), already_there=left_out.get(runner.ALREADY_THERE, 0),
                limits=limits())
    if left_out.get(runner.ALREADY_THERE):
        view["summary"] += (f" {left_out[runner.ALREADY_THERE]:,} are already in your memory "
                            "and will be left exactly as they are.")
    return parsed, view


def _summary_line(view: dict) -> str:
    found = []
    if view["conversations"]:
        found.append(f"{view['conversations']:,} conversations")
    if view["memories"]:
        found.append(f"{view['memories']:,} memories")
    return f"Imported {' and '.join(found) or 'a file'} from {view['label']}"


def start_run(engine: Any, runtime: Any, *, profile_id: str, docs: list[Doc], fmt: str | None,
              label: str, decider: decisions.Decider,
              on_finish: Callable[[], None] | None = None) -> dict:
    """Record the run (and the decision) and start the background job. Returns the run."""
    require_on()
    parsed = parse_any(docs, fmt)
    with engine._db.raw_connection() as conn:
        if not runs.has_tables(conn):
            raise ImportRefused("This store has not been upgraded yet; restart SuperLocalMemory "
                                "once and try again.")
        if runs.active_for(conn, profile_id):
            raise ImportRefused("An import is already running. Wait for it to finish, or "
                                "cancel it first.")
        view = summary.describe(parsed, {})
        run_id = runs.create(
            conn, profile_id=profile_id, fmt=parsed.format, source_label=label,
            actor=decider.actor, channel=decider.channel, conversations=parsed.conversations,
            total_items=len(parsed.items), skipped=dict(parsed.skipped),
            first_at=view["first_at"] or "", last_at=view["last_at"] or "")
        decision_id = decisions.log(
            conn, profile_id=profile_id, kind=decisions.IMPORT, subject_id=run_id, action="run",
            decider=decider, summary=_summary_line(view),
            detail={"format": parsed.format, "items": len(parsed.items)})
        if decision_id:
            conn.execute("UPDATE import_runs SET decision_id = ? WHERE run_id = ?",
                         (decision_id, run_id))
    ctx = runner.Context(db=engine._db, runtime=runtime, config=engine._config,
                         profile_id=profile_id)

    def job(stop: threading.Event) -> None:
        try:
            runner.execute(ctx, run_id, parsed, stop)
        finally:
            if on_finish:
                on_finish()

    JOBS.start(run_id, job)
    return get_run(engine, profile_id, run_id) or {"run_id": run_id}


def get_run(engine: Any, profile_id: str, run_id: str) -> dict | None:
    conn = _read_conn(engine)
    try:
        found = runs.get(conn, profile_id, run_id)
    finally:
        conn.close()
    return _with_progress(found)


def _with_progress(found: dict | None) -> dict | None:
    if found is None:
        return None
    total = max(0, found["total_items"] - sum(
        found["skipped"].get(k, 0) for k in (runner.ALREADY_THERE, "duplicate")))
    found["progress"] = (min(1.0, (found["saved_items"] + found["skipped"].get(runner.REJECTED, 0))
                             / total) if total else 1.0)
    found["working"] = found["state"] in runs.ACTIVE and JOBS.is_live(found["run_id"])
    return found


def overview(engine: Any, profile_id: str, limit: int = 20) -> dict:
    """Recent runs, the switch, and the limits. Closes a run the daemon lost (a restart)."""
    _close_lost_runs(engine, profile_id)
    conn = _read_conn(engine)
    try:
        listed = [_with_progress(r) for r in runs.recent(conn, profile_id, limit)]
    finally:
        conn.close()
    return {**flag_view(), "limits": limits(), "runs": listed}


def _close_lost_runs(engine: Any, profile_id: str) -> None:
    conn = _read_conn(engine)
    try:
        lost = [r for r in runs.recent(conn, profile_id, runs.MAX_LIST)
                if r["state"] in (runs.QUEUED, runs.RUNNING) and not JOBS.is_live(r["run_id"])]
    finally:
        conn.close()
    for r in lost:
        with engine._db.raw_connection() as write:
            runs.set_state(write, r["run_id"], runs.FAILED, only_from=(runs.QUEUED, runs.RUNNING),
                           note="SuperLocalMemory stopped while this was running. What was saved "
                                "is kept; you can undo it or run it again to add the rest.")


def cancel(engine: Any, profile_id: str, run_id: str) -> dict:
    found = get_run(engine, profile_id, run_id)
    if found is None:
        raise LookupError(run_id)
    with engine._db.raw_connection() as conn:
        runs.request_cancel(conn, run_id)
    JOBS.stop(run_id)
    return get_run(engine, profile_id, run_id) or found


def start_undo(engine: Any, runtime: Any, *, profile_id: str, run_id: str,
               decider: decisions.Decider, wait_s: float = 0.0) -> dict:
    """Roll the whole run back in the background; the decision is marked undone when it ends."""
    found = get_run(engine, profile_id, run_id)
    if found is None:
        raise LookupError(run_id)
    if found["state"] == runs.ROLLED_BACK:
        raise ImportRefused("That import was already undone.")
    if JOBS.is_live(run_id):
        if found["state"] == runs.ROLLING_BACK:
            return found
        JOBS.stop(run_id)  # an import still running: stop it first, then take it back
        if not JOBS.join(run_id, 30.0):
            raise ImportRefused("The import is still stopping. Try again in a moment.")
    with engine._db.raw_connection() as conn:
        runs.set_state(conn, run_id, runs.ROLLING_BACK, note="")
    decision_id = found.get("decision_id") or ""

    def finished(conn: Any) -> None:
        if decision_id:
            decisions.mark_undone(conn, decision_id, decider)
        decisions.log(conn, profile_id=profile_id, kind=decisions.IMPORT, subject_id=run_id,
                      action="rollback", decider=decider,
                      summary=f"Took back an import from {found['source_label'] or 'a file'}",
                      detail={"saved": found["saved_items"]})
        conn.execute("UPDATE import_runs SET rolled_back_at = ?, rolled_back_by = ? "
                     "WHERE run_id = ?", (runs.now(), decider.actor, run_id))

    JOBS.start(run_id, lambda stop: _rollback.rollback(engine, runtime, profile_id, run_id, stop,
                                                       finished))
    if wait_s > 0:
        JOBS.join(run_id, wait_s)
    return get_run(engine, profile_id, run_id) or found


__all__ = ["ImportFormatError", "ImportOff", "ImportRefused", "cancel", "flag_view", "get_run",
           "limits", "overview", "preview", "require_on", "start_run", "start_undo"]
