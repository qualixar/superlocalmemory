# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""After a restore: add back what was written since the copy, exactly once.

Runs once the engine is up (a daemon background thread, or ``slm db restore``).
Every memory written after the copy goes back in through the same canonical
ingestion as any other write -- its own profile, scope, sharing and session --
under the idempotency key ``restore-reimport:<memory_id>``, so running this
twice, or resuming it after a crash, never makes a duplicate. A kind is carried
only when every fact of that memory had the same confirmed kind.

Kinds a person confirmed after the copy are put back with their original source
(``_restore_kinds``): the newer confirmation wins.

A memory whose id the store still holds is not added again: that happens when
a restore stopped before its write committed and the export is handed to the
re-import anyway (``_restore_boot._unusable``).

When nothing is left waiting -- every memory is back or already present,
nothing failed, no confirmation still waits for its memory to finish enriching
-- the export is removed (L1-12). It holds the person's words verbatim.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_delta import load_kind_edits, load_memories, metadata_of
from superlocalmemory.storage._restore_types import OUTCOME_NAME, ReimportReport

logger = logging.getLogger(__name__)

SOURCE_TYPE = "restore-reimport"


def _shared_with(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return [s.strip() for s in str(value).split(",") if s.strip()]
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


def _already(conn: sqlite3.Connection, profile_id: str, key: str) -> bool:
    try:
        return conn.execute(
            "SELECT 1 FROM ingestion_operations WHERE profile_id=? AND source_type=? "
            "AND idempotency_key=?", (profile_id, SOURCE_TYPE, key)).fetchone() is not None
    except sqlite3.Error:
        return False


def _still_there(conn: sqlite3.Connection, memory_id: str) -> bool:
    """The store still holds this memory (a restore that never wrote)."""
    try:
        return conn.execute("SELECT 1 FROM memories WHERE memory_id=?",
                            (memory_id,)).fetchone() is not None
    except sqlite3.Error:
        return False


def _readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True)


def _reimport_memories(engine: Any, rows: list[dict[str, Any]], db_path: Path,
                       counts: dict[str, Any]) -> None:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id
    from superlocalmemory.core.ingestion_command import UnknownProfileError
    from superlocalmemory.storage.memory_kinds import METADATA_KEY, parse_kind

    actor = local_trusted_actor_id(SOURCE_TYPE)
    for row in rows:
        key = f"{SOURCE_TYPE}:{row['memory_id']}"
        profile_id = str(row.get("profile_id") or "default")
        with closing(_readonly(db_path)) as conn:
            if _already(conn, profile_id, key) or _still_there(conn, row["memory_id"]):
                counts["already_present"] += 1
                continue
        meta = {k: v for k, v in metadata_of(row).items() if not str(k).startswith("_slm")}
        kind = parse_kind(row.get("memory_kind"))
        trusted = {METADATA_KEY: kind.value} if kind is not None else None
        try:
            result = canonical_store(
                engine, str(row.get("content") or ""), source_type=SOURCE_TYPE,
                trusted_actor_id=actor, metadata=meta, trusted_metadata=trusted,
                scope=str(row.get("scope") or "personal"),
                shared_with=_shared_with(row.get("shared_with")),
                session_id=str(row.get("session_id") or ""),
                session_date=row.get("session_date") or None,
                speaker=str(row.get("speaker") or ""), role=str(row.get("role") or "user"),
                idempotency_key=key, require_complete=False, profile_id=profile_id)
        except UnknownProfileError:
            counts["skipped_unknown_profile"] += 1
            continue
        except Exception as exc:  # noqa: BLE001 - reported per memory, never fatal
            counts["failed"] += 1
            counts["errors"].append(f"{row.get('memory_id')}: {type(exc).__name__}")
            logger.warning("[SLM] Could not add back memory %s: %s", row.get("memory_id"), exc)
            continue
        if result == []:
            counts["skipped_rejected"] += 1
        else:
            counts["added"] += 1


def _reapply_kinds(db_path: Path, edits: list[dict[str, Any]], counts: dict[str, Any],
                   *, give_up: bool = False) -> None:
    from superlocalmemory.storage._restore_kinds import reapply_kinds
    from superlocalmemory.storage.write_lock import get_write_lock

    if not edits:
        return
    with get_write_lock(db_path), closing(sqlite3.connect(str(db_path), timeout=30)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
        if "memory_kind_source" not in cols:
            counts["kinds_skipped"] += len(edits)
            return
        history = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                               "AND name='memory_kind_history'").fetchone() is not None
        reapply_kinds(conn, edits, counts, source_type=SOURCE_TYPE, history=history,
                      give_up=give_up)
        conn.commit()


def _settle(data_root: Path, delta_dir: Path, report: ReimportReport, passes: int) -> None:
    """Record the run; remove the export once nothing in it is waiting."""
    outcome = read_json(Path(data_root) / OUTCOME_NAME)
    pending = report.failed > 0 or report.kinds_waiting > 0
    if outcome is not None:
        write_json_atomic(Path(data_root) / OUTCOME_NAME, {
            **outcome, "reimport_pending": pending, "reimport": report.as_dict(),
            "reimport_passes": passes})
    if pending or report.skipped_rejected or report.skipped_unknown_profile:
        # It still holds a memory that is not in the store. Kept, under the
        # bounded retention of ``_restore_retention`` (and erasure scrubs it).
        return
    try:
        from superlocalmemory.storage._restore_retention import (
            discard_delta, prune_restore_artifacts,
        )

        discard_delta(Path(data_root), Path(delta_dir))
        prune_restore_artifacts(Path(data_root))
    except Exception as exc:  # noqa: BLE001 - housekeeping never fails a re-import
        logger.warning("[SLM] Restore housekeeping after the re-import skipped: %s", exc)


def reimport_delta(engine: Any, delta_dir: Path, *, data_root: Path | None = None
                   ) -> ReimportReport:
    """Add back memories and confirmed kinds from ``delta_dir``. Safe to repeat."""
    from superlocalmemory.storage._restore_kinds import MAX_PASSES

    db_path = Path(engine._db.db_path)
    counts: dict[str, Any] = {k: 0 for k in (
        "added", "already_present", "skipped_unknown_profile", "skipped_rejected",
        "failed", "kinds_reapplied", "kinds_skipped", "kinds_waiting")}
    counts["errors"] = []
    previous = read_json(Path(data_root) / OUTCOME_NAME) if data_root is not None else None
    passes = int((previous or {}).get("reimport_passes") or 0) + 1
    memories = load_memories(delta_dir)
    if memories:
        _reimport_memories(engine, memories, db_path, counts)
    _reapply_kinds(db_path, load_kind_edits(delta_dir), counts, give_up=passes >= MAX_PASSES)
    report = ReimportReport(**{**counts, "errors": counts["errors"][:20]})
    if data_root is not None:
        _settle(Path(data_root), Path(delta_dir), report, passes)
    logger.info("[SLM] After the restore: %s", report.as_dict())
    return report


def run_pending_reimport(engine: Any, data_root: Path) -> ReimportReport | None:
    """Daemon hook: re-import once when the last restore left work to do."""
    outcome = read_json(Path(data_root) / OUTCOME_NAME)
    if not outcome or not outcome.get("reimport_pending") or not outcome.get("delta_dir"):
        return None
    return reimport_delta(engine, Path(outcome["delta_dir"]), data_root=Path(data_root))


__all__ = ["SOURCE_TYPE", "reimport_delta", "run_pending_reimport"]
