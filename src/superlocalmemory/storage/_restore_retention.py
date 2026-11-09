# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restore leftovers are bounded and covered by erasure (L1-12).

A restore leaves two kinds of file holding a person's words:

  restore-delta/<ts>/   the memories written after the copy, verbatim, kept
                        until they are added back
  pre-restore/*.db      the whole store as it was before each restore

Retention:

  * A delta is removed as soon as nothing in it is waiting: every memory was
    added back or is already in the store, no confirmed kind is still waiting
    for its memory to finish enriching, and nothing failed.
  * Anything a pending restore or a pending re-import names is never touched.
  * Beyond that, a leftover delta or safety copy is removed once it is older
    than ``RETENTION_DAYS`` AND is not among the ``KEEP`` newest of its kind --
    so the newest copies survive however old they are, and the folder cannot
    grow without bound.

Erasure: when a profile is erased, its rows are removed from every delta file
at once, and every safety copy that still holds it is registered with the
backup-obligation ledger (the same ledger as ``backups/``), so the erasure
receipt does not claim completeness while such a copy exists and a later
restore re-erases the profile. Pruning a copy discharges its obligation.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import (
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from superlocalmemory.storage._restore_types import (
    DELTA_DIR,
    INTENT_NAME,
    OUTCOME_NAME,
    SAFETY_DIR,
)

logger = logging.getLogger(__name__)

RETENTION_DAYS = 30
KEEP = 2
_SAFETY_COPY = re.compile(
    r"^(?P<stem>memory|learning)-\d{8}-\d{6}-\d{6}(?:-\d+)?-(?:damaged-)?before-restore\.db$")
_COMPANIONS = ("-wal", "-shm", "-journal")
#: Delta files that hold a profile's own rows. ``erasures.jsonl`` is a ledger
#: and ``meta.json`` holds profile ids only.
_PROFILE_ROWS = ("memories.jsonl", "kind_edits.jsonl", "deleted.jsonl", "profiles.jsonl")


def _referenced_deltas(data_root: Path) -> set[Path]:
    """Deltas a pending restore or a pending re-import still needs."""
    keep: set[Path] = set()
    intent = read_json(Path(data_root) / INTENT_NAME) or {}
    for key in ("delta_dir", "final_delta_dir"):
        if intent.get(key):
            keep.add(Path(intent[key]).resolve())
    outcome = read_json(Path(data_root) / OUTCOME_NAME) or {}
    if outcome.get("reimport_pending") and outcome.get("delta_dir"):
        keep.add(Path(outcome["delta_dir"]).resolve())
    return keep


def _busy(data_root: Path) -> bool:
    """A restore part-way through, or a re-import waiting: prune no safety copy."""
    intent = read_json(Path(data_root) / INTENT_NAME) or {}
    outcome = read_json(Path(data_root) / OUTCOME_NAME) or {}
    return intent.get("stage") == "writing" or bool(outcome.get("reimport_pending"))


def discard_delta(data_root: Path, delta_dir: Path) -> bool:
    """Remove a settled delta (never one a pending restore or re-import names)."""
    from superlocalmemory.storage.upgrade_restore import _remove_delta_dir

    delta_dir = Path(delta_dir)
    if delta_dir.resolve() in _referenced_deltas(data_root):
        return False
    _remove_delta_dir(data_root, delta_dir)
    outcome = read_json(Path(data_root) / OUTCOME_NAME)
    if outcome and outcome.get("delta_dir") and Path(outcome["delta_dir"]) == delta_dir:
        write_json_atomic(Path(data_root) / OUTCOME_NAME,
                          {**outcome, "delta_dir": None, "delta_removed": True})
    return not delta_dir.exists()


def _old_and_surplus(paths: list[Path], now: float) -> list[Path]:
    """Of ``paths``, those beyond the ``KEEP`` newest AND older than the window."""
    ordered = sorted(paths, key=lambda p: p.name, reverse=True)   # names carry the stamp
    horizon = now - RETENTION_DAYS * 86400
    return [p for p in ordered[KEEP:] if p.stat().st_mtime < horizon]


def _discharge(data_root: Path, path: Path) -> None:
    if not (Path(data_root) / "backup_obligations.db").exists():
        return
    from superlocalmemory.infra.backup_obligations import BackupObligationStore

    BackupObligationStore(Path(data_root)).discharge_for_snapshot(
        str(path), "pre-restore copy removed by retention")


def prune_restore_artifacts(data_root: Path, *, now: float | None = None) -> list[Path]:
    """Apply the retention above. Returns what was removed (explicit paths only)."""
    from superlocalmemory.storage.upgrade_restore import _remove_delta_dir

    data_root, now = Path(data_root), (time.time() if now is None else now)
    removed: list[Path] = []
    deltas = data_root / DELTA_DIR
    if deltas.is_dir() and not deltas.is_symlink():
        keep = _referenced_deltas(data_root)
        candidates = [d for d in deltas.iterdir()
                      if d.is_dir() and not d.is_symlink() and d.resolve() not in keep]
        for old in _old_and_surplus(candidates, now):
            _remove_delta_dir(data_root, old)
            if not old.exists():
                removed.append(old)
    safety = data_root / SAFETY_DIR
    if safety.is_dir() and not safety.is_symlink() and not _busy(data_root):
        copies = [p for p in safety.iterdir()
                  if p.is_file() and not p.is_symlink() and _SAFETY_COPY.match(p.name)]
        for stem in ("memory", "learning"):
            mine = [p for p in copies if _SAFETY_COPY.match(p.name).group("stem") == stem]
            for old in _old_and_surplus(mine, now):
                for companion in (old, *(Path(f"{old}{s}") for s in _COMPANIONS)):
                    companion.unlink(missing_ok=True)
                _discharge(data_root, old)
                removed.append(old)
    if removed:
        logger.info("[SLM] Removed %d old restore file(s)", len(removed))
    return removed


def _scrub_delta(delta_dir: Path, profile_id: str) -> int:
    dropped = 0
    for name in _PROFILE_ROWS:
        path = delta_dir / name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        kept = [r for r in rows if r.get("profile_id") != profile_id]
        if len(kept) != len(rows):
            write_jsonl_atomic(path, kept)
            dropped += len(rows) - len(kept)
    return dropped


def cover_profile_erasure(data_root: Path, profile_id: str, *, erasure_id: str,
                          store: Any, retention_days: int) -> dict[str, int]:
    """Erase ``profile_id`` from every delta; register every safety copy holding it."""
    from superlocalmemory.infra.backup_obligations import scan_backup_snapshots

    data_root = Path(data_root)
    scrubbed = 0
    deltas = data_root / DELTA_DIR
    if deltas.is_dir() and not deltas.is_symlink():
        for delta in sorted(deltas.iterdir()):
            if delta.is_dir() and not delta.is_symlink():
                scrubbed += _scrub_delta(delta, profile_id)
    registered = 0
    report = scan_backup_snapshots(data_root / SAFETY_DIR, profile_id)
    for kind, found in (("contains", report.hits), ("unreadable", report.unreadable)):
        for snap_path, epoch in found:
            store.record(profile_id=profile_id, erasure_id=erasure_id, snapshot_path=snap_path,
                         snapshot_epoch=epoch, retention_days=retention_days, kind=kind)
            registered += 1
    return {"restore_delta_rows_erased": scrubbed, "pre_restore_copies_registered": registered}


__all__ = ["KEEP", "RETENTION_DAYS", "cover_profile_erasure", "discard_delta",
           "prune_restore_artifacts"]
