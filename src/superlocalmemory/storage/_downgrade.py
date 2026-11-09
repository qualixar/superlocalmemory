# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Prepare to go back to an older version, without losing anything.

After an update the store is stamped with its schema version, and an older
build refuses to open a store stamped newer than it understands ("This
installation is too old for this database"). That refusal is what keeps an old
build from silently mis-writing a store it does not understand -- so it is only
lifted when every change since the target version is one the older build can
safely ignore:

  * every migration recorded complete above ``target_schema`` declares
    ``DOWNGRADE_FLOOR <= target_schema``, and
  * no completed migration declares ``BREAKING_VERSION > target_schema``.

Then a verified copy is taken (reason ``pre-downgrade``), both stores are
stamped ``target_schema``, and a marker records who prepared it. While the
marker holds, this build does not stamp the store back up at its next start.

The marker holds only while NO other build has run: it records the exact
``.last_version`` file (content and modification time) seen when it was
written. Any other build that starts rewrites that file, and the next start of
this build then drops the marker and stamps normally -- refusing an old build
again is the safe direction.

THE HOLD SURVIVES RESTARTS OF THIS BUILD, ON PURPOSE (L1-11). Between
"prepare" and "install the older version" the daemon may restart on its own
(login items, a crash, ``ensure_daemon`` from a client). If such a restart
undid the preparation, the older version would then refuse the store and the
person would be stranded on it. Holding is safe: preparation is only allowed
when every change since the target version is one the older build ignores, and
this build keeps writing the same schema. It is undone by Cancel (``slm db
prepare-downgrade --cancel`` or the dashboard), taking effect at the next start,
or by any other version starting. The message says exactly that.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_types import (
    DOWNGRADE_MARKER,
    SNAPSHOTS_DIR,
    DowngradeRefusedError,
    DowngradeReport,
)

logger = logging.getLogger(__name__)

_SERIAL = re.compile(r"^M(\d{3})_")

#: The first release that opens a store stamped at each schema version. A
#: store prepared for schema N opens in that release and every later one.
OLDEST_RELEASE_FOR_SCHEMA = {51: "4.1.18"}


def _parse(version: str) -> tuple[int, ...] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", str(version or "").strip())
    return tuple(int(part) for part in match.groups()) if match else None


def downgrade_target_release(target_schema: int, current: str) -> str | None:
    """The release to go back to: the one just before ``current``.

    Never older than the first release that opens the prepared store; ``None``
    when either version is unknown, so the message never names a wrong one.
    """
    oldest = OLDEST_RELEASE_FOR_SCHEMA.get(int(target_schema))
    now, floor = _parse(current), _parse(oldest or "")
    if now is None or floor is None:
        return oldest
    major, minor, patch = now
    previous = (major, minor, patch - 1) if patch > 0 else None
    if previous is None or previous < floor or previous >= now:
        return oldest
    return ".".join(str(part) for part in previous)


def downgrade_install_commands(version: str) -> list[str]:
    """The exact commands that install ``version`` over a newer one."""
    return [f'pipx install --force "superlocalmemory=={version}"',
            f"npm install -g superlocalmemory@{version}",
            f'pip install "superlocalmemory=={version}"']


def _ready_message(target_schema: int, version: str | None) -> str:
    keeps = ("Your memories were copied first. Restarting this version keeps the "
             "preparation; to undo it, choose Cancel (slm db prepare-downgrade --cancel) "
             "and it is undone at the next start.")
    if not version:
        return "Ready to go back. Install the older version now. " + keeps
    oldest = OLDEST_RELEASE_FOR_SCHEMA.get(int(target_schema))
    also = (f" (any version from {oldest} on can open the store now)"
            if oldest and oldest != version else "")
    pipx, npm, pip = downgrade_install_commands(version)
    return (f"Ready to go back to {version}{also}. Install it now with the tool you "
            f"installed SuperLocalMemory with: {pipx}  |  {npm}  |  {pip}. " + keeps)


def _last_version_fingerprint(data_root: Path) -> dict[str, Any]:
    path = Path(data_root) / ".last_version"
    try:
        info = path.stat()
        return {"last_version": path.read_text(encoding="utf-8").strip(),
                "last_version_mtime_ns": info.st_mtime_ns}
    except OSError:
        return {"last_version": None, "last_version_mtime_ns": None}


def _blockers(learning_db: Path, memory_db: Path, target_schema: int) -> list[str]:
    from superlocalmemory.storage._migration_internals import _MODULES, _read_log
    from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS, MIGRATIONS

    logs = {"learning": _read_log(learning_db), "memory": _read_log(memory_db)}
    problems: list[str] = []
    for migration in (*MIGRATIONS, *DEFERRED_MIGRATIONS):
        if logs.get(migration.db_target, {}).get(migration.name) != "complete":
            continue
        module = _MODULES.get(migration.name)
        breaking = int(getattr(module, "BREAKING_VERSION", 0) or 0)
        if breaking > target_schema:
            problems.append(f"{migration.name} changed the store in a way version "
                            f"{target_schema} cannot read")
            continue
        match = _SERIAL.match(migration.name)
        if match and int(match.group(1)) > target_schema:
            floor = getattr(module, "DOWNGRADE_FLOOR", None)
            if floor is None or int(floor) > target_schema:
                problems.append(f"{migration.name} cannot be undone for version "
                                f"{target_schema}")
    return problems


def _stamp(db: Path, version: int) -> None:
    import sqlite3
    from contextlib import closing

    from superlocalmemory.storage._schema_version import (
        ensure_schema_version_table,
        write_schema_version,
    )

    with closing(sqlite3.connect(str(db), timeout=30)) as conn:
        ensure_schema_version_table(conn)
        write_schema_version(conn, version)
        conn.commit()


def prepare_downgrade(*, target_schema: int = 51, requested_by: str, data_root: Path,
                      memory_db: Path, learning_db: Path) -> DowngradeReport:
    """Make the store openable by the version at ``target_schema``. Raises
    ``DowngradeRefusedError`` (nothing changed) when that would not be safe."""
    from superlocalmemory.storage import _snapshot_manifest as sm
    from superlocalmemory.storage.backup import (
        InsufficientDiskSpaceError,
        SnapshotUnusableError,
        _pre_migration_backup,
    )

    data_root, memory_db, learning_db = Path(data_root), Path(memory_db), Path(learning_db)
    problems = _blockers(learning_db, memory_db, int(target_schema))
    if problems:
        raise DowngradeRefusedError("Going back is not safe for this store: "
                                    + "; ".join(problems) + ". Nothing was changed.")
    snaps = data_root / SNAPSHOTS_DIR
    before = {p.name for p in snaps.glob("memory-*-pre-migration.db")} if snaps.is_dir() else set()
    try:
        _pre_migration_backup(learning_db, memory_db, backups_root=snaps,
                              reason="pre-downgrade")
    except InsufficientDiskSpaceError as exc:
        raise DowngradeRefusedError(
            f"Not enough free disk space for a safety copy first (needs "
            f"{exc.needed_bytes:,} bytes, {exc.free_bytes:,} free). Nothing was changed."
        ) from exc
    except SnapshotUnusableError as exc:
        raise DowngradeRefusedError(
            f"The safety copy could not be verified ({exc}). Nothing was changed.") from exc
    new = sorted({p.name for p in snaps.glob("memory-*-pre-migration.db")} - before)
    point_id = sm.stamp_of(Path(new[-1])) if new else None
    for db in (learning_db, memory_db):
        if db.exists():
            _stamp(db, int(target_schema))
    marker = write_json_atomic(data_root / DOWNGRADE_MARKER, {
        "prepared_by": sm.package_version(), "target_schema": int(target_schema),
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "requested_by": str(requested_by)[:200], "point_id": point_id,
        **_last_version_fingerprint(data_root),
    })
    logger.info("[SLM] Store prepared for schema %s by %s (copy %s)",
                target_schema, requested_by, point_id)
    version = downgrade_target_release(int(target_schema), sm.package_version())
    return DowngradeReport(
        prepared=True, target_schema=int(target_schema), point_id=point_id,
        marker=str(marker), message=_ready_message(int(target_schema), version),
        target_version=version,
        install_commands=downgrade_install_commands(version) if version else [])


def cancel_downgrade(data_root: Path) -> bool:
    """Drop the marker; the next start stamps the store as current again."""
    marker = Path(data_root) / DOWNGRADE_MARKER
    if not marker.exists():
        return False
    marker.unlink()
    return True


def downgrade_status(data_root: Path) -> dict[str, Any] | None:
    return read_json(Path(data_root) / DOWNGRADE_MARKER)


def downgrade_hold_active(data_root: Path) -> bool:
    """True while the store is prepared for an older version and no other build ran.

    Called by the migration runner before it stamps. A marker that no longer
    holds (another build ran, a different version wrote it, it is unreadable)
    is removed, and the store is stamped normally.
    """
    from superlocalmemory.storage._snapshot_manifest import package_version

    marker_path = Path(data_root) / DOWNGRADE_MARKER
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    seen = _last_version_fingerprint(Path(data_root))
    holds = bool(marker) and (
        marker.get("prepared_by") == package_version()
        and marker.get("last_version") == seen["last_version"]
        and marker.get("last_version_mtime_ns") == seen["last_version_mtime_ns"])
    if not holds:
        marker_path.unlink(missing_ok=True)
        logger.info("[SLM] Prepare-to-go-back no longer applies (another version ran); "
                    "the store is stamped as current again.")
    return holds


__all__ = ["cancel_downgrade", "downgrade_hold_active", "downgrade_status",
           "prepare_downgrade"]
