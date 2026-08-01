# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Automated backup manager for SuperLocalMemory V3.

Provides:
    * Configurable interval (daily / weekly)
    * Timestamped SQLite-safe backups via the ``sqlite3.backup()`` API
    * Retention policy (keeps last *N* backups)
    * Restore with automatic pre-restore safety snapshot

V3 change: base directory is ``~/.superlocalmemory/`` (was ``~/.claude-memory/``).
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from superlocalmemory.infra.data_root import DynamicStatePath, canonical_data_root

logger = logging.getLogger("superlocalmemory.backup")

# ---------------------------------------------------------------------------
# V3 paths
# ---------------------------------------------------------------------------
MEMORY_DIR = DynamicStatePath()
DB_PATH = DynamicStatePath("memory.db")
BACKUP_DIR = DynamicStatePath("backups")
CONFIG_FILE = DynamicStatePath("backup_config.json")

# Defaults
DEFAULT_INTERVAL_HOURS = 168   # 7 days
DEFAULT_MAX_BACKUPS = 10
MIN_INTERVAL_HOURS = 1

# ---------------------------------------------------------------------------
# SLM Managed Database Registry
# ---------------------------------------------------------------------------
# Every database that SLM creates and manages. The backup system backs up
# ONLY these databases — nothing else. When a new SLM module creates a new
# database file, add it here so it gets included in backups.
#
# Each user may have a different subset (e.g., some don't have code_graph.db
# if they never used the code graph feature). The backup system checks which
# ones exist and only backs up what's present.

MANAGED_DATABASES: tuple[str, ...] = (
    "memory.db",        # Core: facts, entities, graph, embeddings, sessions
    "learning.db",      # Learning pipeline: signals, patterns, ranker
    "audit_chain.db",   # Audit trail: compliance, provenance chain
    "code_graph.db",    # Code knowledge graph: symbols, references
    "pending.db",       # Pending operations queue
    "audit.db",         # Legacy audit (pre-v3.4)
)


# ---------------------------------------------------------------------------
# Coherent multi-store backup set
# ---------------------------------------------------------------------------


class BackupVerificationError(Exception):
    """Raised when a backup set fails checksum re-verification."""


class BackupRestoreError(Exception):
    """Raised when a restore cannot be safely completed."""


@dataclass(frozen=True)
class StoreEntry:
    """Describes one database file within a backup set."""

    store_name: str   # filename, e.g. "memory.db"
    file_path: str    # absolute path inside the final backup directory
    size_bytes: int
    sha256: str       # SHA-256 hex digest of the backup copy


@dataclass(frozen=True)
class BackupSetManifest:
    """Describes a coherent snapshot of all managed databases.

    All stores share a single epoch so callers can detect sets assembled
    from different points in time and reject them. Checksums allow
    independent verification of every backup file before restore.
    """

    set_id: str                       # unique identifier for this backup set
    epoch: int                        # Unix timestamp when the set was created
    stores: tuple[StoreEntry, ...]    # one entry per backed-up store
    manifest_hash: str                # SHA-256 over sorted store checksums
    verified: bool                    # True only after Phase-4 re-verification
    created_at: str = ""              # ISO-8601 UTC creation timestamp
    product_version: str = ""         # reserved for version tracking


class BackupCoordinator:
    """Creates and verifies coherent backup sets spanning all managed databases.

    A backup set groups every managed store under a single epoch and publishes
    an atomic manifest only when all per-store checksums pass re-verification.
    Any mismatch detected during re-verification causes the entire staging
    directory to be removed without publication.

    Args:
        managed_databases: Ordered tuple of DB filenames to include.
        base_dir: Directory where the live databases reside.
        backup_dir: Directory where backup sets are written.
    """

    def __init__(
        self,
        managed_databases: tuple[str, ...],
        base_dir: Path,
        backup_dir: Path,
    ) -> None:
        self._managed_databases = managed_databases
        self._base_dir = Path(base_dir)
        self._backup_dir = Path(backup_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_backup_set(self) -> BackupSetManifest:
        """Copy all existing managed stores and publish a verified manifest.

        The algorithm has six phases:
          1. Identify which stores exist on disk.
          2. Create a staging directory.
          3. Fence SQLite writers (BEGIN IMMEDIATE) then copy every store and
             record per-file SHA-256 checksums (Phase 3). If a ``lance/``
             directory exists under the base dir, copy it into staging as well.
          4. Re-read every staging file and compare to Phase-3 hashes.
             Mismatch → staging removed, BackupVerificationError raised.
          5. Build the manifest from the verified checksums.
          6. Atomically rename staging → final directory and write manifest.json.

        The ``lance/`` directory (LanceDB vector index) is treated as an
        out-of-manifest companion: it is copied recursively into the backup set
        and restored alongside the SQLite stores. Its presence is noted in the
        server log. If absent, the step is silently skipped.

        Returns:
            BackupSetManifest with verified=True.

        Raises:
            BackupVerificationError: if any staging file's content changed
                between Phase 3 and Phase 4.
        """
        set_id = uuid.uuid4().hex[:16]
        epoch = int(time.time())
        staging_dir = self._backup_dir / f".staging_{set_id}"
        staging_dir.mkdir(parents=True, exist_ok=True)

        existing_dbs = [
            db for db in self._managed_databases
            if (self._base_dir / db).exists()
        ]
        sqlite_paths = [self._base_dir / db for db in existing_dbs]

        # staging_records: (db_name, staging_path, size_bytes, phase3_sha256)
        staging_records: list[tuple[str, Path, int, str]] = []

        # Phases 2–3: fence writers, copy, hash
        with self._writer_fence(sqlite_paths):
            for db_name in existing_dbs:
                src = self._base_dir / db_name
                dest = staging_dir / db_name
                self._sqlite_backup(src, dest)
                sha = self._compute_entry_sha256(dest)
                staging_records.append((db_name, dest, dest.stat().st_size, sha))

            # Copy the LanceDB vector directory if present.
            lance_src = self._base_dir / "lance"
            if lance_src.is_dir():
                lance_staging = staging_dir / "lance"
                shutil.copytree(str(lance_src), str(lance_staging))
                logger.info(
                    "Backup set %s: captured lance/ directory (%d items)",
                    set_id,
                    sum(1 for _ in lance_staging.rglob("*") if _.is_file()),
                )

        # Phase 4: re-verify every staging copy
        for db_name, staging_path, _size, expected_sha in staging_records:
            actual_sha = self._compute_entry_sha256(staging_path)
            if actual_sha != expected_sha:
                shutil.rmtree(str(staging_dir), ignore_errors=True)
                raise BackupVerificationError(
                    f"Checksum mismatch for {db_name}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

        # Phase 5: build manifest (file_path points to where files will land)
        final_dir = self._backup_dir / f"backup_{set_id}"
        entries = tuple(
            StoreEntry(
                store_name=db_name,
                file_path=str(final_dir / db_name),
                size_bytes=size,
                sha256=sha,
            )
            for db_name, _sp, size, sha in staging_records
        )
        manifest = BackupSetManifest(
            set_id=set_id,
            epoch=epoch,
            stores=entries,
            manifest_hash=self._compute_manifest_hash(entries),
            verified=True,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Phase 6: atomic publish
        staging_dir.rename(final_dir)
        (final_dir / "manifest.json").write_text(
            json.dumps(asdict(manifest), indent=2)
        )

        return manifest

    def restore_from_manifest(self, manifest: BackupSetManifest) -> None:
        """Restore all stores from a verified manifest.

        The restore proceeds in three phases to guarantee cross-store atomicity:

        Phase A — Verify:
            Re-derive the manifest hash and verify every backup file's checksum.
            No live files are touched until all checks pass. Raises
            BackupRestoreError on any failure.

        Phase B — Pre-restore snapshot:
            Copy the current live version of every store being restored into a
            ``<store>.pre_restore`` sibling file, and the current ``lance/``
            directory (if present) to ``lance.pre_restore/``. These snapshots
            allow a full rollback if the write phase is interrupted.

        Phase C — Staged write:
            Copy each backup file to a ``<store>.restore_staging`` temporary,
            then rename it into place. If any step raises, all pre-restore
            snapshots are copied back to their live locations before the error
            is re-raised, returning the live set to its original coherent state.
            On full success, all snapshot files are removed.

        The ``lance/`` directory companion is handled with the same snapshot and
        rollback discipline: if the backup set contains a ``lance/`` subdirectory
        it is restored recursively; absence is silently skipped on both sides.

        Raises:
            BackupRestoreError: on unverified manifest, hash mismatch, missing
                or corrupted backup files, or if the staged write phase fails
                and the rollback itself encounters an error.
        """
        if not manifest.verified:
            raise BackupRestoreError("Cannot restore from an unverified manifest")

        # Phase A-1: Re-derive manifest_hash from the store entries and compare.
        # This detects tampering of manifest.json where an attacker changes a
        # store's sha256 entry without recalculating manifest_hash.
        computed_hash = self._compute_manifest_hash(manifest.stores)
        if computed_hash != manifest.manifest_hash:
            raise BackupRestoreError(
                "Manifest hash mismatch: backup set is incoherent or has been tampered"
            )

        # Phase A-2: Verify every backup file exists and matches its checksum.
        for entry in manifest.stores:
            src = Path(entry.file_path)
            if not src.exists():
                raise BackupRestoreError(f"Backup file missing: {entry.file_path}")
            actual_sha = self._compute_entry_sha256(src)
            if actual_sha != entry.sha256:
                raise BackupRestoreError(
                    f"Corrupted backup file (checksum mismatch): {entry.store_name}"
                )

        # Determine the backup set directory.
        # Primary: derive from the first store's file_path (most reliable).
        # Fallback for empty-store manifests: reconstruct from the backup_dir +
        # set_id, which is how create_backup_set names the final directory.
        if manifest.stores:
            backup_set_dir: Optional[Path] = Path(manifest.stores[0].file_path).parent
        else:
            candidate = self._backup_dir / f"backup_{manifest.set_id}"
            backup_set_dir = candidate if candidate.is_dir() else None

        lance_backup: Optional[Path] = (
            backup_set_dir / "lance"
            if backup_set_dir is not None and (backup_set_dir / "lance").is_dir()
            else None
        )

        # Phase B: Snapshot the current live copies so we can roll back.
        # pre_restore_map: live_target -> pre_restore_snapshot_path
        pre_restore_map: dict[Path, Path] = {}
        pre_restore_lance: Optional[Path] = None
        try:
            for entry in manifest.stores:
                target = self._base_dir / entry.store_name
                snapshot = target.parent / f"{entry.store_name}.pre_restore"
                if target.exists():
                    shutil.copy2(str(target), str(snapshot))
                pre_restore_map[target] = snapshot

            live_lance = self._base_dir / "lance"
            if lance_backup is not None and live_lance.is_dir():
                pre_restore_lance = self._base_dir / "lance.pre_restore"
                if pre_restore_lance.exists():
                    shutil.rmtree(str(pre_restore_lance))
                shutil.copytree(str(live_lance), str(pre_restore_lance))

        except Exception as exc:
            # Snapshot creation failed — clean up any partial snapshots and abort
            # before touching any live files.
            self._cleanup_pre_restore_snapshots(pre_restore_map, pre_restore_lance)
            raise BackupRestoreError(
                f"Pre-restore snapshot failed, no live files were modified: {exc}"
            ) from exc

        # Phase C: Write restored files into the live directory.
        try:
            for entry in manifest.stores:
                target = self._base_dir / entry.store_name
                staging = target.parent / f"{entry.store_name}.restore_staging"
                shutil.copy2(entry.file_path, str(staging))
                staging.rename(target)

            if lance_backup is not None:
                live_lance = self._base_dir / "lance"
                lance_staging = self._base_dir / "lance.restore_staging"
                if lance_staging.exists():
                    shutil.rmtree(str(lance_staging))
                shutil.copytree(str(lance_backup), str(lance_staging))
                if live_lance.exists():
                    shutil.rmtree(str(live_lance))
                lance_staging.rename(live_lance)
                logger.info("Restore: lance/ directory restored from backup set")

        except Exception as exc:
            # Staged write failed — roll back all live files from pre-restore
            # snapshots so the live set returns to its original coherent state.
            logger.error(
                "Restore write phase failed (%s); rolling back live files to "
                "pre-restore state.",
                exc,
            )
            rollback_errors: list[str] = []
            for target, snapshot in pre_restore_map.items():
                if snapshot.exists():
                    try:
                        shutil.copy2(str(snapshot), str(target))
                    except Exception as rb_exc:
                        rollback_errors.append(f"{target.name}: {rb_exc}")
            # Roll back the lance/ directory if a snapshot was taken.
            if pre_restore_lance is not None and pre_restore_lance.exists():
                try:
                    live_lance = self._base_dir / "lance"
                    if live_lance.exists():
                        shutil.rmtree(str(live_lance))
                    shutil.copytree(str(pre_restore_lance), str(live_lance))
                except Exception as rb_exc:
                    rollback_errors.append(f"lance/: {rb_exc}")

            self._cleanup_pre_restore_snapshots(pre_restore_map, pre_restore_lance)

            if rollback_errors:
                raise BackupRestoreError(
                    f"Restore failed and rollback encountered errors: "
                    f"{'; '.join(rollback_errors)}. Original error: {exc}"
                ) from exc
            raise BackupRestoreError(
                f"Restore write phase failed; live files rolled back to "
                f"pre-restore state: {exc}"
            ) from exc

        # Phase C succeeded — remove pre-restore snapshots.
        self._cleanup_pre_restore_snapshots(pre_restore_map, pre_restore_lance)

    # ------------------------------------------------------------------
    # Internal helpers (factored out for subclass testability)
    # ------------------------------------------------------------------

    def _compute_entry_sha256(self, path: Path) -> str:
        """Return the SHA-256 hex digest of a file's raw bytes."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _compute_manifest_hash(
        self, entries: tuple[StoreEntry, ...]
    ) -> str:
        """Deterministic hash of all store checksums (sorted for stability)."""
        sorted_checksums = sorted(e.sha256 for e in entries)
        payload = "|".join(sorted_checksums).encode()
        return hashlib.sha256(payload).hexdigest()

    @contextmanager
    def _writer_fence(
        self, db_paths: list[Path]
    ) -> Generator[None, None, None]:
        """Hold BEGIN IMMEDIATE on every live SQLite DB during the copy window.

        This blocks concurrent writers for the duration of the copy loop,
        ensuring the source files do not change while being read by
        sqlite3.backup(). Connections are rolled back and closed on exit.
        """
        conns: list[sqlite3.Connection] = []
        for path in db_paths:
            if path.exists():
                conn = sqlite3.connect(str(path))
                conn.execute("BEGIN IMMEDIATE")
                conns.append(conn)
        try:
            yield
        finally:
            for conn in conns:
                try:
                    conn.rollback()
                    conn.close()
                except Exception:  # pragma: no cover – cleanup best-effort
                    pass

    def _sqlite_backup(self, src: Path, dest: Path) -> None:
        """Copy a SQLite database using the Online Backup API (hot copy)."""
        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()

    @staticmethod
    def _cleanup_pre_restore_snapshots(
        snapshot_map: dict[Path, Path],
        lance_snapshot: Optional[Path],
    ) -> None:
        """Remove pre-restore snapshot files and directories.

        Called after a successful restore to tidy up, and on the error path
        after rollback completes. Best-effort: individual removal failures are
        logged but do not raise.
        """
        for snapshot_path in snapshot_map.values():
            if snapshot_path.exists():
                try:
                    snapshot_path.unlink()
                except OSError as exc:
                    logger.warning(
                        "Could not remove pre-restore snapshot %s: %s",
                        snapshot_path.name, exc,
                    )
        if lance_snapshot is not None and lance_snapshot.exists():
            try:
                shutil.rmtree(str(lance_snapshot))
            except OSError as exc:
                logger.warning(
                    "Could not remove lance pre-restore snapshot %s: %s",
                    lance_snapshot, exc,
                )

# ---------------------------------------------------------------------------
# Legacy per-file backup manager (preserved for backward compatibility)
# ---------------------------------------------------------------------------


class BackupManager:
    """Automated backup manager for SuperLocalMemory V3.

    Args:
        db_path: Path to the primary database file.
        backup_dir: Directory where backup files are stored.
        base_dir: Base SLM directory (used for config file + learning DB).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else canonical_data_root()
        self.db_path = db_path or (self.base_dir / "memory.db")
        self.backup_dir = backup_dir or (self.base_dir / "backups")
        self._config_file = self.base_dir / "backup_config.json"
        self.config = self._load_config()
        self._ensure_backup_dir()

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def _ensure_backup_dir(self) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict:
        if self._config_file.exists():
            try:
                raw = json.loads(self._config_file.read_text())
                defaults = self._default_config()
                for k in defaults:
                    raw.setdefault(k, defaults[k])
                return raw
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_config()

    @staticmethod
    def _default_config() -> Dict:
        return {
            "enabled": True,
            "interval_hours": DEFAULT_INTERVAL_HOURS,
            "max_backups": DEFAULT_MAX_BACKUPS,
            "last_backup": None,
            "last_backup_file": None,
        }

    def _save_config(self) -> None:
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            self._config_file.write_text(json.dumps(self.config, indent=2))
        except IOError as exc:
            logger.error("Failed to save backup config: %s", exc)

    def configure(
        self,
        interval_hours: Optional[int] = None,
        max_backups: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> Dict:
        """Update backup configuration and return current status."""
        if interval_hours is not None:
            self.config["interval_hours"] = max(MIN_INTERVAL_HOURS, interval_hours)
        if max_backups is not None:
            self.config["max_backups"] = max(1, max_backups)
        if enabled is not None:
            self.config["enabled"] = enabled
        self._save_config()
        return self.get_status()

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def is_backup_due(self) -> bool:
        """Return ``True`` when a backup should be taken."""
        if not self.config.get("enabled", True):
            return False
        last = self.config.get("last_backup")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
            interval = timedelta(hours=self.config.get("interval_hours", DEFAULT_INTERVAL_HOURS))
            return datetime.now() >= last_dt + interval
        except (ValueError, TypeError):
            return True

    def check_and_backup(self) -> Optional[str]:
        """Create a backup only when one is due. Returns filename or ``None``."""
        if not self.is_backup_due():
            return None
        return self.create_backup()

    # ------------------------------------------------------------------
    # Core backup / restore
    # ------------------------------------------------------------------

    def create_backup(self, label: Optional[str] = None) -> str:
        """Create a timestamped backup via the SQLite online-backup API.

        Returns:
            Backup filename on success, empty string on failure.
        """
        if not self.db_path.exists():
            logger.warning("No database to backup at %s", self.db_path)
            return ""

        self._ensure_backup_dir()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        backup_name = f"memory-{timestamp}{suffix}.db"
        backup_path = self.backup_dir / backup_name

        try:
            source = sqlite3.connect(str(self.db_path))
            dest = sqlite3.connect(str(backup_path))
            try:
                source.backup(dest)
            finally:
                dest.close()
                source.close()

            size_mb = backup_path.stat().st_size / (1024 * 1024)
            self.config["last_backup"] = datetime.now(timezone.utc).isoformat()
            self.config["last_backup_file"] = backup_name
            self._save_config()
            logger.info("Backup created: %s (%.1f MB)", backup_name, size_mb)

            # v3.4.10: Backup ALL .db files in the SLM directory
            self._backup_all_dbs(timestamp, suffix)

            self._enforce_retention()
            return backup_name

        except Exception as exc:
            logger.error("Backup failed: %s", exc)
            if backup_path.exists():
                backup_path.unlink()
            return ""

    def _backup_all_dbs(self, timestamp: str, suffix: str) -> None:
        """Backup all SLM-managed databases alongside the main memory.db.

        Uses the managed database registry — only backs up databases that
        SLM knows about. Add new databases to MANAGED_DATABASES when new
        modules create them.
        """
        slm_dir = self.db_path.parent
        backed_up = 0
        for db_name in MANAGED_DATABASES:
            if db_name == "memory.db":
                continue  # Already backed up by create_backup()
            db_file = slm_dir / db_name
            if not db_file.exists():
                continue  # This user doesn't have this DB — skip

            try:
                prefix = db_file.stem
                name = f"{prefix}-{timestamp}{suffix}.db"
                path = self.backup_dir / name
                src = sqlite3.connect(str(db_file))
                dst = sqlite3.connect(str(path))
                try:
                    src.backup(dst)
                finally:
                    dst.close()
                    src.close()
                backed_up += 1
                logger.info(
                    "Backup: %s (%.1f MB)", name,
                    path.stat().st_size / (1024 * 1024),
                )
            except Exception as exc:
                logger.warning(
                    "%s backup failed (non-critical): %s",
                    db_name, exc,
                )
        if backed_up:
            logger.info("Backed up %d companion databases", backed_up)

    def _enforce_retention(self) -> None:
        """Remove old backups exceeding the configured max."""
        max_backups = self.config.get("max_backups", DEFAULT_MAX_BACKUPS)
        # Build patterns from the managed database registry
        patterns = [f"{Path(db).stem}-*.db" for db in MANAGED_DATABASES]
        for pattern in patterns:
            backups = sorted(
                self.backup_dir.glob(pattern),
                key=lambda f: f.stat().st_mtime,
            )
            while len(backups) > max_backups:
                oldest = backups.pop(0)
                try:
                    oldest.unlink()
                    logger.info("Removed old backup: %s", oldest.name)
                except OSError as exc:
                    logger.error("Failed to remove backup %s: %s", oldest.name, exc)

    def restore_backup(self, filename: str) -> bool:
        """Restore the database from *filename*.

        A safety snapshot of the current state is taken first.
        """
        # Containment: filename must be a bare .db name inside backup_dir — no
        # path separators or traversal. Prevents restoring (and thus copying
        # over memory.db) an arbitrary file the daemon user can read.
        if (not filename or "/" in filename or "\\" in filename
                or ".." in filename or not filename.endswith(".db")):
            logger.error("Restore rejected: invalid backup filename: %r", filename)
            return False
        backup_dir = self.backup_dir.resolve()
        backup_path = (self.backup_dir / filename).resolve()
        if backup_path.parent != backup_dir:
            logger.error("Restore rejected: path escapes backup dir: %r", filename)
            return False
        if not backup_path.exists():
            logger.error("Backup not found: %s", filename)
            return False

        # Derive the target database from the backup filename stem.
        # Backup files are named "{stem}-{timestamp}.db", where stem is the
        # database name without extension (e.g., "audit_chain" for audit_chain.db).
        stem_map = {Path(db).stem: db for db in MANAGED_DATABASES}
        file_stem = filename.split("-", 1)[0]
        target_name = stem_map.get(file_stem)
        if target_name is None:
            logger.error(
                "Restore rejected: unrecognised database stem %r in %r; "
                "expected one of %s",
                file_stem,
                filename,
                list(stem_map),
            )
            return False
        target = self.db_path.parent / target_name

        try:
            self.create_backup(label="pre-restore")

            src = sqlite3.connect(str(backup_path))
            dst = sqlite3.connect(str(target))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()

            logger.info("Restored: %s -> %s", filename, target.name)
            return True

        except Exception as exc:
            logger.error("Restore failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Listing / status
    # ------------------------------------------------------------------

    def list_backups(self) -> List[Dict]:
        """Return metadata for all available backups (newest first)."""
        if not self.backup_dir.exists():
            return []

        result: List[Dict] = []
        for pattern in ("memory-*.db", "learning-*.db"):
            for f in sorted(self.backup_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
                st = f.stat()
                db_type = "learning" if f.name.startswith("learning-") else "memory"
                result.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_mb": round(st.st_size / (1024 * 1024), 2),
                    "created": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    "age_hours": round(
                        (datetime.now() - datetime.fromtimestamp(st.st_mtime)).total_seconds() / 3600, 1
                    ),
                    "type": db_type,
                })
        result.sort(key=lambda b: b["created"], reverse=True)
        return result

    def get_status(self) -> Dict:
        """Return a status summary of the backup system."""
        backups = self.list_backups()
        next_backup = None

        if self.config.get("enabled") and self.config.get("last_backup"):
            try:
                last_dt = datetime.fromisoformat(self.config["last_backup"])
                interval = timedelta(hours=self.config.get("interval_hours", DEFAULT_INTERVAL_HOURS))
                nxt = last_dt + interval
                next_backup = nxt.isoformat() if nxt > datetime.now() else "overdue"
            except (ValueError, TypeError):
                next_backup = "unknown"

        hours = self.config.get("interval_hours", DEFAULT_INTERVAL_HOURS)
        if hours >= 168:
            display = f"{hours // 168} week(s)"
        elif hours >= 24:
            display = f"{hours // 24} day(s)"
        else:
            display = f"{hours} hour(s)"

        mem_bk = [b for b in backups if b.get("type") == "memory"]
        learn_bk = [b for b in backups if b.get("type") == "learning"]

        return {
            "enabled": self.config.get("enabled", True),
            "interval_hours": hours,
            "interval_display": display,
            "max_backups": self.config.get("max_backups", DEFAULT_MAX_BACKUPS),
            "last_backup": self.config.get("last_backup"),
            "last_backup_file": self.config.get("last_backup_file"),
            "next_backup": next_backup,
            "backup_count": len(mem_bk),
            "learning_backup_count": len(learn_bk),
            "total_size_mb": round(sum(b["size_mb"] for b in backups), 2),
            "backups": backups,
        }
