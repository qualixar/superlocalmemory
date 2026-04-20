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

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("superlocalmemory.backup")

# ---------------------------------------------------------------------------
# V3 paths
# ---------------------------------------------------------------------------
MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"
BACKUP_DIR = MEMORY_DIR / "backups"
CONFIG_FILE = MEMORY_DIR / "backup_config.json"

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
        self.base_dir = base_dir or MEMORY_DIR
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
        backup_path = self.backup_dir / filename
        if not backup_path.exists():
            logger.error("Backup not found: %s", filename)
            return False

        try:
            self.create_backup(label="pre-restore")

            target = (
                self.db_path.parent / "learning.db"
                if filename.startswith("learning-")
                else self.db_path
            )

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
