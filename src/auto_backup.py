#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Auto Backup System

Automated backup system for memory.db:
- Configurable interval: 24h (daily) or 7 days (weekly, default)
- Timestamped backups in ~/.claude-memory/backups/
- Retention policy: keeps last N backups (default 10)
- Auto-triggers on memory operations when backup is due
- Manual backup via CLI
"""
import sqlite3
import shutil
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".claude-memory"
DB_PATH = MEMORY_DIR / "memory.db"
BACKUP_DIR = MEMORY_DIR / "backups"
CONFIG_FILE = MEMORY_DIR / "backup_config.json"

# Defaults
DEFAULT_INTERVAL_HOURS = 168  # 7 days
DEFAULT_MAX_BACKUPS = 10
MIN_INTERVAL_HOURS = 1  # Safety: no more than once per hour


class AutoBackup:
    """Automated backup manager for SuperLocalMemory V2."""

    def __init__(self, db_path: Path = DB_PATH, backup_dir: Path = BACKUP_DIR):
        self.db_path = db_path
        self.backup_dir = backup_dir
        self.config = self._load_config()
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Create backup directory if it doesn't exist."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict:
        """Load backup configuration."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                defaults = self._default_config()
                for key in defaults:
                    if key not in config:
                        config[key] = defaults[key]
                return config
            except (json.JSONDecodeError, IOError):
                pass
        return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'enabled': True,
            'interval_hours': DEFAULT_INTERVAL_HOURS,
            'max_backups': DEFAULT_MAX_BACKUPS,
            'last_backup': None,
            'last_backup_file': None,
        }

    def _save_config(self):
        """Save configuration to disk."""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save backup config: {e}")

    def configure(self, interval_hours: Optional[int] = None,
                  max_backups: Optional[int] = None,
                  enabled: Optional[bool] = None) -> Dict:
        """
        Update backup configuration.

        Args:
            interval_hours: Hours between backups (24 for daily, 168 for weekly)
            max_backups: Maximum number of backup files to retain
            enabled: Enable/disable auto-backup

        Returns:
            Updated configuration
        """
        if interval_hours is not None:
            self.config['interval_hours'] = max(MIN_INTERVAL_HOURS, interval_hours)
        if max_backups is not None:
            self.config['max_backups'] = max(1, max_backups)
        if enabled is not None:
            self.config['enabled'] = enabled

        self._save_config()
        return self.get_status()

    def is_backup_due(self) -> bool:
        """Check if a backup is due based on configured interval."""
        if not self.config.get('enabled', True):
            return False

        last_backup = self.config.get('last_backup')
        if not last_backup:
            return True  # Never backed up

        try:
            last_dt = datetime.fromisoformat(last_backup)
            interval = timedelta(hours=self.config.get('interval_hours', DEFAULT_INTERVAL_HOURS))
            return datetime.now() >= last_dt + interval
        except (ValueError, TypeError):
            return True  # Invalid date, backup now

    def check_and_backup(self) -> Optional[str]:
        """
        Check if backup is due and create one if needed.
        Called automatically by memory operations.

        Returns:
            Backup filename if created, None if not due
        """
        if not self.is_backup_due():
            return None

        return self.create_backup()

    def create_backup(self, label: Optional[str] = None) -> str:
        """
        Create a backup of memory.db.

        Args:
            label: Optional label for the backup (e.g., 'pre-migration')

        Returns:
            Backup filename
        """
        if not self.db_path.exists():
            logger.warning("No database to backup")
            return ""

        self._ensure_backup_dir()

        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        label_suffix = f"-{label}" if label else ""
        backup_name = f"memory-{timestamp}{label_suffix}.db"
        backup_path = self.backup_dir / backup_name

        try:
            # Use SQLite backup API for consistency (safe even during writes)
            source_conn = sqlite3.connect(self.db_path)
            backup_conn = sqlite3.connect(backup_path)
            try:
                source_conn.backup(backup_conn)
            finally:
                backup_conn.close()
            source_conn.close()

            # Get backup size
            size_mb = backup_path.stat().st_size / (1024 * 1024)

            # Update config
            self.config['last_backup'] = datetime.now().isoformat()
            self.config['last_backup_file'] = backup_name
            self._save_config()

            logger.info(f"Backup created: {backup_name} ({size_mb:.1f} MB)")

            # v2.7.4: Also backup learning.db if it exists
            learning_db = self.db_path.parent / "learning.db"
            if learning_db.exists():
                try:
                    learning_backup_name = f"learning-{timestamp}{label_suffix}.db"
                    learning_backup_path = self.backup_dir / learning_backup_name
                    l_source = sqlite3.connect(learning_db)
                    l_backup = sqlite3.connect(learning_backup_path)
                    try:
                        l_source.backup(l_backup)
                    finally:
                        l_backup.close()
                    l_source.close()
                    l_size = learning_backup_path.stat().st_size / (1024 * 1024)
                    logger.info(f"Learning backup created: {learning_backup_name} ({l_size:.1f} MB)")
                except Exception as le:
                    logger.warning(f"Learning DB backup failed (non-critical): {le}")

            # Enforce retention policy
            self._enforce_retention()

            return backup_name

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up partial backup
            if backup_path.exists():
                backup_path.unlink()
            return ""

    def _enforce_retention(self):
        """Remove old backups exceeding max_backups limit."""
        max_backups = self.config.get('max_backups', DEFAULT_MAX_BACKUPS)

        # Enforce for both memory and learning backups (v2.7.4)
        for pattern in ['memory-*.db', 'learning-*.db']:
            backups = sorted(
                self.backup_dir.glob(pattern),
                key=lambda f: f.stat().st_mtime
            )

            while len(backups) > max_backups:
                oldest = backups.pop(0)
                try:
                    oldest.unlink()
                    logger.info(f"Removed old backup: {oldest.name}")
                except OSError as e:
                    logger.error(f"Failed to remove old backup {oldest.name}: {e}")

    def list_backups(self) -> List[Dict]:
        """
        List all available backups (memory.db + learning.db).

        Returns:
            List of backup info dictionaries
        """
        backups = []

        if not self.backup_dir.exists():
            return backups

        # v2.7.4: List both memory and learning backups
        for pattern in ['memory-*.db', 'learning-*.db']:
            for backup_file in sorted(
                self.backup_dir.glob(pattern),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            ):
                stat = backup_file.stat()
                db_type = 'learning' if backup_file.name.startswith('learning-') else 'memory'
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'age_hours': round((datetime.now() - datetime.fromtimestamp(stat.st_mtime)).total_seconds() / 3600, 1),
                    'type': db_type,
                })

        # Sort all by creation time (newest first)
        backups.sort(key=lambda b: b['created'], reverse=True)
        return backups

    def restore_backup(self, filename: str) -> bool:
        """
        Restore database from a backup file.

        Args:
            filename: Backup filename to restore from

        Returns:
            Success status
        """
        backup_path = self.backup_dir / filename

        if not backup_path.exists():
            logger.error(f"Backup file not found: {filename}")
            return False

        try:
            # Create a safety backup of current state first
            self.create_backup(label='pre-restore')

            # Determine target DB based on filename prefix
            if filename.startswith('learning-'):
                target_db = self.db_path.parent / "learning.db"
            else:
                target_db = self.db_path

            # Restore using SQLite backup API
            source_conn = sqlite3.connect(backup_path)
            target_conn = sqlite3.connect(target_db)
            try:
                source_conn.backup(target_conn)
            finally:
                target_conn.close()
            source_conn.close()

            logger.info(f"Restored from backup: {filename} → {target_db.name}")
            return True

        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False

    def get_status(self) -> Dict:
        """
        Get backup system status.

        Returns:
            Status dictionary
        """
        backups = self.list_backups()
        next_backup = None

        if self.config.get('enabled') and self.config.get('last_backup'):
            try:
                last_dt = datetime.fromisoformat(self.config['last_backup'])
                interval = timedelta(hours=self.config.get('interval_hours', DEFAULT_INTERVAL_HOURS))
                next_dt = last_dt + interval
                if next_dt > datetime.now():
                    next_backup = next_dt.isoformat()
                else:
                    next_backup = 'overdue'
            except (ValueError, TypeError):
                next_backup = 'unknown'

        # Calculate interval display
        hours = self.config.get('interval_hours', DEFAULT_INTERVAL_HOURS)
        if hours >= 168:
            interval_display = f"{hours // 168} week(s)"
        elif hours >= 24:
            interval_display = f"{hours // 24} day(s)"
        else:
            interval_display = f"{hours} hour(s)"

        # v2.7.4: Separate counts for memory vs learning backups
        memory_backups = [b for b in backups if b.get('type') == 'memory']
        learning_backups = [b for b in backups if b.get('type') == 'learning']

        return {
            'enabled': self.config.get('enabled', True),
            'interval_hours': hours,
            'interval_display': interval_display,
            'max_backups': self.config.get('max_backups', DEFAULT_MAX_BACKUPS),
            'last_backup': self.config.get('last_backup'),
            'last_backup_file': self.config.get('last_backup_file'),
            'next_backup': next_backup,
            'backup_count': len(memory_backups),
            'learning_backup_count': len(learning_backups),
            'total_size_mb': round(sum(b['size_mb'] for b in backups), 2),
            'backups': backups,
        }


# CLI Interface
if __name__ == "__main__":
    import sys

    backup = AutoBackup()

    if len(sys.argv) < 2:
        print("Auto Backup System - SuperLocalMemory V2")
        print("\nUsage:")
        print("  python auto_backup.py status              Show backup status")
        print("  python auto_backup.py backup [label]      Create backup now")
        print("  python auto_backup.py list                List all backups")
        print("  python auto_backup.py restore <filename>  Restore from backup")
        print("  python auto_backup.py configure           Show/set configuration")
        print("    --interval <hours>     Set backup interval (24=daily, 168=weekly)")
        print("    --max-backups <N>      Set max retained backups")
        print("    --enable               Enable auto-backup")
        print("    --disable              Disable auto-backup")
        sys.exit(0)

    command = sys.argv[1]

    if command == "status":
        status = backup.get_status()
        print(f"\nAuto-Backup Status")
        print(f"{'='*40}")
        print(f"  Enabled:       {'Yes' if status['enabled'] else 'No'}")
        print(f"  Interval:      {status['interval_display']}")
        print(f"  Max backups:   {status['max_backups']}")
        print(f"  Last backup:   {status['last_backup'] or 'Never'}")
        print(f"  Next backup:   {status['next_backup'] or 'N/A'}")
        print(f"  Backup count:  {status['backup_count']}")
        print(f"  Total size:    {status['total_size_mb']} MB")

    elif command == "backup":
        label = sys.argv[2] if len(sys.argv) > 2 else None
        print("Creating backup...")
        result = backup.create_backup(label=label)
        if result:
            print(f"Backup created: {result}")
        else:
            print("Backup failed!")
            sys.exit(1)

    elif command == "list":
        backups = backup.list_backups()
        if not backups:
            print("No backups found.")
        else:
            print(f"\n{'Filename':<45} {'Size':<10} {'Age':<15} {'Created'}")
            print("-" * 95)
            for b in backups:
                age = f"{b['age_hours']:.0f}h" if b['age_hours'] < 48 else f"{b['age_hours']/24:.0f}d"
                created = b['created'][:19]
                print(f"{b['filename']:<45} {b['size_mb']:<10.2f} {age:<15} {created}")

    elif command == "restore":
        if len(sys.argv) < 3:
            print("Error: Backup filename required")
            print("Usage: python auto_backup.py restore <filename>")
            sys.exit(1)
        filename = sys.argv[2]
        print(f"Restoring from {filename}...")
        if backup.restore_backup(filename):
            print("Restore successful! Restart any running tools to use the restored data.")
        else:
            print("Restore failed!")
            sys.exit(1)

    elif command == "configure":
        interval = None
        max_bk = None
        enabled = None

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--interval' and i + 1 < len(sys.argv):
                interval = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--max-backups' and i + 1 < len(sys.argv):
                max_bk = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--enable':
                enabled = True
                i += 1
            elif sys.argv[i] == '--disable':
                enabled = False
                i += 1
            else:
                i += 1

        if interval is not None or max_bk is not None or enabled is not None:
            status = backup.configure(
                interval_hours=interval,
                max_backups=max_bk,
                enabled=enabled
            )
            print("Configuration updated:")
        else:
            status = backup.get_status()
            print("Current configuration:")

        print(f"  Enabled:     {'Yes' if status['enabled'] else 'No'}")
        print(f"  Interval:    {status['interval_display']}")
        print(f"  Max backups: {status['max_backups']}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
