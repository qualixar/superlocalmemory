# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""SkillActivator — atomic activation of quarantined evolved skills.

Copies a VERIFIED_QUARANTINED skill artifact into the live skill directory
(~/.claude/skills/{skill_name}/SKILL.md), retains the prior artifact as a
.bak file, and provides a tested rollback path.

Invariants:
  1. The destination directory is a child of live_root — path traversal raises ValueError.
  2. The backup is written BEFORE the live copy is overwritten.
  3. Activation is atomic via a .tmp file → os.replace (POSIX atomic on same fs).
  4. Rollback restores from the .bak file and leaves the live copy unchanged.
  5. Only the quarantined artifact identified by quarantine_dir_name is copied;
     no arbitrary file can be activated.

CRIT-2 note: skill_name and quarantine_dir_name are ALWAYS different strings:
  - skill_name: the original skill identifier (e.g. "brainstorming")
  - quarantine_dir_name: the sanitized quarantine subdir (e.g. "brainstorming-vabc12")
Both are required; neither defaults from the other.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from superlocalmemory.infra.data_root import canonical_data_root

logger = logging.getLogger(__name__)

LIVE_SKILLS_ROOT: Path = Path.home() / ".claude" / "skills"
BACKUP_ROOT: Path = canonical_data_root() / "skill_backups"
QUARANTINE_ROOT: Path = canonical_data_root() / "quarantine" / "skills"

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class SkillActivationError(Exception):
    """Raised when activation fails after writing the backup.

    The caller must invoke rollback() after receiving this error to restore
    the previous live artifact.
    """


class SkillActivator:
    """Activates a quarantined evolved skill into the live skill directory."""

    def __init__(
        self,
        *,
        live_root: Path | None = None,
        backup_root: Path | None = None,
        quarantine_root: Path | None = None,
    ) -> None:
        self._live_root = live_root or LIVE_SKILLS_ROOT
        self._backup_root = backup_root or BACKUP_ROOT
        self._quarantine_root = quarantine_root or QUARANTINE_ROOT

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _safe_skill_dir(self, skill_name: str, base: Path) -> Path:
        """Return base / sanitized_name, raising ValueError on traversal.

        Two-layer check (fail-fast + belt-and-suspenders):
          1. Reject the raw skill_name if it contains '..' or is absolute.
          2. After sanitization, resolve the target path and verify it remains
             inside base.  This catches any edge case the regex misses.
        """
        # Fail-fast: detect traversal in raw name before sanitization.
        raw = Path(skill_name)
        if raw.is_absolute() or ".." in raw.parts:
            raise ValueError(
                f"skill_name {skill_name!r} contains path traversal sequences"
            )
        safe = _SAFE_NAME_RE.sub("-", skill_name).lower()[:50] or "skill"
        target = (base / safe).resolve()
        base_resolved = base.resolve()
        if not str(target).startswith(str(base_resolved)):
            raise ValueError(
                f"skill_name {skill_name!r} escapes sandbox: {target}"
            )
        return target

    def _quarantine_path(self, quarantine_dir_name: str) -> Path:
        """Return the SKILL.md path inside quarantine for a given dir name.

        quarantine_dir_name is the sanitized directory name (CRIT-2), e.g.
        'brainstorming-vabc12'.  It is NOT the skill_name.
        """
        return self._quarantine_root / quarantine_dir_name / "SKILL.md"

    def _live_path(self, skill_name: str) -> Path:
        return self._safe_skill_dir(skill_name, self._live_root) / "SKILL.md"

    def _backup_path(self, skill_name: str) -> Path:
        return self._safe_skill_dir(skill_name, self._backup_root) / "SKILL.md.bak"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(
        self,
        skill_name: str,
        quarantine_dir_name: str,
        *,
        actor_id: str = "",
    ) -> dict:
        """Atomically activate a quarantined skill.

        Steps:
          1. Resolve quarantine_path using quarantine_dir_name and verify it exists.
          2. Read quarantine content and compute SHA-256.
          3. Write backup of existing live SKILL.md (if any) to backup_root.
          4. Write quarantine content to live path via .tmp → os.replace.
          5. Return activation metadata.

        Args:
            skill_name: The original skill identifier (CRIT-2), e.g. "brainstorming".
            quarantine_dir_name: The sanitized quarantine subdir, e.g. "brainstorming-vabc12".
            actor_id: Who triggered activation (e.g. "auto" or a user id).

        Raises:
            FileNotFoundError: Quarantine artifact does not exist.
            ValueError: Path traversal detected in skill_name.
            SkillActivationError: Any OS error after the backup was already written.
        """
        q_path = self._quarantine_path(quarantine_dir_name)
        live_path = self._live_path(skill_name)  # raises ValueError on traversal
        backup_path = self._backup_path(skill_name)

        if not q_path.exists():
            raise FileNotFoundError(
                f"Quarantine artifact not found: {q_path}"
            )

        content = q_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        ts = datetime.now(timezone.utc).isoformat()

        # Step 3: backup existing live artifact BEFORE any overwrite
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_written = False
        if live_path.exists():
            shutil.copy2(live_path, backup_path)
            backup_written = True
            logger.info(
                "skill_activator: backed up %s → %s",
                live_path, backup_path,
            )

        # Step 4: atomic write via .tmp → rename (same directory = same fs)
        live_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = live_path.with_suffix(".tmp")
        try:
            tmp_path.write_bytes(content)
            os.replace(tmp_path, live_path)
        except OSError as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise SkillActivationError(
                f"atomic write failed for {live_path}: {exc}"
            ) from exc

        logger.info(
            "skill_activator: activated %s @ %s (sha256=%s, actor=%s)",
            skill_name, live_path, content_hash[:12], actor_id,
        )
        return {
            "skill_name": skill_name,
            "live_path": str(live_path),
            "backup_path": str(backup_path) if backup_written else None,
            "content_hash": content_hash,
            "activated_at": ts,
            "actor_id": actor_id,
        }

    def rollback(self, skill_name: str) -> dict:
        """Restore the live SKILL.md from the backup written by activate().

        If no backup exists, removes the live file (the skill was new).
        Never raises on missing backup — logs a warning instead.

        Args:
            skill_name: The original skill identifier, same value used in activate().
        """
        live_path = self._live_path(skill_name)
        backup_path = self._backup_path(skill_name)
        ts = datetime.now(timezone.utc).isoformat()

        if backup_path.exists():
            shutil.copy2(backup_path, live_path)
            logger.info(
                "skill_activator: rolled back %s ← %s",
                live_path, backup_path,
            )
            return {
                "skill_name": skill_name,
                "rolled_back": True,
                "restored_from": str(backup_path),
                "rolled_back_at": ts,
            }
        elif live_path.exists():
            live_path.unlink()
            logger.warning(
                "skill_activator: no backup for %s; removed live file",
                skill_name,
            )
            return {
                "skill_name": skill_name,
                "rolled_back": True,
                "restored_from": None,
                "note": "no backup; live file removed",
                "rolled_back_at": ts,
            }
        else:
            logger.warning(
                "skill_activator: rollback called but no live file for %s",
                skill_name,
            )
            return {
                "skill_name": skill_name,
                "rolled_back": False,
                "note": "nothing to rollback",
                "rolled_back_at": ts,
            }
