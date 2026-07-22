# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Named retention rules engine for compliance (GDPR, HIPAA, custom).

Rules are bound to profiles. Each rule specifies a retention period
in days. The engine can identify expired facts and enforce deletion.

Retention rules are stored in a dedicated SQLite table and operate
independently of the main memory lifecycle system.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RETENTION_RULES_TABLE = """
CREATE TABLE IF NOT EXISTS retention_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    days INTEGER NOT NULL,
    description TEXT DEFAULT '',
    framework TEXT NOT NULL DEFAULT 'custom',
    action TEXT NOT NULL DEFAULT 'archive',
    applies_to TEXT NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(profile_id, rule_name)
)
"""

# Terminal lifecycle zone each action moves an expired fact into. 'notify'
# changes nothing — it only surfaces the count so an operator can act.
# atomic_facts.lifecycle CHECK allows only active/warm/cold/archived, so both
# retention actions land in the 'archived' lifecycle zone. 'tombstone' is
# additionally flagged purgeable via archive_status so a purge job can find it —
# this avoids a risky rebuild of the (large) atomic_facts CHECK constraint.
_ACTION_LIFECYCLE = {"archive": "archived", "tombstone": "archived"}
_TOMBSTONE_ACTIONS = frozenset({"tombstone"})
_VALID_ACTIONS = ("archive", "tombstone", "notify")

_FACTS_TABLE_CHECK = """
SELECT name FROM sqlite_master
WHERE type='table' AND name='atomic_facts'
"""


class RetentionEngine:
    """Named retention rules for compliance (GDPR, HIPAA, custom).

    Rules are bound to profiles. Each rule specifies a retention period.
    The engine can identify expired facts and enforce deletion.
    """

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        self._ensure_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the retention_rules table, self-migrating older schemas.

        Pre-GDPR-model tables had only (rule_name, days, description). Add the
        framework/action/applies_to columns in place so existing rules keep
        working (defaults: framework='custom', action='archive').
        """
        self._db.execute(_RETENTION_RULES_TABLE)
        cols = {r[1] for r in self._db.execute(
            "PRAGMA table_info(retention_rules)").fetchall()}
        for col, ddl in (
            ("framework", "framework TEXT NOT NULL DEFAULT 'custom'"),
            ("action", "action TEXT NOT NULL DEFAULT 'archive'"),
            ("applies_to", "applies_to TEXT NOT NULL DEFAULT '{}'"),
        ):
            if col not in cols:
                self._db.execute(f"ALTER TABLE retention_rules ADD COLUMN {ddl}")
        self._db.commit()

    def _has_facts_table(self) -> bool:
        """Check if atomic_facts table exists in the database."""
        row = self._db.execute(_FACTS_TABLE_CHECK).fetchone()
        return row is not None

    @staticmethod
    def _age_in_days(created_at_str: str) -> float:
        """Calculate age from an ISO timestamp string to now, in days."""
        try:
            created = datetime.fromisoformat(created_at_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - created).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        profile_id: str,
        rule_name: str,
        days: int,
        description: str = "",
    ) -> None:
        """Add a retention rule to a profile.

        Args:
            profile_id: Profile this rule applies to.
            rule_name: Human-readable name (e.g. 'GDPR-30d').
            days: Retention period in days.
            description: Optional description of the rule.

        Raises:
            sqlite3.IntegrityError: If rule_name already exists for profile.
        """
        self._db.execute(
            "INSERT OR REPLACE INTO retention_rules "
            "(profile_id, rule_name, days, description) "
            "VALUES (?, ?, ?, ?)",
            (profile_id, rule_name, days, description),
        )
        self._db.commit()
        logger.info(
            "Added retention rule '%s' (%d days) to profile '%s'",
            rule_name, days, profile_id,
        )

    def remove_rule(self, profile_id: str, rule_name: str) -> None:
        """Remove a retention rule.

        Args:
            profile_id: Profile the rule belongs to.
            rule_name: Name of the rule to remove.
        """
        self._db.execute(
            "DELETE FROM retention_rules "
            "WHERE profile_id = ? AND rule_name = ?",
            (profile_id, rule_name),
        )
        self._db.commit()
        logger.info(
            "Removed retention rule '%s' from profile '%s'",
            rule_name, profile_id,
        )

    def get_rules(self, profile_id: str) -> list[dict[str, Any]]:
        """Get all retention rules for a profile (full GDPR shape)."""
        rows = self._db.execute(
            "SELECT id, rule_name, days, description, framework, action, "
            "applies_to, created_at FROM retention_rules "
            "WHERE profile_id = ? ORDER BY id",
            (profile_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                applies = json.loads(r[6]) if r[6] else {}
            except (ValueError, TypeError):
                applies = {}
            out.append({
                "id": r[0],
                "name": r[1],
                "rule_name": r[1],          # legacy alias
                "retention_days": r[2],
                "days": r[2],               # legacy alias
                "description": r[3],
                "framework": r[4],
                "action": r[5],
                "applies_to": applies,
                "created_at": r[7],
            })
        return out

    # ------------------------------------------------------------------
    # GDPR-model API (used by the compliance route + dashboard)
    # ------------------------------------------------------------------

    def create_rule(
        self,
        *,
        name: str,
        framework: str,
        retention_days: int,
        action: str,
        applies_to: dict | None,
        profile_id: str,
    ) -> int:
        """Create a retention rule and return its row id.

        ``action`` is one of archive | tombstone | notify. Raises ValueError on
        an invalid action so the route returns a clear 4xx rather than storing
        an unenforceable rule.
        """
        if action not in _VALID_ACTIONS:
            raise ValueError(f"action must be one of {_VALID_ACTIONS}")
        cur = self._db.execute(
            "INSERT OR REPLACE INTO retention_rules "
            "(profile_id, rule_name, days, description, framework, action, applies_to) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (profile_id, name, int(retention_days), "", framework, action,
             json.dumps(applies_to or {})),
        )
        self._db.commit()
        logger.info(
            "Created retention rule '%s' (%dd, %s/%s) for profile '%s'",
            name, retention_days, framework, action, profile_id,
        )
        return int(cur.lastrowid or 0)

    def list_rules(self, profile_id: str | None = None) -> list[dict[str, Any]]:
        """List rules for one profile, or all rules when profile_id is None."""
        if profile_id is not None:
            return self.get_rules(profile_id)
        rows = self._db.execute(
            "SELECT DISTINCT profile_id FROM retention_rules"
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.extend(self.get_rules(r[0]))
        return out

    def delete_rule(self, profile_id: str, name: str) -> bool:
        """Delete a rule by (profile_id, name). Returns True if a row was removed."""
        cur = self._db.execute(
            "DELETE FROM retention_rules WHERE profile_id = ? AND rule_name = ?",
            (profile_id, name),
        )
        self._db.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Expiration detection
    # ------------------------------------------------------------------

    def get_expired_facts(self, profile_id: str) -> list[str]:
        """Get fact IDs that have exceeded their retention period.

        Checks each fact's created_at against the profile's shortest
        retention rule. A fact is expired if its age exceeds the
        minimum retention days across all rules for the profile.

        Args:
            profile_id: Profile to check facts for.

        Returns:
            List of expired fact IDs (as strings).
        """
        rules = self.get_rules(profile_id)
        if not rules:
            return []

        # Use the shortest retention period (most restrictive)
        min_days = min(r["retention_days"] for r in rules)

        if not self._has_facts_table():
            return []

        # atomic_facts is keyed by fact_id (TEXT hex), NOT an integer id.
        rows = self._db.execute(
            "SELECT fact_id, created_at FROM atomic_facts WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()

        expired: list[str] = []
        for row in rows:
            fact_id = str(row[0])
            created_at = row[1] if len(row) > 1 else None
            if created_at and self._age_in_days(created_at) > min_days:
                expired.append(fact_id)

        return expired

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def enforce(self, profile_id: str) -> dict[str, Any]:
        """Enforce every retention rule for a profile, honoring each action.

        For each rule, facts in the profile older than its retention_days are
        moved to the rule's terminal lifecycle zone:
          * archive   → lifecycle 'archived'  (retained, hidden from recall)
          * tombstone → lifecycle 'tombstoned' (soft-deleted, purgeable)
          * notify    → unchanged (only counted, so an operator can act)

        GDPR-safe: soft-state transitions, never a raw DELETE — erasure is an
        explicit separate operation. All updates are scoped to ``profile_id``
        and keyed by ``fact_id`` (TEXT), fixing the prior ``id``/int() bug that
        made enforcement crash outright.

        Returns per-action counts + the affected fact ids.
        """
        rules = self.get_rules(profile_id)
        result: dict[str, Any] = {
            "profile_id": profile_id,
            "archived": 0, "tombstoned": 0, "notified": 0,
            "deleted_count": 0,  # legacy alias (== tombstoned)
            "affected_ids": [],
        }
        if not rules or not self._has_facts_table():
            return result

        rows = self._db.execute(
            "SELECT fact_id, created_at FROM atomic_facts WHERE profile_id = ?",
            (profile_id,),
        ).fetchall()

        affected: set[str] = set()
        for rule in rules:
            action = rule.get("action", "archive")
            days = rule.get("retention_days", rule.get("days", 0))
            expired = [
                str(r[0]) for r in rows
                if r[1] and self._age_in_days(r[1]) > days
            ]
            if not expired:
                continue
            if action == "notify":
                result["notified"] += len(expired)
                affected.update(expired)
                continue
            zone = _ACTION_LIFECYCLE.get(action)
            if zone is None:
                continue
            placeholders = ",".join("?" for _ in expired)
            self._db.execute(
                f"UPDATE atomic_facts SET lifecycle = ? "
                f"WHERE profile_id = ? AND fact_id IN ({placeholders})",
                [zone, profile_id, *expired],
            )
            if action in _TOMBSTONE_ACTIONS:
                # Flag purgeable without violating the lifecycle CHECK.
                try:
                    self._db.execute(
                        f"UPDATE atomic_facts SET archive_status = 'tombstoned' "
                        f"WHERE profile_id = ? AND fact_id IN ({placeholders})",
                        [profile_id, *expired],
                    )
                except Exception:
                    pass
            result["archived" if action == "archive" else "tombstoned"] += len(expired)
            affected.update(expired)

        self._db.commit()
        result["affected_ids"] = sorted(affected)
        result["deleted_count"] = result["tombstoned"]  # legacy alias
        logger.info(
            "Retention enforcement for '%s': archived=%d tombstoned=%d notified=%d",
            profile_id, result["archived"], result["tombstoned"], result["notified"],
        )
        return result
