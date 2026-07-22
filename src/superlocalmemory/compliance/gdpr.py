# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — GDPR Compliance.

Implements GDPR rights: right to access, right to erasure (forget),
right to data portability (export), and audit trail.
Profile-scoped. All operations logged to compliance_audit.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Friendly export keys → canonical table names (stable Art.20 export contract).
_EXPORT_ALIASES = {
    "facts": "atomic_facts",
    "entities": "canonical_entities",
    "edges": "graph_edges",
    "feedback": "feedback_records",
    "scenes": "memory_scenes",
}


class GDPRCompliance:
    """GDPR compliance operations for memory data.

    Supports:
    - Right to Access (Art. 15): Export all data for a profile
    - Right to Erasure (Art. 17): Delete all data for a profile/entity
    - Right to Portability (Art. 20): Export in machine-readable format
    - Audit Trail: Log all data operations
    """

    # Tables that carry a profile_id column but are NOT tenant memory to be
    # erased/exported wholesale. `profiles` is the tenant record (handled
    # separately, deleted last).
    _NON_MEMORY_SCOPED = frozenset({"profiles"})

    def __init__(self, db) -> None:
        self._db = db

    def _profile_scoped_tables(self) -> list[str]:
        """Every table carrying a ``profile_id`` column — discovered live from
        the schema so a newly-added table can never be silently missed by
        export or erasure (the class of bug that breaks GDPR completeness)."""
        try:
            names = [
                dict(r)["name"]
                for r in self._db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ]
        except Exception:
            return []
        out: list[str] = []
        for t in names:
            if t.startswith("sqlite_") or t in self._NON_MEMORY_SCOPED:
                continue
            try:
                cols = {dict(c)["name"] for c in self._db.execute(f"PRAGMA table_info({t})")}
            except Exception:
                continue
            if "profile_id" in cols:
                out.append(t)
        return out

    # -- Right to Access (Art. 15) -----------------------------------------

    def export_profile_data(self, profile_id: str) -> dict:
        """Export ALL data for a profile in machine-readable format (Art. 15 /
        Art. 20). Covers every profile-scoped table discovered from the schema,
        plus the profile record itself."""
        self._audit("export", "profile", profile_id, "Full data export")

        data: dict = {"profile_id": profile_id, "exported_at": _now()}
        for table in self._profile_scoped_tables():
            try:
                rows = self._db.execute(
                    f"SELECT * FROM {table} WHERE profile_id = ?", (profile_id,)
                )
                data[table] = [dict(r) for r in rows]
            except Exception as exc:  # pragma: no cover — defensive per-table
                logger.warning("export: table %s skipped: %s", table, exc)

        # Profile record itself (the tenant metadata).
        try:
            rows = self._db.execute(
                "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
            )
            data["profile_record"] = [dict(r) for r in rows]
        except Exception:
            data["profile_record"] = []

        # total_items counts the canonical (table-name) keys only, before
        # friendly aliases are added, so it is not double-counted.
        data["total_items"] = sum(
            len(v) for v in data.values() if isinstance(v, list)
        )

        # Backward-compatible friendly aliases for the well-known keys (stable
        # export contract) — they reference the same lists, not copies.
        for friendly, table in _EXPORT_ALIASES.items():
            if table in data:
                data[friendly] = data[table]

        logger.info("Exported %d items for profile '%s'", data["total_items"], profile_id)
        return data

    # -- Right to Erasure (Art. 17) ----------------------------------------

    def forget_profile(self, profile_id: str) -> dict:
        """Delete ALL data for a profile (right to be forgotten, Art. 17).

        Erases every profile-scoped table discovered from the live schema, so a
        newly-added table is covered automatically. The erasure is recorded in
        the tamper-proof audit chain BEFORE any deletion (Art. 5(2)
        accountability) — the in-DB compliance_audit row is itself erased, so
        the chain in a separate DB is the durable evidence.
        """
        if profile_id == "default":
            raise ValueError("Cannot delete the default profile via GDPR erasure. "
                             "Use profile deletion instead.")

        # 1) Durable, tamper-evident record FIRST — survives the erasure.
        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path
            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase", agent_id="gdpr", profile_id=profile_id,
                metadata={"basis": "GDPR Art.17 right-to-erasure"},
            )
        except Exception as exc:
            logger.warning("GDPR erase: audit-chain log failed: %s", exc)
        self._audit("delete", "profile", profile_id, "GDPR erasure request")

        counts: dict[str, int] = {}
        tables = self._profile_scoped_tables()
        # Pass 1 — count every table BEFORE any deletion, so a CASCADE that
        # removes a child (e.g. atomic_facts via memories) does not zero the
        # attribution. Completeness is independent of this.
        for table in tables:
            try:
                rows = self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {table} WHERE profile_id = ?",
                    (profile_id,),
                )
                counts[table] = int(dict(rows[0])["c"]) if rows else 0
            except Exception as exc:  # pragma: no cover
                logger.warning("GDPR erase: count %s failed: %s", table, exc)
                counts[table] = 0
        # Pass 2 — full-tenant wipe with FK enforcement OFF so table order is
        # irrelevant (every profile row in every table goes). FTS shadow rows
        # are still removed by the base-table delete triggers.
        try:
            self._db.execute("PRAGMA foreign_keys=OFF")
        except Exception:
            pass
        try:
            for table in tables:
                try:
                    self._db.execute(
                        f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,)
                    )
                except Exception as exc:  # pragma: no cover — defensive per-table
                    logger.warning("GDPR erase: delete %s failed: %s", table, exc)
            # Delete the profile record itself.
            self._db.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
            counts["profiles"] = 1
        finally:
            try:
                self._db.execute("PRAGMA foreign_keys=ON")
            except Exception:
                pass

        # Erase learning database (separate DB file)
        try:
            from superlocalmemory.learning.database import LearningDatabase
            from superlocalmemory.core.config import DEFAULT_BASE_DIR
            learning_db = LearningDatabase(DEFAULT_BASE_DIR / "learning.db")
            learning_db.reset(profile_id)
            counts["learning_db"] = 1
        except Exception:
            pass

        # VACUUM to remove deleted data from physical file
        try:
            self._db.execute("VACUUM")
        except Exception:
            pass

        logger.info("GDPR erasure for '%s': %d tables, %s", profile_id, len(tables), counts)
        return counts

    def forget_entity(self, entity_name: str, profile_id: str) -> dict:
        """Delete all data related to a specific entity.

        Removes facts mentioning the entity, edges, temporal events,
        and the entity itself. For targeted erasure requests.
        """
        self._audit("delete", "entity", entity_name,
                     f"GDPR entity erasure in profile {profile_id}",
                     profile_id=profile_id)

        entity = self._db.get_entity_by_name(entity_name, profile_id)
        if entity is None:
            return {"deleted": 0, "entity": entity_name, "found": False}

        eid = entity.entity_id
        counts: dict[str, int] = {}

        # Delete facts mentioning this entity
        rows = self._db.execute(
            "SELECT fact_id FROM atomic_facts WHERE profile_id = ? "
            "AND canonical_entities_json LIKE ?",
            (profile_id, f'%"{eid}"%'),
        )
        fact_ids = [dict(r)["fact_id"] for r in rows]
        for fid in fact_ids:
            self._db.delete_fact(fid)
        counts["facts"] = len(fact_ids)

        # Delete temporal events
        self._db.execute(
            "DELETE FROM temporal_events WHERE entity_id = ? AND profile_id = ?",
            (eid, profile_id),
        )

        # Delete entity profile
        self._db.execute(
            "DELETE FROM entity_profiles WHERE entity_id = ? AND profile_id = ?",
            (eid, profile_id),
        )

        # Delete aliases + entity (profile-scoped — entity_id is UUID-global but
        # keep the tenant predicate for consistent Art.17 isolation).
        self._db.execute(
            "DELETE FROM entity_aliases WHERE entity_id = ? AND profile_id = ?",
            (eid, profile_id))
        self._db.execute(
            "DELETE FROM canonical_entities WHERE entity_id = ? AND profile_id = ?",
            (eid, profile_id))
        counts["entity"] = 1

        logger.info("Entity erasure '%s' in '%s': %s", entity_name, profile_id, counts)
        return counts

    # -- Audit Trail -------------------------------------------------------

    def get_audit_trail(
        self, profile_id: str, limit: int = 100
    ) -> list[dict]:
        """Get compliance audit trail for a profile."""
        rows = self._db.execute(
            "SELECT * FROM compliance_audit WHERE profile_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [dict(r) for r in rows]

    def _audit(
        self, action: str, target_type: str, target_id: str, details: str,
        profile_id: str | None = None,
    ) -> None:
        """Log a compliance action."""
        from superlocalmemory.storage.models import _new_id
        pid = profile_id if profile_id is not None else target_id
        self._db.execute(
            "INSERT INTO compliance_audit "
            "(audit_id, profile_id, action, target_type, target_id, details, timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (_new_id(), pid, action, target_type, target_id, details, _now()),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()
