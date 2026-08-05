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

import logging
from datetime import UTC, datetime
from pathlib import Path

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
    # erased/exported wholesale.
    # `profiles` — the tenant record (handled separately, deleted last).
    # `erasure_receipts` — tamper-evident audit chain for Art.17 erasure events;
    #   must survive the profile wipe so operators can prove deletion occurred.
    _NON_MEMORY_SCOPED = frozenset({"profiles", "erasure_receipts"})

    def __init__(self, db, *, engine=None, data_root: str | Path | None = None) -> None:
        self._db = db
        self._engine = engine
        self._data_root = Path(data_root).resolve() if data_root is not None else None

    def _memory_has_siblings(self, memory_id: str, profile_id: str) -> bool:
        try:
            return bool(self._db.execute(
                "SELECT 1 FROM atomic_facts "
                "WHERE memory_id = ? AND profile_id = ? LIMIT 1",
                (memory_id, profile_id),
            ))
        except Exception:
            return True

    def _tombstone(self, fact_id: str, profile_id: str, memory_id: str | None) -> None:
        try:
            import time
            import uuid

            from superlocalmemory.core.transactions.erasure import write_tombstones

            write_tombstones(
                self._db, profile_id, (fact_id,), uuid.uuid4().hex,
                time.time(), memory_id,
            )
        except Exception:
            pass

    def _purge_fact_projections(self, fact_id: str, profile_id: str) -> None:
        try:
            self._db.delete_bm25_tokens_for_fact(fact_id)
        except Exception:
            pass
        engine = self._engine
        if engine is None:
            return
        store = getattr(engine, "_vector_store", None)
        ann = getattr(engine, "_ann_index", None)
        if store is not None and getattr(store, "available", False):
            try:
                store.delete(fact_id)
            except Exception:
                pass
        if ann is not None and hasattr(ann, "remove"):
            try:
                ann.remove(fact_id)
            except Exception:
                pass

    def _purge_vector_and_ann(self, profile_id: str) -> tuple[int, int]:
        engine = self._engine
        if engine is None:
            return 0, 0
        store = getattr(engine, "_vector_store", None)
        ann = getattr(engine, "_ann_index", None)

        purged = 0
        failures = 0

        try:
            db_fact_ids = [
                dict(r)["fact_id"]
                for r in self._db.execute(
                    "SELECT fact_id FROM atomic_facts WHERE profile_id = ?",
                    (profile_id,),
                )
            ]
        except Exception as exc:
            logger.warning("GDPR erase: fact_id enumeration failed: %s", exc)
            db_fact_ids = []
            failures += 1

        store_available = store is not None and getattr(store, "available", False)
        store_fact_ids: list[str] = []
        if store_available:
            try:
                store_fact_ids = list(store.indexed_fact_ids(profile_id))
            except Exception as exc:
                logger.warning("GDPR erase: vector enumeration failed: %s", exc)
                failures += 1
                store_fact_ids = list(db_fact_ids)
            for fid in store_fact_ids:
                try:
                    if store.delete(fid):
                        purged += 1
                    else:
                        failures += 1
                except Exception as exc:
                    logger.warning("GDPR erase: vector delete failed for %s: %s", fid, exc)
                    failures += 1
        else:
            # No usable vector backend: raw vec0/map payload cannot be removed.
            # Count residual raw vectors as failures so the receipt cannot claim
            # a complete erasure while physical vectors survive.
            residue = self._count_vector_residue(profile_id)
            if residue:
                failures += residue

        if ann is not None and hasattr(ann, "remove"):
            all_to_purge = set(store_fact_ids) | set(db_fact_ids)
            for fid in all_to_purge:
                try:
                    ann.remove(fid)
                except Exception as exc:
                    logger.warning("GDPR erase: ANN remove failed for %s: %s", fid, exc)

        return purged, failures

    def _count_vector_residue(self, profile_id: str) -> int:
        total = 0
        for table in ("vector_row_map", "embedding_metadata"):
            try:
                rows = self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {table} WHERE profile_id = ?",
                    (profile_id,),
                )
                total = max(total, int(dict(rows[0])["c"]) if rows else 0)
            except Exception:
                continue
        return total

    def _fact_vector_residue(self, profile_id: str, fact_ids: list[str]) -> int:
        if not fact_ids:
            return 0
        residue: set[str] = set()
        placeholders = ",".join("?" for _ in fact_ids)
        for table in ("vector_row_map", "embedding_metadata"):
            try:
                rows = self._db.execute(
                    f"SELECT fact_id FROM {table} "
                    f"WHERE profile_id = ? AND fact_id IN ({placeholders})",
                    (profile_id, *fact_ids),
                )
                residue |= {dict(r)["fact_id"] for r in rows}
            except Exception:
                continue
        return len(residue)

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

        counts: dict[str, int] = {}

        # 1) Durable, tamper-evident record FIRST — a HARD precondition
        #    (Art. 5(2) accountability). If it cannot be written we fail closed
        #    and delete nothing, so no erasure ever occurs without an
        #    accountability record.
        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path
            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase", agent_id="gdpr", profile_id=profile_id,
                metadata={"basis": "GDPR Art.17 right-to-erasure"},
            )
        except Exception as exc:
            logger.error(
                "GDPR erase ABORTED for %r: pre-deletion audit-chain log failed: %s",
                profile_id, exc,
            )
            counts["audit_request_failed"] = 1
            counts["erasure_aborted"] = 1
            return counts
        self._audit("delete", "profile", profile_id, "GDPR erasure request")
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

        # Purge context-cache entries BEFORE main-DB row deletions.
        #
        # Crash-recovery rationale: the cache and the main DB live in separate
        # SQLite files — they cannot share one ACID transaction.  Ordering the
        # cache purge first ensures that any crash between the two steps leaves
        # the profile record still present in the main DB, so a retry of
        # forget_profile re-runs the full sequence and completes safely.  The
        # reverse order (cache after main delete) would orphan cache PII in a
        # state that no retry can reach.
        #
        # The cache DB lives under the data root (same directory as the main DB)
        # or in an immediate subdirectory.  Scan both levels to cover the default
        # layout and any explicitly-namespaced cache dirs.
        #
        # Destructive sidecar erasure requires an authoritative root. Never
        # fall back to a process-global default, which might be another SLM
        # installation.
        data_root = self._data_root
        try:
            from superlocalmemory.core.context_cache import purge_profile_from_cache_db
            if data_root is None:
                db_path = getattr(self._db, "db_path", None)
                if db_path is not None:
                    data_root = Path(db_path).resolve().parent

            if data_root is None:
                logger.warning(
                    "GDPR erase: context-cache purge skipped for profile %r — "
                    "data root could not be resolved; pass data_root explicitly "
                    "for custom DB wrappers.",
                    profile_id,
                )

            if data_root is not None:
                cache_name = "active_brain_cache.db"
                candidates: list = [data_root / cache_name]
                try:
                    for child in data_root.iterdir():
                        if child.is_dir():
                            candidates.append(child / cache_name)
                except Exception:
                    pass
                cache_purged = 0
                for candidate in candidates:
                    cache_purged += purge_profile_from_cache_db(candidate, profile_id)
                if cache_purged:
                    counts["context_cache"] = cache_purged
        except Exception as exc:
            # Fail-closed: a context-cache purge failure must not be silently
            # tolerated — it can leave profile PII in the cache DB.
            logger.warning("GDPR erase: context-cache purge failed: %s", exc)
            counts["context_cache_failed"] = 1

        try:
            vector_purged, vector_failures = self._purge_vector_and_ann(profile_id)
            counts["vector_store"] = vector_purged
            if vector_failures:
                counts["vector_store_failures"] = vector_failures
        except Exception as exc:
            # Fail-closed: a top-level vector-purge exception (as opposed to the
            # per-fact failures returned in vector_failures) must set an explicit
            # marker, or erasure_complete could still report 1 despite the vector
            # projection never being purged.
            logger.warning("GDPR erase: vector purge failed: %s", exc)
            counts["vector_store_failures"] = counts.get("vector_store_failures", 0) or 1

        # Erasure receipt (P1-5) — route the profile wipe through ErasureService
        # so the receipt captures real per-owner proofs (not proofs:[]).
        #
        # erasure_receipts is in _NON_MEMORY_SCOPED so Pass 2 does NOT delete
        # the receipt — it survives as the tamper-evident Art.17 audit chain.
        # remove() + finalize() therefore run here, before Pass 2, while
        # atomic_facts is still queryable for embedding presence checks.
        #
        # Wrapped in try-except so a missing M033/M035 schema never blocks the
        # Art.17 right-to-erasure.
        import time as _time
        import uuid as _uuid

        _profile_fact_ids: tuple[str, ...] = ()
        try:
            _fact_rows = self._db.execute(
                "SELECT fact_id FROM atomic_facts WHERE profile_id = ?",
                (profile_id,),
            )
            _profile_fact_ids = tuple(sorted(
                dict(r)["fact_id"] for r in _fact_rows
                if dict(r).get("fact_id") is not None
            ))
        except Exception as exc:
            logger.warning("GDPR profile erase: fact_id scan failed: %s", exc)

        # Always write an erasure receipt — even for empty profiles (fact_ids=()).
        # Skipping the receipt for no-fact profiles left an Art.17 accountability
        # gap: a destructive wipe with no durable audit record.  ErasureService
        # handles empty fact_ids safely (all owners vacuously return erased=True).
        # If finalize() raises (e.g. signing-key unavailable), propagate — we must
        # not silently proceed with a wipe that has no accountability record.
        try:
            from superlocalmemory.core.transactions.concrete_owners import (
                build_erasure_service_for_db,
            )
            from superlocalmemory.core.transactions.owners import OperationContext

            _erasure_svc = build_erasure_service_for_db(self._db, self._engine)
            _ctx = OperationContext(
                operation_id=_uuid.uuid4().hex,
                profile_id=profile_id,
                subject_id=profile_id,
                fact_ids=_profile_fact_ids,
            )
            _remove_result = _erasure_svc.remove(self._db, _ctx)
            _receipt = _erasure_svc.finalize(
                self._db, _ctx,
                subject_type="profile",
                subject_id=profile_id,
                requested_by="gdpr",
                requested_at=_time.time(),
                remove_result=_remove_result,
            )
            if not _receipt.persisted:
                counts["receipt_persist_failed"] = 1
            if not _receipt.all_erased:
                counts["owner_erasure_incomplete"] = 1
        except Exception as exc:
            counts["receipt_error"] = str(exc)
            raise

        # Pass 2 — full-tenant wipe with FK enforcement OFF so table order is
        # irrelevant (every profile row in every table goes). FTS shadow rows
        # are still removed by the base-table delete triggers.
        try:
            self._db.execute("PRAGMA foreign_keys=OFF")
        except Exception:
            pass
        table_delete_failures: list[str] = []
        try:
            for table in tables:
                try:
                    self._db.execute(
                        f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,)
                    )
                except Exception as exc:  # pragma: no cover — defensive per-table
                    logger.warning("GDPR erase: delete %s failed: %s", table, exc)
                    table_delete_failures.append(table)
            # Delete the profile record itself.
            self._db.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
            counts["profiles"] = 1
        finally:
            try:
                self._db.execute("PRAGMA foreign_keys=ON")
            except Exception:
                pass
        if table_delete_failures:
            counts["table_delete_failures"] = len(table_delete_failures)

        # Erase the learning sidecar next to the active memory database.  A
        # custom SLM data root must never fall back to another installation's
        # DEFAULT_BASE_DIR: doing so can both miss the subject data and erase
        # unrelated learning state.
        try:
            from superlocalmemory.learning.database import LearningDatabase
            if data_root is None:
                raise RuntimeError("active data root could not be resolved")
            learning_db = LearningDatabase(data_root / "learning.db")
            learning_db.reset(profile_id)
            counts["learning_db"] = 1
        except Exception as exc:
            logger.warning("GDPR erase: learning-db reset failed: %s", exc)
            counts["learning_db_failed"] = 1

        # VACUUM to remove deleted data from physical file
        try:
            self._db.execute("VACUUM")
        except Exception:
            pass

        # Fail-closed completeness: re-count residue across the wiped tables and
        # surface an explicit erasure_complete flag so a partial wipe is reported
        # as failure rather than silent success.
        residue_rows = 0
        residue_recount_failed = False
        for table in tables:
            try:
                _r = self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {table} WHERE profile_id = ?",
                    (profile_id,),
                )
                residue_rows += int(dict(_r[0])["c"]) if _r else 0
            except Exception as exc:
                # Fail-closed: a residue re-count that cannot be performed is a
                # verification failure, not zero residue. We cannot certify the
                # table is clean, so erasure must not report complete.
                logger.warning(
                    "GDPR erase: residue re-count for %s failed: %s", table, exc
                )
                residue_recount_failed = True
        counts["residue_rows"] = residue_rows
        if residue_recount_failed:
            counts["residue_recount_failed"] = 1
        counts["erasure_complete"] = 1 if (
            residue_rows == 0
            and not residue_recount_failed
            and not table_delete_failures
            and not counts.get("learning_db_failed")
            and not counts.get("vector_store_failures")
            and not counts.get("context_cache_failed")
            and not counts.get("owner_erasure_incomplete")
        ) else 0

        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path
            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase_complete", agent_id="gdpr", profile_id=profile_id,
                metadata={
                    "basis": "GDPR Art.17 right-to-erasure",
                    "tables_erased": len(tables),
                    "vector_store_failures": counts.get("vector_store_failures", 0),
                },
            )
        except Exception as exc:
            logger.error("GDPR erase: completion audit-chain log failed: %s", exc)
            counts["audit_completion_failed"] = 1

        logger.info("GDPR erasure for '%s': %d tables, %s", profile_id, len(tables), counts)
        return counts

    def forget_entity(self, entity_name: str, profile_id: str) -> dict:
        """Delete all data related to a specific entity.

        Removes facts mentioning the entity, edges, temporal events,
        and the entity itself. For targeted erasure requests.
        """
        import time
        requested_at = time.time()
        audit_request_ok = True
        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path
            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase_entity", agent_id="gdpr", profile_id=profile_id,
                metadata={
                    "basis": "GDPR Art.17 right-to-erasure",
                    "entity": entity_name,
                },
            )
        except Exception as exc:
            logger.warning("GDPR entity erase: audit-chain log failed: %s", exc)
            audit_request_ok = False
        self._audit("delete", "entity", entity_name,
                     f"GDPR entity erasure in profile {profile_id}",
                     profile_id=profile_id)

        entity = self._db.get_entity_by_name(entity_name, profile_id)
        if entity is None:
            result: dict[str, object] = {"deleted": 0, "entity": entity_name, "found": False}
            if not audit_request_ok:
                result["audit_request_failed"] = 1
            return result

        eid = entity.entity_id
        counts: dict[str, int] = {}

        # Delete facts mentioning this entity — use ErasureService for projection
        # erasure so the receipt captures real per-owner proofs (not proofs:[]).
        rows = self._db.execute(
            "SELECT fact_id, memory_id FROM atomic_facts WHERE profile_id = ? "
            "AND canonical_entities_json LIKE ?",
            (profile_id, f'%"{eid}"%'),
        )
        targets = [(dict(r)["fact_id"], dict(r).get("memory_id")) for r in rows]
        target_fact_ids = [fid for fid, _ in targets]
        counts["facts"] = len(targets)

        if targets:
            import uuid as _uuid

            from superlocalmemory.core.transactions.concrete_owners import (
                build_erasure_service_for_db,
            )
            from superlocalmemory.core.transactions.owners import OperationContext

            erasure_svc = build_erasure_service_for_db(self._db, self._engine)
            op_id = _uuid.uuid4().hex
            ctx = OperationContext(
                operation_id=op_id,
                profile_id=profile_id,
                subject_id=entity_name,
                fact_ids=tuple(sorted(target_fact_ids)),
            )
            erasure_svc.remove(self._db, ctx)
            receipt = erasure_svc.finalize(
                self._db, ctx,
                subject_type="entity",
                subject_id=entity_name,
                requested_by="gdpr",
                requested_at=requested_at,
            )
            if not receipt.persisted:
                counts["receipt_persist_failed"] = 1
            if not receipt.all_erased:
                counts["vector_store_failures"] = sum(
                    1 for p in receipt.proofs if not p.erased
                )

        for fid, mid in targets:
            self._db.delete_fact(fid)
            if mid and not self._memory_has_siblings(mid, profile_id):
                try:
                    self._db.execute(
                        "DELETE FROM memories WHERE memory_id = ? AND profile_id = ?",
                        (mid, profile_id),
                    )
                except Exception:
                    pass

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
        if not audit_request_ok:
            counts["audit_request_failed"] = 1

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
