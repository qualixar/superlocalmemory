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
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# C1 — Backup residue obligations
# Imported lazily inside methods to avoid circular-import risk at module load.
# The sentinel guards against environments where infra.backup_obligations is
# unavailable (e.g. minimal test installs); compliance logic degrades safely.
_BACKUP_OBLIGATIONS_AVAILABLE: bool | None = None


def _get_backup_obligations_module():
    """Lazy import guard — returns module or None if unavailable."""
    global _BACKUP_OBLIGATIONS_AVAILABLE
    try:
        import superlocalmemory.infra.backup_obligations as _m
        _BACKUP_OBLIGATIONS_AVAILABLE = True
        return _m
    except Exception as exc:  # noqa: BLE001
        if _BACKUP_OBLIGATIONS_AVAILABLE is None:
            logger.warning("backup_obligations module unavailable: %s", exc)
        _BACKUP_OBLIGATIONS_AVAILABLE = False
        return None


def _retention_days_from_config() -> int:
    """Read the configured obligation retention window (default 90 days).

    Consumes ``SLMConfig.backup_retention_days`` or any attribute matching
    ('retention', 'retain') with numeric value on ``SLMConfig``.  The config
    field is owned by another agent; we consume it here and fall back to 90.
    """
    try:
        from superlocalmemory.core.config import SLMConfig
        cfg = SLMConfig()
        # Try the canonical field name first
        for attr in ("backup_retention_days", "obligation_retention_days",
                     "retention_window_days", "backup_obligation_retention_days"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        # Fallback: search nested sub-configs
        for attr in dir(cfg):
            if attr.startswith("_"):
                continue
            sub = getattr(cfg, attr, None)
            if not hasattr(sub, "__dict__") and not hasattr(sub, "__dataclass_fields__"):
                continue
            for sub_attr in dir(sub):
                if any(h in sub_attr.lower() for h in ("retention", "retain")):
                    v = getattr(sub, sub_attr, None)
                    if isinstance(v, int) and v > 0:
                        return v
    except Exception:  # noqa: BLE001
        pass
    return 90  # owner-set default

#: Rows fetched per page when an export reads a table in full.
_EXPORT_PAGE_ROWS = 1000


def _every_row(cursor: sqlite3.Cursor) -> list[dict]:
    """All rows of ``cursor`` as dicts, fetched a page at a time."""
    out: list[dict] = []
    while True:
        page = cursor.fetchmany(_EXPORT_PAGE_ROWS)
        if not page:
            return out
        out.extend(dict(row) for row in page)


# Friendly export keys → canonical table names (stable Art.20 export contract).
_EXPORT_ALIASES = {
    "facts": "atomic_facts",
    "entities": "canonical_entities",
    "edges": "graph_edges",
    "feedback": "feedback_records",
    "scenes": "memory_scenes",
}


def _unproject_entity(entity_id: str) -> bool:
    """Remove an entity node from the graph projection, if there is one.

    The projection is a separate storage engine, so deleting the row from
    SQLite reaches nothing there. An entity node left behind still carries the
    name it was created with, which is exactly the thing an erasure was asked
    to remove.

    Returns False when a projection exists and refused, so a caller that
    reports completeness can report this honestly. No projection at all is not
    a failure -- there is nothing to remove.
    """
    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator
    except Exception:  # noqa: BLE001
        return True
    try:
        orchestrator = get_orchestrator()
        graph = orchestrator.get_graph_backend() if orchestrator else None
        if graph is None:
            return True
        remove = getattr(graph, "remove_entity", None)
        if remove is None:
            return True
        remove(entity_id)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "the graph projection still holds entity %s (%s)", entity_id[:12], exc,
        )
        return False


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
            return bool(
                self._db.execute(
                    "SELECT 1 FROM atomic_facts WHERE memory_id = ? AND profile_id = ? LIMIT 1",
                    (memory_id, profile_id),
                )
            )
        except Exception:
            return True

    def _tombstone(self, fact_id: str, profile_id: str, memory_id: str | None) -> None:
        try:
            import time
            import uuid

            from superlocalmemory.core.transactions.erasure import write_tombstones

            write_tombstones(
                self._db,
                profile_id,
                (fact_id,),
                uuid.uuid4().hex,
                time.time(),
                memory_id,
            )
        except Exception:
            pass

    def _purge_graph_and_vector_projections(
        self, profile_id: str,
    ) -> tuple[int, int]:
        """Remove a tenant's facts from CozoDB and LanceDB. Returns (purged, failed).

        These projections live in their own storage engines — RocksDB under Cozo,
        Lance files under the vector store — so nothing about deleting rows from
        SQLite reaches them. A memory left in the graph after erasure is still
        reachable by recall, which makes the erasure receipt a false statement.

        A store with no projection open returns ``(0, 0)``: there is nothing to
        purge, which is not a failure. A projection that is open and refuses to
        delete IS a failure and is counted, so ``erasure_complete`` cannot come
        back true while a tenant's facts are still in the graph.
        """
        try:
            from superlocalmemory.core.backend_orchestrator import get_orchestrator
        except Exception:
            return 0, 0
        orchestrator = get_orchestrator()
        if orchestrator is None:
            return 0, 0
        graph = orchestrator.get_graph_backend()
        vector = orchestrator.get_vector_backend()
        if graph is None and vector is None:
            return 0, 0

        try:
            fact_ids = [
                dict(r)["fact_id"]
                for r in self._db.execute(
                    "SELECT fact_id FROM atomic_facts WHERE profile_id = ?",
                    (profile_id,),
                )
            ]
        except Exception as exc:
            logger.warning("GDPR erase: could not list facts to unproject: %s", exc)
            return 0, 1

        purged = failures = 0
        for fact_id in fact_ids:
            removed = True
            for backend, method in ((graph, "remove_fact"), (vector, "remove_vector")):
                if backend is None:
                    continue
                try:
                    getattr(backend, method)(fact_id)
                except Exception as exc:
                    logger.warning(
                        "GDPR erase: %s failed for %s: %s", method, fact_id[:12], exc,
                    )
                    removed = False
            if removed:
                purged += 1
            else:
                failures += 1
        return purged, failures

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
                for r in self._db.execute("SELECT name FROM sqlite_master WHERE type='table'")
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
            rows = self._db.execute("SELECT * FROM profiles WHERE profile_id = ?", (profile_id,))
            data["profile_record"] = [dict(r) for r in rows]
        except Exception:
            data["profile_record"] = []

        # C2 — include code_graph.db in Art.15 export (repo paths, file names,
        # and symbol names are identifying data in a work context).
        # Without an explicit root the files next to the memory database are
        # this installation's — the same rule erasure uses. Reading is safe;
        # leaving them out made the dashboard export silently incomplete.
        export_root = self._data_root
        if export_root is None:
            db_path = getattr(self._db, "db_path", None)
            if db_path is not None:
                export_root = Path(db_path).resolve().parent
        if export_root is not None:
            code_graph_data = self._export_code_graph(export_root)
            if code_graph_data is not None:
                data["code_graph"] = code_graph_data

            # What the system learned about how this person works: which
            # questions they asked, which answers they found useful, and the
            # weights derived from that. It was erased on request and could not
            # be asked for -- which makes an export that claims to be everything
            # untrue. It lives in a second file, so the table sweep above never
            # reached it.
            behaviour = self._export_learning_signals(export_root, profile_id)
            if behaviour is not None:
                data["learning_signals"] = behaviour

        # total_items counts the canonical (table-name) keys only, before
        # friendly aliases are added, so it is not double-counted.
        data["total_items"] = sum(len(v) for v in data.values() if isinstance(v, list))

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
            raise ValueError(
                "Cannot delete the default profile via GDPR erasure. Use profile deletion instead."
            )

        counts: dict[str, int] = {}

        # 1) Durable, tamper-evident record FIRST — a HARD precondition
        #    (Art. 5(2) accountability). If it cannot be written we fail closed
        #    and delete nothing, so no erasure ever occurs without an
        #    accountability record.
        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path

            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase",
                agent_id="gdpr",
                profile_id=profile_id,
                metadata={"basis": "GDPR Art.17 right-to-erasure"},
            )
        except Exception as exc:
            logger.error(
                "GDPR erase ABORTED for %r: pre-deletion audit-chain log failed: %s",
                profile_id,
                exc,
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

        # The graph and vector projections are separate storage engines, so a
        # profile wipe against SQLite does not reach them. They have to be
        # purged here, before Pass 2: that sweep finds a tenant's tables by
        # looking for a profile_id column, which includes projection_outbox — so
        # it would delete the very rows that carry the queued removals, leaving
        # the erased facts in the graph with nothing left to take them out.
        try:
            graph_purged, graph_failures = self._purge_graph_and_vector_projections(
                profile_id
            )
            if graph_purged:
                counts["graph_projection"] = graph_purged
            if graph_failures:
                counts["graph_projection_failures"] = graph_failures
        except Exception as exc:
            logger.warning("GDPR erase: graph projection purge failed: %s", exc)
            counts["graph_projection_failures"] = (
                counts.get("graph_projection_failures", 0) or 1
            )

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
            _profile_fact_ids = tuple(
                sorted(dict(r)["fact_id"] for r in _fact_rows if dict(r).get("fact_id") is not None)
            )
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

            _erasure_svc = build_erasure_service_for_db(self._db, self._engine, data_root)
            _ctx = OperationContext(
                operation_id=_uuid.uuid4().hex,
                profile_id=profile_id,
                subject_id=profile_id,
                fact_ids=_profile_fact_ids,
            )
            _remove_result = _erasure_svc.remove(self._db, _ctx)
            _receipt = _erasure_svc.finalize(
                self._db,
                _ctx,
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

        from superlocalmemory.media.erasure import erase_profile_into  # images and their cached text

        erase_profile_into(counts, data_root, profile_id)

        # C2 — erase this profile's share of code_graph.db before the main
        # profile rows. How much of it is "this profile's share" depends on
        # whether anybody else is in the store; see _erase_code_graph. A failure
        # here does not abort the rest of the erasure, but it does block the
        # completeness claim below.
        if data_root is not None:
            code_graph_result = self._erase_code_graph(
                data_root,
                profile_id=profile_id,
                sole_profile=self._is_sole_profile(profile_id),
            )
            counts["code_graph"] = code_graph_result.get("rows_deleted", 0)
            counts["code_graph_scope"] = code_graph_result.get("scope", "")
            if code_graph_result.get("retained_reason"):
                counts["code_graph_retained_reason"] = code_graph_result[
                    "retained_reason"
                ]
            if code_graph_result.get("error"):
                counts["code_graph_failed"] = 1

        # Purge the learning sidecar *before* removing memory/profile rows. A
        # learning failure is retryable and must leave the profile intact; the
        # former best-effort-after-delete ordering could orphan receipts.
        if data_root is None:
            # Compatibility for third-party legacy wrappers that expose no
            # durable path. We cannot safely guess another installation's
            # sidecar. Native v4.0.2 runtime objects always provide the root.
            logger.warning(
                "GDPR erase: learning receipt purge skipped for profile %r — "
                "data root could not be resolved",
                profile_id,
            )
            counts["learning_db_skipped"] = 1
        else:
            try:
                from superlocalmemory.learning.database import LearningDatabase

                learning_db = LearningDatabase(data_root / "learning.db")
                learning_db.reset(profile_id)
                counts["learning_db"] = 1
                from superlocalmemory.core.answer_check_history_store import (
                    erase_profile_everywhere,
                )
                counts["answer_check_history"] = erase_profile_everywhere(
                    data_root / "learning.db", profile_id)
                # Saved views hold the person's own query text (4.1.21, #113).
                from superlocalmemory.views import ViewStore

                counts["saved_views"] = ViewStore(
                    data_root / "learning.db").erase_profile(profile_id)
            except Exception as exc:
                logger.warning("GDPR erase: learning-db reset failed: %s", exc)
                counts["learning_db_failed"] = 1
                raise RuntimeError(
                    "learning receipt purge failed; profile deletion was not started"
                ) from exc

        # Tables keyed on a fact rather than on a profile are invisible to the
        # profile-scoped sweep below, which discovers tables by looking for a
        # profile_id column. They have to go FIRST, while the facts that name
        # them still exist to be joined against.
        self._erase_fact_keyed_tables(profile_id, counts)

        # Correction history does not block erasure (compliance/erased_cases.py):
        # the ledger refers to facts ON DELETE RESTRICT, which no table order
        # and no deferred check can get past, so its cases go first.
        from superlocalmemory.compliance.erased_cases import erase_profile_cases

        counts["correction_cases_erased"] = erase_profile_cases(
            self._db, profile_id, erasure_id=_uuid.uuid4().hex)

        # Pass 2 — full-tenant wipe (every profile row in every table goes).
        # Each statement runs on its own connection with foreign keys ON (a
        # PRAGMA here would only reach a throwaway connection), so cascades
        # apply. FTS shadow rows are removed by the base-table delete triggers.
        table_delete_failures: list[str] = []
        for table in tables:
            try:
                self._db.execute(f"DELETE FROM {table} WHERE profile_id = ?", (profile_id,))
            except Exception as exc:  # pragma: no cover — defensive per-table
                logger.warning("GDPR erase: delete %s failed: %s", table, exc)
                table_delete_failures.append(table)
        # Delete the profile record itself.
        self._db.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
        counts["profiles"] = 1
        if table_delete_failures:
            counts["table_delete_failures"] = len(table_delete_failures)

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
                logger.warning("GDPR erase: residue re-count for %s failed: %s", table, exc)
                residue_recount_failed = True
        counts["residue_rows"] = residue_rows
        if residue_recount_failed:
            counts["residue_recount_failed"] = 1

        # I7 — post-erasure residue sweep: FTS shadow tables + WAL sanity.
        # This makes I7 enforceable rather than aspirational.
        self._scan_fts_residue(profile_id, tables, counts)
        self._scan_wal_residue(counts)

        # C1 — record outstanding obligations against backup snapshots.
        # Done AFTER the main-DB residue scan so counts reflect live-store state
        # before we tally the backups outstanding obligation count.
        # Fail-closed: if the scan itself errors, set backup_scan_failed so
        # completeness cannot be claimed.
        backup_obligations_pending = 0
        if data_root is not None:
            try:
                backup_obligations_pending = self._record_backup_obligations(
                    data_root=data_root,
                    profile_id=profile_id,
                    erasure_id=_uuid.uuid4().hex,  # unique id for this obligation batch
                    counts=counts,
                )
                counts["backup_obligations_pending"] = backup_obligations_pending
            except Exception as exc:
                logger.error(
                    "GDPR erase: backup obligation recording FAILED: %s — "
                    "setting backup_scan_failed to block completeness",
                    exc,
                )
                counts["backup_scan_failed"] = 1
        else:
            # Cannot scan backups without data_root — treat as outstanding
            # obligation so completeness is blocked.
            counts["backup_obligations_pending"] = 0  # unknown but not confirmed clean
            # We won't block completeness when data_root is unknown (legacy wrapper)
            # but we do log the gap.
            logger.warning(
                "GDPR erase: backup obligation scan skipped — data_root unknown. "
                "Backup snapshots may still contain the erased profile's data."
            )

        # In-process residue. The erased profile's recently-shown memories are
        # held in a per-session working set that biases ranking; those are the
        # subject's data too, and a later session reusing one of their session
        # ids would otherwise inherit the bias.
        #
        # Runs after every DB delete — an in-memory dict cannot participate in a
        # rollback, so clearing it earlier would be unrecoverable if the wipe
        # then failed — but BEFORE the completeness verdict below, and counted by
        # it. Computing the verdict first would let this fail while the API
        # reported a complete erasure, which is the one thing an Art.17 receipt
        # must never do.
        try:
            from superlocalmemory.core.working_memory import discard_profile

            counts["working_sets"] = discard_profile(profile_id)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("GDPR erase: working-set discard failed: %s", exc)
            counts["working_sets_failed"] = 1

        counts["erasure_complete"] = (
            1
            if (
                residue_rows == 0
                and not residue_recount_failed
                and not table_delete_failures
                and not counts.get("learning_db_failed")
                and not counts.get("learning_db_skipped")
                and not counts.get("vector_store_failures")
                # A fact still in the graph is still recallable, so an erasure
                # that could not reach the projection is not complete.
                and not counts.get("graph_projection_failures")
                and not counts.get("context_cache_failed")
                and not counts.get("media_failed")
                and not counts.get("owner_erasure_incomplete")
                and not counts.get("backup_obligations_pending")
                and not counts.get("backup_scan_failed")
                and not counts.get("fts_residue_rows")
                and not counts.get("working_sets_failed")
                and not counts.get("code_graph_failed")
                and not any(
                    counts.get(f"{t}_failed") for t, _ in self._FACT_KEYED_TABLES
                )
            )
            else 0
        )

        # Whether the data is gone and whether we can PROVE it is gone are two
        # different questions, and one answer cannot carry both. Article 5(2) is
        # accountability: an erasure whose tamper-evident receipt was not
        # written really did delete the rows, and really cannot be demonstrated
        # afterwards. Folding that into erasure_complete would report an
        # erasure that happened as one that did not; leaving it out entirely —
        # which is what happened until now — lets a caller reading one field
        # believe it covers both.
        counts["erasure_provable"] = (
            1
            if (
                counts["erasure_complete"] == 1
                and not counts.get("receipt_persist_failed")
                and not counts.get("receipt_error")
            )
            else 0
        )

        try:
            from superlocalmemory.compliance.audit import AuditChain
            from superlocalmemory.infra.data_root import state_path

            AuditChain(str(state_path("audit_chain.db"))).log(
                "gdpr_erase_complete",
                agent_id="gdpr",
                profile_id=profile_id,
                metadata={
                    "basis": "GDPR Art.17 right-to-erasure",
                    "tables_erased": len(tables),
                    "vector_store_failures": counts.get("vector_store_failures", 0),
                    "backup_obligations_pending": backup_obligations_pending,
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
                "gdpr_erase_entity",
                agent_id="gdpr",
                profile_id=profile_id,
                metadata={
                    "basis": "GDPR Art.17 right-to-erasure",
                    "entity": entity_name,
                },
            )
        except Exception as exc:
            logger.warning("GDPR entity erase: audit-chain log failed: %s", exc)
            audit_request_ok = False
        self._audit(
            "delete",
            "entity",
            entity_name,
            f"GDPR entity erasure in profile {profile_id}",
            profile_id=profile_id,
        )

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
        from superlocalmemory.core import erasure_scrub  # words outside projections
        named = {fid: erasure_scrub.entities_of(self._db, profile_id, fid) for fid in target_fact_ids}
        erasure_scrub.prepare(self._db)

        if targets:
            import uuid as _uuid

            from superlocalmemory.core.transactions.concrete_owners import (
                build_erasure_service_for_db,
            )
            from superlocalmemory.core.transactions.owners import OperationContext

            erasure_svc = build_erasure_service_for_db(self._db, self._engine, self._data_root)
            op_id = _uuid.uuid4().hex
            ctx = OperationContext(
                operation_id=op_id,
                profile_id=profile_id,
                subject_id=entity_name,
                fact_ids=tuple(sorted(target_fact_ids)),
            )
            erasure_svc.remove(self._db, ctx)
            receipt = erasure_svc.finalize(
                self._db,
                ctx,
                subject_type="entity",
                subject_id=entity_name,
                requested_by="gdpr",
                requested_at=requested_at,
            )
            if not receipt.persisted:
                counts["receipt_persist_failed"] = 1
            if not receipt.all_erased:
                counts["vector_store_failures"] = sum(1 for p in receipt.proofs if not p.erased)

        # The same residue the profile wipe had: the search-expansion index is
        # keyed on a fact, not a profile, and ``delete_fact`` does not touch it.
        # Erasing an entity left the alternate keys of its memories behind, and
        # a search could still match them.
        self._erase_fact_keyed_tables_for(
            [fid for fid, _mid in targets], counts,
        )

        if targets:  # correction history does not block erasure (compliance/erased_cases.py)
            from superlocalmemory.compliance.erased_cases import delete_erased_facts
            delete_erased_facts(self._db, targets, profile_id, erasure_id=op_id,
                                has_siblings=self._memory_has_siblings, counts=counts)

        try:
            for fid in target_fact_ids:
                erasure_scrub.scrub(self._db, profile_id, fid, named[fid])
        except Exception as exc:
            logger.error("GDPR entity erase: text scrub failed: %s", exc)
            counts["text_scrub_failed"] = 1

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
            "DELETE FROM entity_aliases WHERE entity_id = ? AND profile_id = ?", (eid, profile_id)
        )
        self._db.execute(
            "DELETE FROM canonical_entities WHERE entity_id = ? AND profile_id = ?",
            (eid, profile_id),
        )
        # The graph holds its own copy of this node, with the name in it.
        # Deleting the row above does not reach it.
        if not _unproject_entity(eid):
            counts["projection_failed"] = 1
        counts["entity"] = 1
        if not audit_request_ok:
            counts["audit_request_failed"] = 1

        # The same two questions the profile path answers. Their absence here
        # meant a caller could not tell a complete entity erasure from a partial
        # one at all — it just got a dict of counts.
        counts["erasure_complete"] = (
            0
            if any(
                counts.get(marker)
                for marker in (
                    "vector_store_failures",
                    "audit_request_failed", "text_scrub_failed", "projection_failed",
                    *(f"{table}_failed" for table, _ in self._FACT_KEYED_TABLES),
                )
            )
            else 1
        )
        counts["erasure_provable"] = (
            1
            if counts["erasure_complete"] == 1
            and not counts.get("receipt_persist_failed")
            and not counts.get("audit_completion_failed")
            else 0
        )

        logger.info("Entity erasure '%s' in '%s': %s", entity_name, profile_id, counts)
        return counts

    # -- C2: code_graph helpers --------------------------------------------

    #: Tables that hold a person's text but are keyed on a fact, not a profile.
    #: The erasure sweep finds tables by looking for a profile_id column, so
    #: these are invisible to it — the search-expansion index kept a copy of a
    #: memory's alternate keys after every trace of the memory itself was gone.
    #: Erasing them needs a join, and the join needs the facts to still exist,
    #: so it runs before the sweep rather than after.
    _FACT_KEYED_TABLES: tuple[tuple[str, str], ...] = (
        ("fact_expansion_fts", "fact_id"),
    )

    def _erase_fact_keyed_tables_for(
        self, fact_ids: list[str], counts: dict,
    ) -> None:
        """Erase fact-keyed rows for an explicit list of facts.

        The profile wipe derives the list from a profile; a targeted entity
        erasure already knows which facts it is removing. Both need the same
        tables cleared, so they share the loop rather than one of them
        forgetting — which is exactly what happened.
        """
        if not fact_ids:
            return
        for table, column in self._FACT_KEYED_TABLES:
            try:
                exists = self._db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = ?", (table,)
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("GDPR erase: cannot look for %s: %s", table, exc)
                counts[f"{table}_failed"] = 1
                continue
            if not exists:
                continue
            removed = 0
            try:
                for start in range(0, len(fact_ids), 500):
                    chunk = fact_ids[start:start + 500]
                    placeholders = ",".join("?" * len(chunk))
                    present = self._db.execute(
                        f"SELECT COUNT(*) AS c FROM {table} "
                        f"WHERE {column} IN ({placeholders})",
                        tuple(chunk),
                    )
                    removed += int(dict(present[0])["c"]) if present else 0
                    self._db.execute(
                        f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
                        tuple(chunk),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("GDPR erase: delete from %s failed: %s", table, exc)
                counts[f"{table}_failed"] = 1
                continue
            counts[table] = counts.get(table, 0) + removed

    def _erase_fact_keyed_tables(self, profile_id: str, counts: dict) -> None:
        """Delete rows that name this profile's facts but not the profile.

        Delegates rather than repeating the loop. It WAS a second copy, and the
        commit that introduced the shared helper claimed otherwise — which is
        exactly the failure the helper existed to prevent: the entity path could
        have been fixed with the profile path left untouched, and nothing would
        have noticed.
        """
        fact_ids = self._fact_ids_for(profile_id)
        if fact_ids is None:
            # Could not find out what to erase. Say so; do not report zero.
            for table, _column in self._FACT_KEYED_TABLES:
                counts[f"{table}_failed"] = 1
            return
        self._erase_fact_keyed_tables_for(fact_ids, counts)

    def _is_sole_profile(self, profile_id: str) -> bool:
        """Whether this profile is the only one in the store.

        Decides how much of the shared code graph an erasure may take. Errs
        toward FALSE — the narrower erasure — because failing to answer is not
        a reason to delete somebody else's records.
        """
        try:
            rows = self._db.execute("SELECT profile_id FROM profiles")
        except Exception as exc:  # noqa: BLE001 - reported by erring narrow
            logger.warning(
                "GDPR erase: could not count profiles (%s); erasing only this "
                "profile's own rows from the code graph", exc,
            )
            return False
        found = set()
        for row in rows:
            try:
                found.add(str(dict(row)["profile_id"]))
            except Exception:  # noqa: BLE001 - row shape varies by driver
                found.add(str(row[0]))
        return found <= {profile_id}

    def _fact_ids_for(self, profile_id: str) -> list[str] | None:
        """Every fact id belonging to this profile, for cross-database joins.

        Returns None when the listing FAILED, which is not the same as a
        profile with no facts. Collapsing the two is how a locked database
        turned into "nothing to erase" and then into a receipt saying complete.
        """
        try:
            rows = self._db.execute(
                "SELECT fact_id FROM atomic_facts WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GDPR erase: could not list facts for %r: %s", profile_id, exc)
            return None
        out: list[str] = []
        for row in rows:
            try:
                out.append(str(dict(row)["fact_id"]))
            except Exception:  # noqa: BLE001
                out.append(str(row[0]))
        return out

    def _erase_code_graph(
        self, data_root: Path, *, profile_id: str = "", sole_profile: bool = True,
    ) -> dict:
        """Erase this profile's share of the live code_graph.db (C2 — Art.17).

        The graph carries repository paths, file names and symbol names, and no
        table in it has a profile_id. When the store holds ONE profile the whole
        graph belongs to that person and wiping it is exactly right.

        When it holds more than one, wiping it destroys the other people's data
        too — and an erasure request that erases a second data subject is itself
        a breach, not an over-achievement. So in that case only the rows that
        are unambiguously this profile's are removed: ``code_memory_links``
        joins a code node to an SLM fact, and facts carry a profile. The graph
        of the source code stays, because it describes a repository rather than
        a person, and the receipt says plainly that it was left.

        Fail-open: an error is recorded in the returned dict so the caller can
        surface it, but it does NOT abort the rest of the erasure.
        """
        result: dict = {"rows_deleted": 0}
        code_graph_path = data_root / "code_graph.db"
        if not code_graph_path.exists():
            return result
        try:
            conn = sqlite3.connect(str(code_graph_path))
            conn.isolation_level = None  # autocommit so VACUUM can run
            try:
                conn.execute("PRAGMA foreign_keys=OFF")
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                ]
                total = 0
                conn.execute("BEGIN")
                if sole_profile:
                    for tbl in tables:
                        # Skip FTS virtual-table shadow files — deleting base rows handles them
                        if tbl.endswith((
                            "_fts", "_fts_data", "_fts_idx",
                            "_fts_content", "_fts_docsize", "_fts_config",
                        )):
                            continue
                        cur = conn.execute(f"DELETE FROM {tbl}")  # noqa: S608
                        total += cur.rowcount
                    result["scope"] = "whole_graph"
                else:
                    result["scope"] = "links_only"
                    result["retained_reason"] = (
                        "another profile shares this graph; the source-code "
                        "structure describes a repository, not this person, and "
                        "deleting it would erase another data subject's records"
                    )
                    owned = self._fact_ids_for(profile_id) if profile_id else []
                    if owned is None:
                        # Not knowing which links are this person's is a failure
                        # to erase, not an empty erasure.
                        raise sqlite3.OperationalError(
                            "could not list this profile's facts, so its code "
                            "links cannot be identified"
                        )
                    if "code_memory_links" in tables and owned:
                        cur = conn.execute(
                            "DELETE FROM code_memory_links WHERE slm_fact_id IN "
                            "(SELECT value FROM json_each(?))",
                            (json.dumps(owned),),
                        )
                        total += cur.rowcount
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("COMMIT")
                # VACUUM must run outside any transaction (autocommit mode required)
                conn.execute("VACUUM")
                result["rows_deleted"] = total
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GDPR erase: code_graph.db wipe failed: %s", exc)
            result["error"] = str(exc)
        return result

    def _export_code_graph(self, data_root: Path) -> dict | None:
        """Read code_graph.db for Art.15 export (C2).

        Returns a dict keyed by table name whose values are lists of row dicts,
        or None if the file does not exist or cannot be read.
        """
        code_graph_path = data_root / "code_graph.db"
        if not code_graph_path.exists():
            return None
        export: dict = {}
        try:
            uri = f"file:{code_graph_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                tables = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                ]
                for tbl in tables:
                    if tbl.endswith((
                        "_fts", "_fts_data", "_fts_idx",
                        "_fts_content", "_fts_docsize", "_fts_config",
                    )):
                        continue
                    try:
                        # Every row, read a page at a time: a fixed LIMIT here
                        # once cut a large repository's graph at 10,000 rows
                        # and the export said nothing.
                        export[tbl] = _every_row(
                            conn.execute(f"SELECT * FROM {tbl}")  # noqa: S608
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("GDPR export: code_graph table %s failed: %s", tbl, exc)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GDPR export: code_graph.db read failed: %s", exc)
            return None
        return export if export else None

    def _export_learning_signals(
        self, data_root: Path, profile_id: str
    ) -> dict | None:
        """Read this workspace's rows out of ``learning.db`` for an export.

        Everything the system inferred about how somebody works is personal
        data about them, and an export that leaves it out is not the export it
        says it is. Erasure already covers this file; asking for it did not.

        Scoped by ``profile_id`` wherever the table has one, and skipped where
        it does not -- a table with no workspace column holds nothing that can
        be attributed to one workspace, and returning it would export another
        workspace's rows into this person's file.

        Every row, never a sample: a fixed ``LIMIT`` here once returned the
        oldest 10,000 Answer Check history rows of a profile holding more and
        said nothing. Rows are read in pages so the database is not asked to
        materialise the whole table at once.
        """
        learning_path = data_root / "learning.db"
        if not learning_path.exists():
            return None
        export: dict = {}
        try:
            conn = sqlite3.connect(
                f"file:{learning_path}?mode=ro", uri=True, timeout=5,
            )
            conn.row_factory = sqlite3.Row
            try:
                tables = [
                    row[0] for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                ]
                for table in tables:
                    if table.endswith((
                        "_fts", "_fts_data", "_fts_idx",
                        "_fts_content", "_fts_docsize", "_fts_config",
                    )):
                        continue
                    try:
                        columns = {
                            row[1] for row in
                            conn.execute(f"PRAGMA table_info({table})")  # noqa: S608
                        }
                        if "profile_id" not in columns:
                            continue
                        rows = _every_row(conn.execute(
                            f"SELECT * FROM {table} WHERE profile_id = ?",  # noqa: S608
                            (profile_id,),
                        ))
                        if rows:
                            export[table] = rows
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "export: learning table %s skipped: %s", table, exc,
                        )
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("export: learning.db could not be read: %s", exc)
            return None
        return export or None

    # -- C1: backup obligation helpers -------------------------------------

    def _record_backup_obligations(
        self,
        data_root: Path,
        profile_id: str,
        erasure_id: str,
        counts: dict,
    ) -> int:
        """Scan backup snapshots and record any that contain *profile_id* data.

        Returns the total count of pending obligations after recording
        (including those created in prior erasure passes for the same profile).
        Fail-closed: any unhandled exception propagates to the caller who sets
        ``backup_scan_failed`` to block the completeness claim.
        """
        bom = _get_backup_obligations_module()
        if bom is None:
            logger.warning(
                "GDPR erase: backup_obligations module unavailable — "
                "backup residue will not be tracked for profile %r",
                profile_id,
            )
            # Cannot track → treat as pending so completeness is blocked.
            return 1

        backup_dir = data_root / "backups"
        retention_days = _retention_days_from_config()
        store = bom.BackupObligationStore(data_root)

        # Scan all backup snapshots for this profile's data.
        try:
            hits = bom.scan_backup_snapshots_for_profile(backup_dir, profile_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "GDPR erase: backup snapshot scan raised: %s — "
                "failing closed to block completeness",
                exc,
            )
            counts["backup_scan_error"] = str(exc)
            raise  # propagate so caller sets backup_scan_failed

        snapshots_with_data = len(hits)
        counts["backup_snapshots_scanned"] = snapshots_with_data
        recorded = 0
        for snap_path, snap_epoch in hits:
            try:
                store.record(
                    profile_id=profile_id,
                    erasure_id=erasure_id,
                    snapshot_path=snap_path,
                    snapshot_epoch=snap_epoch,
                    retention_days=retention_days,
                )
                recorded += 1
            except Exception as exc:  # noqa: BLE001
                # Recording failure for a single snapshot must not silently
                # skip the obligation — log and count so completeness is blocked.
                logger.error(
                    "GDPR erase: failed to record obligation for snapshot %r: %s",
                    snap_path, exc,
                )
                counts["backup_record_errors"] = counts.get("backup_record_errors", 0) + 1
                # Still include in pending count (fail-closed).
                recorded += 1

        # A restore's leftovers hold the same words: the deltas (memories written
        # after a copy, verbatim) lose this profile's rows now, and every
        # pre-restore copy that still holds it is registered like a backup.
        # Raises on failure, which blocks the completeness claim (fail-closed).
        from superlocalmemory.storage._restore_retention import cover_profile_erasure

        covered = cover_profile_erasure(data_root, profile_id, erasure_id=erasure_id,
                                        store=store, retention_days=retention_days)
        counts.update(covered)
        recorded += covered["pre_restore_copies_registered"]

        counts["backup_obligations_recorded"] = recorded
        # Return the authoritative pending count (includes obligations from prior
        # erasure passes for the same profile that were not yet discharged).
        return store.count_pending(profile_id)

    # -- I7: post-erasure residue scanner ----------------------------------

    def _scan_fts_residue(
        self,
        profile_id: str,
        tables: list[str],
        counts: dict,
    ) -> None:
        """Sweep FTS5 shadow tables for orphaned rowids after main-table erasure.

        FTS5 tables maintain several shadow tables (``_data``, ``_idx``,
        ``_content``, ``_docsize``, ``_config``).  A correct DELETE + VACUUM
        cycle via FTS5's ``content=`` mechanism will purge shadow rows
        automatically.  This sweep cross-checks: if any main memory/facts table
        is empty for the profile yet the corresponding FTS ``_content`` shadow
        still has rows, that is residue.  Updates ``counts["fts_residue_rows"]``.
        """
        fts_residue = 0
        try:
            all_tables = {
                row[0]
                for row in self._db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("GDPR I7: FTS shadow table listing failed: %s", exc)
            return

        for tbl in tables:
            content_shadow = f"{tbl}_fts_content"
            if content_shadow not in all_tables:
                continue
            try:
                # _fts_content stores one row per indexed row.  After erasure
                # the content rows should be zero for this profile.  We cannot
                # filter by profile_id directly (FTS content is a rowid join),
                # so we count ALL content rows and compare with the main table
                # row count for this profile (should both be 0).
                main_count_rows = self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {tbl} WHERE profile_id = ?",  # noqa: S608
                    (profile_id,),
                )
                main_count = int(dict(main_count_rows[0])["c"]) if main_count_rows else 0
                if main_count > 0:
                    # Main table still has rows — not an FTS-shadow issue, already
                    # caught by the main residue recount.
                    continue
                fts_rows = self._db.execute(
                    f"SELECT COUNT(*) AS c FROM {content_shadow}"  # noqa: S608
                )
                fts_count = int(dict(fts_rows[0])["c"]) if fts_rows else 0
                if fts_count > 0:
                    logger.warning(
                        "GDPR I7: FTS shadow %s has %d rows after profile %r erasure",
                        content_shadow, fts_count, profile_id,
                    )
                    fts_residue += fts_count
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GDPR I7: FTS shadow scan for %s failed: %s", content_shadow, exc
                )
        if fts_residue:
            counts["fts_residue_rows"] = fts_residue

    def _scan_wal_residue(self, counts: dict) -> None:
        """Trigger a WAL checkpoint so the WAL does not re-introduce erased data.

        After VACUUM the WAL should already be flushed, but an explicit
        PRAGMA wal_checkpoint(TRUNCATE) ensures the WAL file is zeroed and
        cannot carry deleted pages forward into a subsequent read.
        """
        try:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "GDPR I7: WAL checkpoint failed: %s — WAL may retain erased pages",
                exc,
            )
            counts["wal_checkpoint_failed"] = 1

    # -- Audit Trail -------------------------------------------------------

    def get_audit_trail(self, profile_id: str, limit: int = 100) -> list[dict]:
        """Get compliance audit trail for a profile."""
        rows = self._db.execute(
            "SELECT * FROM compliance_audit WHERE profile_id = ? ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [dict(r) for r in rows]

    def _audit(
        self,
        action: str,
        target_type: str,
        target_id: str,
        details: str,
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
