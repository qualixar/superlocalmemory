# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import json
from typing import Any

from superlocalmemory.core.transactions.erasure import ErasureService
from superlocalmemory.core.transactions.owners import (
    OperationContext,
    OwnerErasureProof,
    OwnerHealth,
    OwnerResult,
)
from superlocalmemory.core.transactions.service import MemoryTransactionService

REQUIRED_ADMISSION_OWNERS: tuple[str, ...] = ("bm25", "temporal", "vector")


def _scope_checksum(owner: str, fingerprints: dict[str, str]) -> str:
    parts = [f"{fact_id}={fingerprints[fact_id]}" for fact_id in sorted(fingerprints)]
    payload = owner + "|" + "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _erasure_checksum(
    owner: str, targets: tuple[str, ...], residue: set[str],
) -> str:
    payload = "\0".join([
        owner,
        "targets=" + ",".join(sorted(set(targets))),
        "residue=" + ",".join(sorted(residue)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


def _row_fact_id(row: Any) -> str | None:
    try:
        return dict(row)["fact_id"]
    except (KeyError, ValueError, TypeError):
        pass
    try:
        return row["fact_id"]
    except Exception:  # noqa: BLE001
        pass
    try:
        return row[0]
    except Exception:  # noqa: BLE001
        return None


def _db_table_exists(db: Any, name: str) -> bool:
    try:
        rows = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        )
    except Exception:  # noqa: BLE001
        return False
    return bool(list(rows))


def _present_fact_ids(rows: Any, targets: set[str]) -> set[str]:
    present: set[str] = set()
    unreadable = False
    for row in rows:
        fid = _row_fact_id(row)
        if fid is None:
            unreadable = True
            continue
        present.add(fid)
    if unreadable:
        return set(targets)
    return present & targets


class _FactScopedOwner:
    _name: str

    def __init__(self, db: Any) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return self._name

    def _required(self, context: OperationContext) -> set[str]:
        return set(context.fact_ids) - self._tombstoned(context)

    def _tombstoned(self, context: OperationContext) -> set[str]:
        if not context.fact_ids:
            return set()
        try:
            rows = self._db.execute(
                f"SELECT fact_id FROM projection_tombstones WHERE profile_id = ? "
                f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
                (context.profile_id, *context.fact_ids),
            )
        except Exception:
            return set()
        return {fid for fid in (_row_fact_id(row) for row in rows) if fid is not None}

    def _fingerprints(self, context: OperationContext) -> dict[str, str]:
        raise NotImplementedError

    def _physical_present(self, context: OperationContext) -> set[str]:
        raise NotImplementedError

    def _heal(self, context: OperationContext, fact_id: str) -> bool:
        return False

    def _remove(self, context: OperationContext, fact_id: str) -> None:
        raise NotImplementedError

    def _result(self, context: OperationContext) -> OwnerResult:
        required = self._required(context)
        fingerprints = {
            fact_id: fp
            for fact_id, fp in self._fingerprints(context).items()
            if fact_id in required
        }
        missing = required - set(fingerprints)
        return OwnerResult(
            owner=self._name,
            ok=not missing,
            checksum=_scope_checksum(self._name, fingerprints),
            detail={} if not missing else {"missing": sorted(missing)},
        )

    def verify(self, context: OperationContext) -> OwnerResult:
        return self._result(context)

    def prove_erased(self, context: OperationContext) -> OwnerErasureProof:
        residue = self._physical_present(context) & set(context.fact_ids)
        detail = {"residue": sorted(residue)} if residue else {}
        return OwnerErasureProof(
            owner=self._name,
            erased=not residue,
            checksum=_erasure_checksum(self._name, context.fact_ids, residue),
            detail=detail,
        )

    def apply(self, context: OperationContext) -> OwnerResult:
        required = self._required(context)
        current = set(self._fingerprints(context))
        errors: list[str] = []
        for fact_id in sorted(required - current):
            try:
                self._heal(context, fact_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{fact_id}: {type(exc).__name__}")
        result = self._result(context)
        if errors:
            return OwnerResult(
                owner=self._name,
                ok=result.ok,
                checksum=result.checksum,
                detail={**dict(result.detail), "errors": errors},
            )
        return result

    def compensate(self, context: OperationContext) -> OwnerResult:
        return self._delete_all(context, phase="compensate")

    def erase(self, context: OperationContext) -> OwnerErasureProof:
        result = self._delete_all(context, phase="erase")
        residue = self._physical_present(context) & set(context.fact_ids)
        detail: dict[str, Any] = {} if result.ok else dict(result.detail)
        if residue:
            detail = {**detail, "residue": sorted(residue)}
        return OwnerErasureProof(
            owner=self._name,
            erased=not residue,
            checksum=_erasure_checksum(self._name, context.fact_ids, residue),
            detail=detail,
        )

    def _delete_all(self, context: OperationContext, *, phase: str) -> OwnerResult:
        errors: list[str] = []
        for fact_id in context.fact_ids:
            try:
                self._remove(context, fact_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{fact_id}: {type(exc).__name__}")
        return OwnerResult(
            owner=self._name,
            ok=not errors,
            detail={} if not errors else {"phase": phase, "errors": errors},
        )

    def health(self) -> OwnerHealth:
        return OwnerHealth(owner=self._name, healthy=True)

    def _fact_content(self, context: OperationContext, fact_id: str) -> str | None:
        rows = list(self._db.execute(
            "SELECT content FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, context.profile_id),
        ))
        if not rows:
            return None
        try:
            return dict(rows[0])["content"]
        except (KeyError, ValueError, TypeError):
            try:
                return rows[0]["content"]
            except Exception:  # noqa: BLE001
                return None


class Bm25Owner(_FactScopedOwner):
    _name = "bm25"

    def __init__(self, db: Any, *, retrieval: Any = None) -> None:
        super().__init__(db)
        self._retrieval = retrieval

    def _fingerprints(self, context: OperationContext) -> dict[str, str]:
        if not context.fact_ids:
            return {}
        from superlocalmemory.retrieval.bm25_channel import tokenize

        rows = self._db.execute(
            f"SELECT fact_id, tokens FROM bm25_tokens WHERE profile_id = ? "
            f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
            (context.profile_id, *context.fact_ids),
        )
        stored: dict[str, list[str]] = {}
        for row in rows:
            record = dict(row)
            fid = record.get("fact_id")
            if fid is None:
                continue
            try:
                tokens = json.loads(record["tokens"])
            except (TypeError, KeyError, json.JSONDecodeError):
                continue
            if not isinstance(tokens, list) or not tokens:
                continue
            if not all(isinstance(token, str) for token in tokens):
                continue
            stored[fid] = tokens
        result: dict[str, str] = {}
        for fact_id, tokens in stored.items():
            content = self._fact_content(context, fact_id)
            if content is None:
                continue
            expected = sorted(tokenize(content))
            if not expected or sorted(tokens) != expected:
                continue
            result[fact_id] = _fingerprint(
                "bm25", json.dumps(expected, separators=(",", ":"))
            )
        return result

    def _physical_present(self, context: OperationContext) -> set[str]:
        if not context.fact_ids:
            return set()
        rows = self._db.execute(
            f"SELECT fact_id FROM bm25_tokens WHERE profile_id = ? "
            f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
            (context.profile_id, *context.fact_ids),
        )
        return _present_fact_ids(rows, set(context.fact_ids))

    def _heal(self, context: OperationContext, fact_id: str) -> bool:
        bm25 = getattr(self._retrieval, "_bm25", None)
        content = self._fact_content(context, fact_id)
        if bm25 is None or content is None:
            return False
        if hasattr(bm25, "update_fact"):
            bm25.update_fact(fact_id, content, context.profile_id)
            return True
        if hasattr(bm25, "add") and hasattr(bm25, "remove_fact"):
            bm25.remove_fact(fact_id)
            bm25.add(fact_id, content, context.profile_id)
            return True
        return False

    def _remove(self, context: OperationContext, fact_id: str) -> None:
        self._db.delete_bm25_tokens_for_fact(fact_id)
        bm25 = getattr(self._retrieval, "_bm25", None)
        if bm25 is not None and hasattr(bm25, "remove_fact"):
            bm25.remove_fact(fact_id)


class TemporalOwner(_FactScopedOwner):
    _name = "temporal"

    def _fingerprints(self, context: OperationContext) -> dict[str, str]:
        if not context.fact_ids:
            return {}
        rows = self._db.execute(
            f"SELECT fact_id, valid_from, valid_until FROM fact_temporal_validity "
            f"WHERE profile_id = ? AND fact_id IN ({_placeholders(len(context.fact_ids))})",
            (context.profile_id, *context.fact_ids),
        )
        result: dict[str, str] = {}
        for row in rows:
            record = dict(row)
            fid = record.get("fact_id")
            if fid is None or record.get("valid_from") is None:
                continue
            result[fid] = _fingerprint(
                "temporal",
                str(record["valid_from"]),
                str(record.get("valid_until") or ""),
            )
        return result

    def _physical_present(self, context: OperationContext) -> set[str]:
        if not context.fact_ids:
            return set()
        rows = self._db.execute(
            f"SELECT fact_id FROM fact_temporal_validity WHERE profile_id = ? "
            f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
            (context.profile_id, *context.fact_ids),
        )
        return _present_fact_ids(rows, set(context.fact_ids))

    def _heal(self, context: OperationContext, fact_id: str) -> bool:
        rows = self._db.execute(
            "SELECT created_at FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, context.profile_id),
        )
        if not rows:
            return False
        valid_from = dict(rows[0]).get("created_at")
        if not valid_from:
            return False
        self._db.store_temporal_validity(fact_id, context.profile_id, valid_from)
        return True

    def _remove(self, context: OperationContext, fact_id: str) -> None:
        self._db.delete_temporal_validity(fact_id)


class VectorOwner(_FactScopedOwner):
    _name = "vector"

    def __init__(
        self, db: Any, *, vector_store: Any = None, ann_index: Any = None,
    ) -> None:
        super().__init__(db)
        self._vector_store = vector_store
        self._ann_index = ann_index

    def _store_available(self) -> bool:
        return self._vector_store is not None and bool(
            getattr(self._vector_store, "available", False)
        )

    def _required(self, context: OperationContext) -> set[str]:
        if not context.fact_ids or not self._store_available():
            return set()
        rows = self._db.execute(
            f"SELECT fact_id FROM atomic_facts WHERE profile_id = ? "
            f"AND embedding IS NOT NULL AND embedding != '' "
            f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
            (context.profile_id, *context.fact_ids),
        )
        found = {fid for fid in (_row_fact_id(row) for row in rows) if fid is not None}
        return found - self._tombstoned(context)

    def _fingerprints(self, context: OperationContext) -> dict[str, str]:
        if not context.fact_ids or not self._store_available():
            return {}
        indexed = self._vector_store.indexed_fact_ids(context.profile_id)
        candidates = [fid for fid in context.fact_ids if fid in indexed]
        if not candidates:
            return {}
        rows = self._db.execute(
            f"SELECT fact_id, embedding FROM atomic_facts WHERE profile_id = ? "
            f"AND fact_id IN ({_placeholders(len(candidates))})",
            (context.profile_id, *candidates),
        )
        result: dict[str, str] = {}
        for row in rows:
            record = dict(row)
            fid = record.get("fact_id")
            embedding = record.get("embedding")
            if fid is None or not embedding:
                continue
            result[fid] = _fingerprint("vector", str(embedding))
        return result

    def _table_present(
        self, table: str, context: OperationContext, targets: set[str],
    ) -> set[str]:
        if not _db_table_exists(self._db, table):
            return set()
        try:
            rows = self._db.execute(
                f"SELECT fact_id FROM {table} WHERE profile_id = ? "
                f"AND fact_id IN ({_placeholders(len(context.fact_ids))})",
                (context.profile_id, *context.fact_ids),
            )
        except Exception:  # noqa: BLE001
            return set(targets)
        return _present_fact_ids(rows, targets)

    def _physical_present(self, context: OperationContext) -> set[str]:
        if not context.fact_ids:
            return set()
        targets = set(context.fact_ids)
        present: set[str] = set()
        present |= self._table_present("embedding_metadata", context, targets)
        present |= self._table_present("vector_row_map", context, targets)
        store = self._vector_store
        if store is not None and hasattr(store, "raw_vector_present"):
            for fid in targets - present:
                try:
                    if store.raw_vector_present(fid):
                        present.add(fid)
                except Exception:  # noqa: BLE001
                    present.add(fid)
        ann = self._ann_index
        if ann is not None and hasattr(ann, "contains"):
            for fid in targets - present:
                try:
                    if ann.contains(fid):
                        present.add(fid)
                except Exception:  # noqa: BLE001
                    present.add(fid)
        return present

    def _heal(self, context: OperationContext, fact_id: str) -> bool:
        if not self._store_available():
            return False
        rows = self._db.execute(
            "SELECT embedding FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
            (fact_id, context.profile_id),
        )
        if not rows:
            return False
        raw = dict(rows[0]).get("embedding")
        if not raw:
            return False
        try:
            embedding = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(embedding, list) or not embedding:
            return False
        ok = self._vector_store.upsert(
            fact_id=fact_id, profile_id=context.profile_id, embedding=embedding,
        )
        if ok and self._ann_index is not None and hasattr(self._ann_index, "add"):
            try:
                self._ann_index.add(fact_id, embedding)
            except Exception:  # noqa: BLE001
                pass
        return bool(ok)

    def _remove(self, context: OperationContext, fact_id: str) -> None:
        if self._vector_store is not None and hasattr(self._vector_store, "delete"):
            self._vector_store.delete(fact_id)
        else:
            self._db.execute(
                "DELETE FROM embedding_metadata WHERE fact_id = ?", (fact_id,)
            )
        if self._ann_index is not None and hasattr(self._ann_index, "remove"):
            self._ann_index.remove(fact_id)


def _admission_owners(engine: Any) -> dict[str, Any]:
    db = engine._db
    retrieval = getattr(engine, "_retrieval_engine", None)
    return {
        "bm25": Bm25Owner(db, retrieval=retrieval),
        "temporal": TemporalOwner(db),
        "vector": VectorOwner(
            db,
            vector_store=getattr(engine, "_vector_store", None),
            ann_index=getattr(engine, "_ann_index", None),
        ),
    }


def build_transaction_service(engine: Any) -> MemoryTransactionService:
    return MemoryTransactionService(_admission_owners(engine))


def build_erasure_service(engine: Any) -> ErasureService:
    audit_logger = _audit_chain_logger()
    return ErasureService(_admission_owners(engine), audit_logger=audit_logger)


def _audit_chain_logger() -> Any:
    def _log(event: dict[str, Any]) -> None:
        from superlocalmemory.compliance.audit import AuditChain
        from superlocalmemory.infra.data_root import state_path

        AuditChain(str(state_path("audit_chain.db"))).log(
            "projection_erase",
            agent_id=str(event.get("requested_by", "")),
            profile_id=str(event.get("profile_id", "")),
            content_hash=str(event.get("audit_hash", "")),
            metadata={
                "erasure_id": event.get("erasure_id", ""),
                "subject_type": event.get("subject_type", ""),
                "subject_id": event.get("subject_id", ""),
                "state": event.get("state", ""),
            },
        )

    return _log


__all__ = [
    "REQUIRED_ADMISSION_OWNERS",
    "Bm25Owner",
    "TemporalOwner",
    "VectorOwner",
    "build_erasure_service",
    "build_transaction_service",
]
