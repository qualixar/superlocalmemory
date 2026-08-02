# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import hashlib
import json
from typing import Any

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


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _placeholders(count: int) -> str:
    return ",".join("?" for _ in range(count))


class _FactScopedOwner:
    _name: str

    def __init__(self, db: Any) -> None:
        self._db = db

    @property
    def name(self) -> str:
        return self._name

    def _required(self, context: OperationContext) -> set[str]:
        return set(context.fact_ids)

    def _fingerprints(self, context: OperationContext) -> dict[str, str]:
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
        remaining = set(self._fingerprints(context)) & set(context.fact_ids)
        return OwnerErasureProof(
            owner=self._name,
            erased=not remaining,
            checksum=_scope_checksum(
                self._name, {fact_id: "erased" for fact_id in remaining}
            ),
            detail={} if result.ok else dict(result.detail),
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
        return dict(rows[0])["content"] if rows else None


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
            try:
                tokens = json.loads(record["tokens"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(tokens, list) or not tokens:
                continue
            if not all(isinstance(token, str) for token in tokens):
                continue
            stored[record["fact_id"]] = tokens
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
            if record.get("valid_from") is None:
                continue
            result[record["fact_id"]] = _fingerprint(
                "temporal",
                str(record["valid_from"]),
                str(record.get("valid_until") or ""),
            )
        return result

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
        return {dict(row)["fact_id"] for row in rows}

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
            embedding = record.get("embedding")
            if not embedding:
                continue
            result[record["fact_id"]] = _fingerprint("vector", str(embedding))
        return result

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


def build_transaction_service(engine: Any) -> MemoryTransactionService:
    db = engine._db
    retrieval = getattr(engine, "_retrieval_engine", None)
    owners = {
        "bm25": Bm25Owner(db, retrieval=retrieval),
        "temporal": TemporalOwner(db),
        "vector": VectorOwner(
            db,
            vector_store=getattr(engine, "_vector_store", None),
            ann_index=getattr(engine, "_ann_index", None),
        ),
    }
    return MemoryTransactionService(owners)


__all__ = [
    "REQUIRED_ADMISSION_OWNERS",
    "Bm25Owner",
    "TemporalOwner",
    "VectorOwner",
    "build_transaction_service",
]
