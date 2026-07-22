#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - Semantic Kernel VectorStoreCollection backend.

Implements Semantic Kernel's ``VectorStoreCollection`` (the post-1.34 vector
store abstraction that replaced the deprecated ``MemoryStoreBase``) backed by a
local SuperLocalMemory data root. Records stay in your local SLM data root
(optional SLM providers, connectors, backup, and downloads have separate
network behavior) and are visible through every other SLM surface.

Scope / status
--------------
Written against Semantic Kernel 1.44 (``semantic_kernel.data.vector``). The SK
vector-store API is still marked **preview**, so this adapter is version-pinned
and must be validated in CI against an installed ``semantic-kernel``. CRUD and
collection lifecycle delegate to the framework-free :class:`V3RecordStore`
(independently tested). ``_inner_search`` performs field-filter retrieval over
stored records; dense-vector ANN ranking is delegated to SuperLocalMemory's
native recall surfaces and is a documented follow-up.

Usage::

    from semantic_kernel_superlocalmemory import SuperLocalMemoryVectorStore

    store = SuperLocalMemoryVectorStore()
    collection = store.get_collection(MyRecord, collection_name="docs")
    await collection.ensure_collection_exists()
    await collection.upsert(MyRecord(id="d1", text="hello"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, TypeVar

from semantic_kernel.data.vector import (
    GetFilteredRecordOptions,
    KernelSearchResults,
    SearchType,
    VectorSearch,
    VectorSearchOptions,
    VectorSearchResult,
    VectorStore,
    VectorStoreCollection,
)

TModel = TypeVar("TModel")


def _data_root() -> Path:
    import os

    value = (
        os.environ.get("SLM_DATA_DIR")
        or os.environ.get("SL_MEMORY_PATH")
        or os.environ.get("SLM_HOME")
    )
    return Path(value).expanduser() if value else Path.home() / ".superlocalmemory"


class SuperLocalMemoryCollection(
    VectorStoreCollection[str, TModel], VectorSearch[str, TModel]
):
    """SK vector-store collection backed by SuperLocalMemory V3."""

    def __init__(
        self,
        record_type: type,
        *,
        collection_name: str = "default",
        db_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            record_type=record_type, collection_name=collection_name, **kwargs
        )
        from semantic_kernel_superlocalmemory._v3_record_store import V3RecordStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3RecordStore(store_path)

    # -- serialization: our store model IS the plain dict ------------------

    def _serialize_dicts_to_store_models(
        self, records: Sequence[dict[str, Any]], **kwargs: Any
    ) -> Sequence[Any]:
        return list(records)

    def _deserialize_store_models_to_dicts(
        self, records: Sequence[Any], **kwargs: Any
    ) -> Sequence[dict[str, Any]]:
        return list(records)

    # -- CRUD --------------------------------------------------------------

    async def _inner_upsert(self, records: Sequence[Any], **kwargs: Any) -> Sequence[str]:
        keys: list[str] = []
        for record in records:
            key = self._extract_key(record)
            self._store.upsert(self.collection_name, key, dict(record))
            keys.append(key)
        return keys

    async def _inner_get(
        self,
        keys: Sequence[str] | None = None,
        options: GetFilteredRecordOptions | None = None,
        **kwargs: Any,
    ) -> Any | None:
        if keys is None:
            return self._store.list_records(self.collection_name)
        return self._store.get_many(self.collection_name, list(keys))

    async def _inner_delete(self, keys: Sequence[str], **kwargs: Any) -> None:
        for key in keys:
            self._store.delete(self.collection_name, key)

    # -- collection lifecycle ---------------------------------------------

    async def ensure_collection_exists(self, **kwargs: Any) -> None:
        self._store.create_collection(self.collection_name)

    async def collection_exists(self, **kwargs: Any) -> bool:
        return self._store.collection_exists(self.collection_name)

    async def ensure_collection_deleted(self, **kwargs: Any) -> None:
        self._store.delete_collection(self.collection_name)

    # -- search (VectorSearch mixin) --------------------------------------

    async def _inner_search(
        self,
        search_type: SearchType,
        options: VectorSearchOptions,
        values: Any | None = None,
        vector: Sequence[float | int] | None = None,
        **kwargs: Any,
    ) -> KernelSearchResults[VectorSearchResult[TModel]]:
        # Field-filter retrieval over stored records. Dense-vector ANN ranking
        # is delegated to SLM's native recall (documented follow-up), so scores
        # are reported as None here.
        records = self._store.list_records(self.collection_name)
        top = getattr(options, "top", None) or 10
        skip = getattr(options, "skip", None) or 0
        page = records[skip : skip + top]

        async def _results():
            for record in page:
                yield VectorSearchResult(record=record, score=None)

        return KernelSearchResults(results=_results(), total_count=len(records))

    def _get_record_from_result(self, result: Any) -> Any:
        return getattr(result, "record", result)

    def _get_score_from_result(self, result: Any) -> float | None:
        return getattr(result, "score", None)

    def _lambda_parser(self, ast_node: Any) -> Any:
        # Lambda-expression filters are not translated to a native store query
        # in this release; SK-side filtering still applies to returned records.
        return None

    def close(self) -> None:
        self._store.close()

    # -- helpers -----------------------------------------------------------

    def _extract_key(self, record: Any) -> str:
        """Best-effort extraction of the record key.

        Prefers the collection's declared key field (from ``self.definition``)
        and falls back to common key names. Defensive because the exact
        definition attribute is a preview-API surface pinned for CI checks.
        """
        for attr in ("key_name", "key_field_name"):
            name = getattr(getattr(self, "definition", None), attr, None)
            if name and isinstance(record, dict) and name in record:
                return str(record[name])
        key_field = getattr(getattr(self, "definition", None), "key_field", None)
        field_name = getattr(key_field, "name", None)
        if field_name and isinstance(record, dict) and field_name in record:
            return str(record[field_name])
        if isinstance(record, dict):
            for fallback in ("id", "key", "record_id"):
                if fallback in record:
                    return str(record[fallback])
        raise ValueError("could not determine record key for upsert")


class SuperLocalMemoryVectorStore(VectorStore):
    """SK vector store that hands out :class:`SuperLocalMemoryCollection`."""

    def __init__(self, db_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._db_path = db_path

    def get_collection(
        self,
        record_type: type,
        *,
        collection_name: str = "default",
        **kwargs: Any,
    ) -> SuperLocalMemoryCollection:
        return SuperLocalMemoryCollection(
            record_type=record_type,
            collection_name=collection_name,
            db_path=self._db_path,
            **kwargs,
        )

    async def list_collection_names(self, **kwargs: Any) -> list[str]:
        # SLM does not maintain a global collection registry independent of
        # records; callers track collection names at the application layer.
        return []
