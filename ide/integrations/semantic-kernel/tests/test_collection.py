"""Integration tests for the SK VectorStoreCollection wrapper.

Require ``semantic-kernel`` installed; skipped otherwise. These validate that
the wrapper's CRUD + collection lifecycle round-trip through the SLM engine.
"""

from dataclasses import dataclass
from typing import Annotated

import pytest

pytest.importorskip("semantic_kernel")

from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel  # noqa: E402

from semantic_kernel_superlocalmemory import (  # noqa: E402
    SuperLocalMemoryVectorStore,
)


@vectorstoremodel
@dataclass
class Doc:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    store = SuperLocalMemoryVectorStore(db_path=str(tmp_path / "rec.db"))
    coll = store.get_collection(Doc, collection_name="docs")
    try:
        yield coll
    finally:
        coll.close()


@pytest.mark.asyncio
async def test_collection_lifecycle(collection):
    assert await collection.collection_exists() is False
    await collection.ensure_collection_exists()
    assert await collection.collection_exists() is True
    await collection.ensure_collection_deleted()
    assert await collection.collection_exists() is False


@pytest.mark.asyncio
async def test_upsert_get_delete(collection):
    await collection.ensure_collection_exists()
    await collection.upsert(Doc(id="d1", text="hello"))
    got = await collection.get("d1")
    assert got is not None
    await collection.delete("d1")
    assert await collection.get("d1") is None
