"""Tests for the CrewAI StorageBackend wrapper.

``crewai`` is not installed in the dev venv, so these tests are skipped unless
the framework is present. The store core is exercised independently in
``test_crewai_store.py``.
"""

import pytest

crewai = pytest.importorskip("crewai")


def test_backend_is_runtime_checkable():
    from crewai.memory.storage.backend import StorageBackend
    from crewai_superlocalmemory.backend import SuperLocalMemoryBackend
    assert isinstance(SuperLocalMemoryBackend(), StorageBackend)


def test_save_and_search(tmp_path):
    from crewai.memory.types import MemoryRecord
    from crewai_superlocalmemory.backend import SuperLocalMemoryBackend

    backend = SuperLocalMemoryBackend(db_path=str(tmp_path / "b.db"))
    rec = MemoryRecord(id="r1", content="hello world", scope="/test",
                       embedding=[0.1, 0.2, 0.3], category="general")
    backend.save([rec])
    results = backend.search([0.1, 0.2, 0.3], scope_prefix="/test", limit=5)
    assert len(results) >= 1
    assert results[0][0].id == "r1"
