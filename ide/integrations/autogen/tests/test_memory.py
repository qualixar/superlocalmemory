"""Tests for the AutoGen Memory ABC wrapper.

``autogen_agentchat`` is not installed in the dev venv, so these tests are
skipped unless the framework is present. The store core is exercised
independently in ``test_autogen_store.py``.
"""

import pytest

autogen_agentchat = pytest.importorskip("autogen_agentchat")


def test_memory_is_memory_subclass(tmp_path):
    from autogen_core.memory import Memory
    from autogen_superlocalmemory.memory import SuperLocalMemoryMemory
    mem = SuperLocalMemoryMemory(db_path=str(tmp_path / "t.db"))
    assert isinstance(mem, Memory)


@pytest.mark.asyncio
async def test_add_and_query(tmp_path):
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_superlocalmemory.memory import SuperLocalMemoryMemory

    mem = SuperLocalMemoryMemory(db_path=str(tmp_path / "t.db"))
    await mem.add(MemoryContent(content="Async agents rock.", mime_type=MemoryMimeType.TEXT))
    result = await mem.query(MemoryContent(content="agents", mime_type=MemoryMimeType.TEXT))
    assert hasattr(result, "results")
    await mem.close()


@pytest.mark.asyncio
async def test_clear(tmp_path):
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_superlocalmemory.memory import SuperLocalMemoryMemory

    mem = SuperLocalMemoryMemory(db_path=str(tmp_path / "t.db"))
    await mem.add(MemoryContent(content="Should be cleared.", mime_type=MemoryMimeType.TEXT))
    await mem.clear()
    result = await mem.query("cleared", limit=10)
    assert result.results == []
    await mem.close()
