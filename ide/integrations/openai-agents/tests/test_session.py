"""Integration tests for the OpenAI Agents SDK SLMSession wrapper.

Require ``openai-agents`` installed; skipped otherwise.  These validate that
the wrapper's async methods round-trip through the SLM engine correctly.
"""

import pytest

pytest.importorskip("agents")

import asyncio  # noqa: E402

from openai_agents_superlocalmemory import SLMSession  # noqa: E402


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    s = SLMSession(
        session_id="test-sess-wrapper",
        db_path=str(tmp_path / "session.db"),
    )
    try:
        yield s
    finally:
        s.close()


@pytest.mark.asyncio
async def test_add_and_get_items(session):
    await session.add_items([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ])
    items = await session.get_items()
    assert len(items) == 2
    assert items[0]["content"] == "Hello"
    assert items[1]["content"] == "Hi there"


@pytest.mark.asyncio
async def test_get_items_with_limit(session):
    await session.add_items([{"role": "user", "content": f"m{i}"} for i in range(5)])
    last_two = await session.get_items(limit=2)
    assert len(last_two) == 2
    assert last_two[0]["content"] == "m3"
    assert last_two[1]["content"] == "m4"


@pytest.mark.asyncio
async def test_pop_item(session):
    await session.add_items([
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ])
    popped = await session.pop_item()
    assert popped is not None
    assert popped["content"] == "second"
    remaining = await session.get_items()
    assert len(remaining) == 1


@pytest.mark.asyncio
async def test_pop_item_empty_returns_none(session):
    result = await session.pop_item()
    assert result is None


@pytest.mark.asyncio
async def test_clear_session(session):
    await session.add_items([{"role": "user", "content": "x"}])
    await session.clear_session()
    items = await session.get_items()
    assert items == []


@pytest.mark.asyncio
async def test_session_id_attribute(session):
    assert session.session_id == "test-sess-wrapper"


@pytest.mark.asyncio
async def test_session_settings_attribute(session):
    # Default is empty dict when not provided.
    assert isinstance(session.session_settings, dict)


@pytest.mark.asyncio
async def test_session_settings_custom():
    """Custom session_settings are stored and returned unchanged."""
    import tempfile, os
    os.environ["SLM_TEST_ISOLATION"] = "1"
    with tempfile.TemporaryDirectory() as d:
        s = SLMSession(
            session_id="s2",
            db_path=f"{d}/test.db",
            session_settings={"model": "gpt-4o", "temperature": 0.7},
        )
        try:
            assert s.session_settings["model"] == "gpt-4o"
        finally:
            s.close()
