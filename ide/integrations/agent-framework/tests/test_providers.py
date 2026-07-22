"""Integration tests for the Agent Framework providers.

Require ``agent-framework-core`` installed; skipped otherwise. They validate
that the history provider round-trips messages through the SLM engine.
"""

import pytest

pytest.importorskip("agent_framework")

from agent_framework import Message  # noqa: E402

from agent_framework_superlocalmemory import (  # noqa: E402
    SuperLocalMemoryHistoryProvider,
)


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    provider = SuperLocalMemoryHistoryProvider(db_path=str(tmp_path / "m.db"))
    try:
        yield provider
    finally:
        provider.close()


@pytest.mark.asyncio
async def test_save_and_get_messages(history):
    await history.save_messages(
        "s1",
        [Message(role="user", text="hello"), Message(role="assistant", text="hi")],
    )
    msgs = await history.get_messages("s1")
    assert len(msgs) == 2


@pytest.mark.asyncio
async def test_session_isolation(history):
    await history.save_messages("a", [Message(role="user", text="in a")])
    await history.save_messages("b", [Message(role="user", text="in b")])
    assert len(await history.get_messages("a")) == 1
    assert len(await history.get_messages("b")) == 1


@pytest.mark.asyncio
async def test_clear(history):
    await history.save_messages("s1", [Message(role="user", text="one")])
    history.clear("s1")
    assert await history.get_messages("s1") == []
