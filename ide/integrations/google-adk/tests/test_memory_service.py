"""Integration tests for the Google ADK BaseMemoryService wrapper.

Require ``google-adk`` installed; skipped otherwise.  These validate that
the wrapper correctly extracts session fields defensively and round-trips
through the SLM engine.
"""

import pytest

pytest.importorskip("google.adk")

import asyncio  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from google_adk_superlocalmemory import SuperLocalMemoryService  # noqa: E402


def _mock_session(app_name: str, user_id: str, session_id: str, texts: list[str]):
    """Build a mock ADK Session whose events return text via .content.parts."""
    session = MagicMock()
    session.app_name = app_name
    session.user_id = user_id
    session.id = session_id

    events = []
    for i, text in enumerate(texts):
        part = MagicMock()
        part.text = text
        content = MagicMock()
        content.parts = [part]
        event = MagicMock()
        event.content = content
        event.author = "user"
        event.timestamp = float(i)
        events.append(event)

    session.events = events
    return session


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    svc = SuperLocalMemoryService(db_path=str(tmp_path / "adk.db"))
    try:
        yield svc
    finally:
        svc.close()


@pytest.mark.asyncio
async def test_add_session_stores_events(service):
    session = _mock_session("myapp", "alice", "s1", ["Hello world", "How are you?"])
    await service.add_session_to_memory(session)
    stored = service._store.list_events_for_session("myapp", "alice", "s1")
    assert len(stored) == 2
    assert stored[0]["event"]["text"] == "Hello world"


@pytest.mark.asyncio
async def test_add_session_replaces_on_re_add(service):
    session = _mock_session("app", "u", "s", ["First", "Second"])
    await service.add_session_to_memory(session)
    session2 = _mock_session("app", "u", "s", ["Only"])
    await service.add_session_to_memory(session2)
    stored = service._store.list_events_for_session("app", "u", "s")
    assert len(stored) == 1
    assert stored[0]["event"]["text"] == "Only"


@pytest.mark.asyncio
async def test_search_memory_returns_response_type(service):
    from google.adk.memory.base_memory_service import SearchMemoryResponse

    session = _mock_session("app", "u", "s1", ["The capital of France is Paris"])
    await service.add_session_to_memory(session)
    resp = await service.search_memory(app_name="app", user_id="u", query="Paris")
    assert isinstance(resp, SearchMemoryResponse)


@pytest.mark.asyncio
async def test_search_memory_empty_when_no_events(service):
    resp = await service.search_memory(app_name="app", user_id="nobody", query="x")
    assert resp.memories == []


@pytest.mark.asyncio
async def test_add_session_defensive_against_missing_fields(service):
    """add_session_to_memory must not raise when optional fields are absent."""
    session = MagicMock()
    session.app_name = "app"
    session.user_id = "u"
    session.id = "s"
    # Event with no content.parts
    event = MagicMock()
    event.content = None
    event.author = "agent"
    event.timestamp = None
    session.events = [event]

    # Must not raise
    await service.add_session_to_memory(session)
    stored = service._store.list_events_for_session("app", "u", "s")
    assert len(stored) == 1
    assert stored[0]["event"]["text"] == ""
