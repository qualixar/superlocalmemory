"""Integration tests for the LangGraph BaseStore wrapper.

These require ``langgraph`` installed and are skipped otherwise. They confirm
the wrapper's convenience methods (inherited from BaseStore and delegating to
our batch/abatch) round-trip through the SLM engine.
"""

import pytest

pytest.importorskip("langgraph")

from langgraph_superlocalmemory import SuperLocalMemoryStore  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    s = SuperLocalMemoryStore(db_path=str(tmp_path / "kv.db"))
    try:
        yield s
    finally:
        s.close()


def test_put_get_roundtrip(store):
    store.put(("users", "1"), "profile", {"name": "Ada"})
    item = store.get(("users", "1"), "profile")
    assert item is not None
    assert item.value == {"name": "Ada"}
    assert item.namespace == ("users", "1")
    assert item.key == "profile"


def test_get_missing(store):
    assert store.get(("nope",), "x") is None


def test_search_prefix(store):
    store.put(("users", "1"), "a", {"role": "eng"})
    store.put(("users", "2"), "b", {"role": "eng"})
    store.put(("orgs",), "c", {"role": "eng"})
    hits = store.search(("users",))
    assert {h.namespace for h in hits} == {("users", "1"), ("users", "2")}
    assert all(h.score is None for h in hits)


def test_search_filter(store):
    store.put(("d",), "d1", {"status": "draft"})
    store.put(("d",), "d2", {"status": "final"})
    hits = store.search(("d",), filter={"status": "final"})
    assert [h.key for h in hits] == ["d2"]


def test_delete(store):
    store.put(("k",), "one", {"v": 1})
    store.delete(("k",), "one")
    assert store.get(("k",), "one") is None


def test_list_namespaces(store):
    store.put(("users", "1"), "a", {})
    store.put(("orgs",), "c", {})
    namespaces = store.list_namespaces()
    assert ("users", "1") in namespaces
    assert ("orgs",) in namespaces


@pytest.mark.asyncio
async def test_abatch_does_not_block(store):
    # aput/aget go through abatch -> to_thread; verify they work.
    await store.aput(("async",), "k", {"v": 1})
    item = await store.aget(("async",), "k")
    assert item.value == {"v": 1}
