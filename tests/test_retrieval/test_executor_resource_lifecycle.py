"""Owned retrieval resources must be reused and deterministically released."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.retrieval.engine import RetrievalEngine


class _ImmediateExecutor:
    instances: list["_ImmediateExecutor"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.shutdown_calls = 0
        self.instances.append(self)

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.shutdown()

    def shutdown(self, *args, **kwargs) -> None:
        self.shutdown_calls += 1


def test_retrieval_engine_reuses_and_closes_one_channel_executor() -> None:
    _ImmediateExecutor.instances = []
    db = MagicMock()
    db.get_facts_by_ids.return_value = []
    bm25 = MagicMock()
    bm25.search.return_value = []

    with patch("concurrent.futures.ThreadPoolExecutor", _ImmediateExecutor):
        engine = RetrievalEngine(
            db=db,
            config=RetrievalConfig(),
            channels={"bm25": bm25},
        )
        engine.recall("first query", "default")
        engine.recall("second query", "default")
        engine.close()
        engine.close()

    assert len(_ImmediateExecutor.instances) == 1
    assert _ImmediateExecutor.instances[0].shutdown_calls == 1


def test_memory_engine_close_releases_retrieval_before_database() -> None:
    order: list[str] = []
    engine = MemoryEngine.__new__(MemoryEngine)
    engine._maintenance_scheduler = None
    engine._retrieval_engine = MagicMock()
    engine._retrieval_engine.close.side_effect = lambda: order.append("retrieval")
    engine._db = MagicMock()
    engine._db.close.side_effect = lambda: order.append("database")
    engine._initialized = True

    engine.close()

    assert order == ["retrieval", "database"]
    assert engine._initialized is False


def test_recall_singletons_release_closed_database() -> None:
    from superlocalmemory.core import recall_pipeline

    db = MagicMock()
    db_key = id(db)
    recall_pipeline._behavioral_tracker_cache[db_key] = object()
    recall_pipeline._forgetting_scheduler_cache[db_key] = object()

    recall_pipeline.release_recall_resources(db)

    assert db_key not in recall_pipeline._behavioral_tracker_cache
    assert db_key not in recall_pipeline._forgetting_scheduler_cache
