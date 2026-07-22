# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Destructive worker writes derive authority from the local capability."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from superlocalmemory.core import recall_worker
from superlocalmemory.core.ingestion_command import IngestionState


def _engine() -> MagicMock:
    engine = MagicMock()
    engine.profile_id = "default"
    engine._profile_id = "default"
    engine._embedder = None
    engine._retrieval_engine = None
    engine._db.execute.return_value = [{"content": "old content"}]
    return engine


def test_delete_uses_capability_actor_and_treats_agent_label_as_metadata(
    monkeypatch,
) -> None:
    engine = _engine()
    monkeypatch.setattr(recall_worker, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        lambda kind: f"trusted:{kind}",
    )

    result = recall_worker._handle_delete_memory(
        "fact-1",
        source_agent_id="caller-selected-admin",
    )

    assert result["ok"] is True
    engine._hooks.run_pre.assert_called_once_with(
        "delete",
        {
            "operation": "delete",
            "agent_id": "trusted:recall-worker",
            "source_agent_id": "caller-selected-admin",
            "profile_id": "default",
            "fact_id": "fact-1",
        },
    )
    engine._db.delete_fact.assert_called_once_with("fact-1", profile_id="default")


def test_update_uses_capability_actor_and_runs_write_authorization(
    monkeypatch,
) -> None:
    engine = _engine()
    monkeypatch.setattr(recall_worker, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        lambda kind: f"trusted:{kind}",
    )

    result = recall_worker._handle_update_memory(
        "fact-1",
        "new content",
        source_agent_id="caller-selected-admin",
    )

    assert result["ok"] is True
    engine._hooks.run_pre.assert_called_once_with(
        "update",
        {
            "operation": "update",
            "agent_id": "trusted:recall-worker",
            "source_agent_id": "caller-selected-admin",
            "profile_id": "default",
            "fact_id": "fact-1",
            "content_preview": "new content",
        },
    )
    engine._db.update_fact.assert_called_once_with(
        "fact-1",
        {"content": "new content"},
        profile_id="default",
    )


def test_store_uses_capability_actor_before_canonical_persistence(
    monkeypatch,
) -> None:
    engine = _engine()
    canonical_store = MagicMock(return_value=SimpleNamespace(
        fact_ids=("fact-1",),
        operation_id="operation-1",
        state=IngestionState.COMPLETE,
    ))
    monkeypatch.setattr(recall_worker, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.local_trusted_actor_id",
        lambda kind: f"trusted:{kind}",
    )
    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.canonical_store",
        canonical_store,
    )

    result = recall_worker._handle_store(
        "remember this",
        {
            "agent_id": "caller-selected-admin",
            "scope": "shared",
            "shared_with": ["research"],
            "idempotency_key": "request-1",
        },
    )

    assert result == {
        "ok": True,
        "fact_ids": ["fact-1"],
        "count": 1,
        "operation_id": "operation-1",
        "pending_id": None,
        "materialization_state": "complete",
    }
    canonical_store.assert_called_once_with(
        engine,
        "remember this",
        source_type="mcp-offline-worker",
        trusted_actor_id="trusted:recall-worker",
        metadata={"agent_id": "caller-selected-admin"},
        scope="shared",
        shared_with=["research"],
        session_id="",
        idempotency_key="request-1",
        return_receipt=True,
    )
