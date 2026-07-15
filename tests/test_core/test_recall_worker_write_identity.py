# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Destructive worker writes derive authority from the local capability."""

from unittest.mock import MagicMock

from superlocalmemory.core import recall_worker


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
    engine._db.delete_fact.assert_called_once_with("fact-1")


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
    )
