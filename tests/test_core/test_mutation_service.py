# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Canonical delete/update mutation service contracts."""

from unittest.mock import MagicMock


def _engine() -> MagicMock:
    engine = MagicMock()
    engine.profile_id = "default"
    engine._profile_id = "default"
    engine._embedder = None
    engine._retrieval_engine = None
    engine._db.execute.return_value = [{"content": "old content"}]
    return engine


def test_authorized_delete_runs_trust_before_persistence() -> None:
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = _engine()
    result = delete_fact_authorized(
        engine,
        "fact-1",
        trusted_actor_id="trusted:cli",
        source_agent_id="cli",
    )

    assert result["ok"] is True
    assert engine._hooks.run_pre.call_args_list[0].args[0] == "delete"
    engine._db.delete_fact.assert_called_once_with("fact-1")
    engine._hooks.run_post.assert_called_once()


def test_authorized_update_refreshes_indexes_after_trust_gate() -> None:
    from superlocalmemory.core.mutations import update_fact_authorized

    engine = _engine()
    bm25 = MagicMock()
    engine._retrieval_engine = MagicMock(_bm25=bm25)

    result = update_fact_authorized(
        engine,
        "fact-1",
        "new content",
        trusted_actor_id="trusted:cli",
        source_agent_id="cli",
    )

    assert result["ok"] is True
    assert engine._hooks.run_pre.call_args_list[0].args[0] == "update"
    engine._db.update_fact.assert_called_once_with(
        "fact-1", {"content": "new content"}
    )
    bm25.add.assert_called_once_with("fact-1", "new content", "default")
    engine._hooks.run_post.assert_called_once()
