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


def test_authorized_delete_runs_trust_before_persistence(engine_with_mock_deps) -> None:
    # The delete path proves cross-store erasure before it reports success, so a
    # bare MagicMock DB cannot model it. Exercise the trust-gate ordering against
    # a real engine: the pre hook must fire (for "delete") before the post hook,
    # and the fact must be verifiably erased.
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )
    from superlocalmemory.core.mutations import delete_fact_authorized

    engine = engine_with_mock_deps
    actor = local_trusted_actor_id("python-api")
    operation = canonical_store(
        engine,
        "Rana relocated to the Lisbon research office on 2025-02-01",
        source_type="python-api",
        trusted_actor_id=actor,
        require_complete=True,
        return_receipt=True,
    )
    fact_ids = list(operation.final_fact_ids)
    assert fact_ids
    fid = fact_ids[0]

    events: list[tuple[str, str]] = []
    real_pre = engine._hooks.run_pre
    real_post = engine._hooks.run_post

    def spy_pre(operation, ctx, *a, **k):
        events.append(("pre", operation))
        return real_pre(operation, ctx, *a, **k)

    def spy_post(operation, ctx, *a, **k):
        events.append(("post", operation))
        return real_post(operation, ctx, *a, **k)

    engine._hooks.run_pre = spy_pre
    engine._hooks.run_post = spy_post
    try:
        result = delete_fact_authorized(
            engine, fid, trusted_actor_id=actor, source_agent_id="python-api",
        )
    finally:
        engine._hooks.run_pre = real_pre
        engine._hooks.run_post = real_post

    assert result["ok"] is True
    assert result["erasure_verified"] is True
    assert events[0] == ("pre", "delete")
    assert ("post", "delete") in events
    assert events.index(("pre", "delete")) < events.index(("post", "delete"))
    assert engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fid,)
    ) == []


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
        "fact-1", {"content": "new content"}, profile_id="default"
    )
    bm25.update_fact.assert_called_once_with("fact-1", "new content", "default")
    engine._hooks.run_post.assert_called_once()
