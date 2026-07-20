# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Regression contracts for low-information ingestion admission."""

from __future__ import annotations

import pytest

from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command
from superlocalmemory.core.ingestion_command import IngestionRequest


def _durable_counts(engine) -> tuple[int, int, int]:
    db = engine._db
    memories = db.execute("SELECT COUNT(*) AS count FROM memories")
    facts = db.execute("SELECT COUNT(*) AS count FROM atomic_facts")
    operations = db.execute("SELECT COUNT(*) AS count FROM ingestion_operations")
    return tuple(
        int(dict(rows[0])["count"])
        for rows in (memories, facts, operations)
    )


def test_public_store_rejects_low_information_without_durable_residue(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    before = _durable_counts(engine)

    assert engine.store("ok", session_id="low-information") == []

    assert _durable_counts(engine) == before


def test_store_fast_rejects_low_information_without_durable_residue(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    before = _durable_counts(engine)

    assert engine.store_fast("ok") == []

    assert _durable_counts(engine) == before


def test_command_rolls_back_operation_when_low_information_is_not_queryable(
    engine_with_mock_deps,
) -> None:
    engine = engine_with_mock_deps
    command = build_engine_ingestion_command(engine)
    before = _durable_counts(engine)
    request = IngestionRequest(
        content="ok",
        profile_id=engine._profile_id,
        source_type="test",
        idempotency_key="low-information-command",
        trusted_actor_id="test-capability",
    )

    with pytest.raises(RuntimeError, match="no queryable facts"):
        command.submit(request)

    assert _durable_counts(engine) == before
