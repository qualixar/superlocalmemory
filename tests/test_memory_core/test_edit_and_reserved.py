# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Edits are prepared like saves; callers cannot set reserved keys; retries survive a toggle."""

from __future__ import annotations

from superlocalmemory.core.ingestion_command import IngestionOperationRepository

from .test_write_paths_redact import (
    EMAIL,
    _daemon_client,
    _disk_bytes,
    _ingest_client,
)

EDIT = f"The platform owner changed and can now be reached at {EMAIL} for reviews."


def _store_then_edit(engine, text: str) -> int:
    with _daemon_client(engine) as client:
        stored = client.post("/remember?wait=true", json={
            "content": "Dashboard edit source memory about the quarterly review.",
            "idempotency_key": "edit-source-1",
        })
        fact_id = stored.json()["fact_ids"][0]
        edit = client.patch(f"/api/memories/{fact_id}", json={"content": text})
        return edit.status_code


def test_edit_redacts_new_text_on_disk(engine_with_mock_deps, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    assert _store_then_edit(engine_with_mock_deps, EDIT) == 202
    blob = _disk_bytes(tmp_path)
    assert EMAIL.encode() not in blob and b"[PII:EMAIL]" in blob


def test_edit_is_byte_identical_when_off(engine_with_mock_deps, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    assert _store_then_edit(engine_with_mock_deps, EDIT) == 202
    blob = _disk_bytes(tmp_path)
    assert EMAIL.encode() in blob and b"[PII:EMAIL]" not in blob


def test_ingest_drops_reserved_keys(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    resp = _ingest_client(engine_with_mock_deps).post("/ingest", json={
        "content": "Gmail message says the production recovery plan was approved.",
        "source_type": "gmail", "dedup_key": "reserved-1",
        "metadata": {"_slm_source": {"page": 1}, "_slm_memory_kind": "rule", "keep": "x"},
    })
    assert resp.status_code == 200, resp.text
    ops = IngestionOperationRepository(engine_with_mock_deps._db).list_operations()
    assert ops[0].metadata == {"keep": "x"}


def test_engine_path_drops_reserved_keys_but_keeps_trusted_ones(
    engine_with_mock_deps, monkeypatch,
) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    engine = engine_with_mock_deps
    canonical_store(
        engine, "The platform review was scheduled for Thursday afternoon.",
        source_type="python-api", trusted_actor_id=local_trusted_actor_id("python-api"),
        metadata={"_slm_memory_kind": "rule", "keep": "x"},
        trusted_metadata={"_slm_memory_kind": "decision"}, require_complete=False,
    )
    canonical_store(
        engine, "The platform review moved to Friday morning instead.",
        source_type="python-api", trusted_actor_id=local_trusted_actor_id("python-api"),
        metadata={"_slm_memory_kind": "rule"}, require_complete=False,
    )
    ops = IngestionOperationRepository(engine._db).list_operations()
    kinds = sorted(str(o.metadata.get("_slm_memory_kind")) for o in ops)
    assert kinds == ["None", "decision"]


def test_ingest_retry_after_redaction_is_turned_on_is_already_ingested(
    engine_with_mock_deps, monkeypatch,
) -> None:
    client = _ingest_client(engine_with_mock_deps)
    payload = {
        "content": f"Gmail message from {EMAIL} says the plan was approved.",
        "source_type": "gmail", "dedup_key": "toggle-1",
    }
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    first = client.post("/ingest", json=payload)
    assert first.json()["ingested"] is True
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    again = client.post("/ingest", json=payload)
    assert again.status_code == 200, again.text
    assert again.json()["ingested"] is False
    assert again.json()["reason"] == "already_ingested"
    assert again.json()["operation_id"] == first.json()["operation_id"]
