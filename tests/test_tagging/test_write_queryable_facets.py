# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Facets are stored by the shared write step and cannot be forged or break a save."""

from __future__ import annotations

import json

from tests.test_memory_core.test_write_paths_redact import (
    EMAIL,
    _daemon_client,
    _ingest_client,
)

TEXT = "Release notes at https://example.com/rel?token=zz were approved on 2026-03-04."
PROSE = "The platform owner confirmed the quarterly reliability review for Thursday."


def _metadata_of(engine, needle: str) -> dict:
    rows = engine._db.execute(
        "SELECT metadata_json FROM memories WHERE content LIKE ?", (f"%{needle}%",),
    )
    assert rows, "memory not stored"
    return json.loads(rows[0]["metadata_json"] or "{}")


def _remember(client, text: str, key: str, **extra) -> None:
    resp = client.post("/remember?wait=true", json={
        "content": text, "idempotency_key": key, **extra,
    })
    assert resp.status_code == 200, resp.text


def test_url_and_date_are_stored(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, TEXT, "facet-1")
    meta = _metadata_of(engine_with_mock_deps, "Release notes")
    assert meta["_slm_extracted"] == {
        "urls": ["https://example.com/rel"], "dates": ["2026-03-04"],
    }


def test_plain_prose_has_no_extracted_key(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, PROSE, "facet-2")
    assert "_slm_extracted" not in _metadata_of(engine_with_mock_deps, "quarterly reliability")


def test_remember_cannot_forge_the_key(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    forged = {"_slm_extracted": {"urls": ["https://evil.example/x"]}}
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, TEXT, "facet-3", metadata=forged)
        _remember(client, PROSE, "facet-3b", metadata=forged)
    meta = _metadata_of(engine_with_mock_deps, "Release notes")
    assert meta["_slm_extracted"]["urls"] == ["https://example.com/rel"]
    assert "_slm_extracted" not in _metadata_of(engine_with_mock_deps, "quarterly reliability")


def test_ingest_cannot_forge_the_key(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    resp = _ingest_client(engine_with_mock_deps).post("/ingest", json={
        "content": PROSE, "source_type": "gmail", "dedup_key": "facet-4",
        "metadata": {"_slm_extracted": {"urls": ["https://evil.example/x"]}},
    })
    assert resp.status_code == 200, resp.text
    assert "evil.example" not in json.dumps(
        _metadata_of(engine_with_mock_deps, "quarterly reliability"))


def test_overwrites_a_preexisting_value() -> None:
    from superlocalmemory.tagging import add_extracted

    meta = {"_slm_extracted": {"urls": ["https://evil.example/x"]}}
    add_extracted(meta, PROSE)
    assert "_slm_extracted" not in meta
    add_extracted(meta, TEXT)
    assert meta["_slm_extracted"]["urls"] == ["https://example.com/rel"]


def test_pii_is_not_in_the_facets(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    text = f"Mail {EMAIL} about https://example.com/pii on 2026-04-05 please."
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, text, "facet-5")
    meta = _metadata_of(engine_with_mock_deps, "about https://example.com/pii")
    assert meta["_slm_extracted"]["urls"] == ["https://example.com/pii"]
    assert EMAIL not in json.dumps(meta)


def test_extractor_failure_does_not_fail_the_save(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)

    def boom(text):
        raise RuntimeError("extractor down")

    monkeypatch.setattr("superlocalmemory.tagging.extractors.extract_deterministic", boom)
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, TEXT, "facet-6")
    assert "_slm_extracted" not in _metadata_of(engine_with_mock_deps, "Release notes")


def test_retry_is_a_duplicate_not_an_error(engine_with_mock_deps, monkeypatch) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    with _daemon_client(engine_with_mock_deps) as client:
        _remember(client, TEXT, "facet-7")
        _remember(client, TEXT, "facet-7")
    rows = engine_with_mock_deps._db.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE content LIKE ?", ("%Release notes%",))
    assert rows[0]["n"] == 1
