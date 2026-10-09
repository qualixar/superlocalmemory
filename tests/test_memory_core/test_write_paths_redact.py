# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""With redaction on, no write door leaves a raw identifier on disk."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes.ingest import router as ingest_router
from superlocalmemory.server.unified_daemon import create_app
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations,
    M032_write_coordinator_admission,
    M033_projection_transactions,
    M034_obligation_integrity,
    M042_correction_case_ledger,
)

EMAIL = "zelda.fitzgerald@example.org"
PHONE = "415-555-0187"


def _text(tag: str) -> str:
    return (
        f"{tag}: the platform owner can be reached at {EMAIL} or {PHONE} "
        "for the quarterly reliability review."
    )


def _disk_bytes(root: Path) -> bytes:
    return b"\n".join(p.read_bytes() for p in root.rglob("*") if p.is_file())


def _assert_raw_absent(root: Path) -> None:
    blob = _disk_bytes(root)
    assert EMAIL.encode() not in blob
    assert PHONE.encode() not in blob
    assert b"[PII:EMAIL]" in blob


@contextmanager
def _daemon_client(engine):
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        M033_projection_transactions.apply(conn)
        M034_obligation_integrity.apply(conn)
        M042_correction_case_ledger.apply(conn)
    app = create_app()
    app.state.engine = engine
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    app.state.canonical_remember_runtime = runtime
    client = TestClient(app)
    client.headers["X-SLM-Daemon-Capability"] = app.state.daemon_descriptor.capability
    client.headers["X-SLM-Target-Instance"] = app.state.daemon_descriptor.instance_id
    try:
        yield client
    finally:
        runtime.stop()


def _ingest_client(engine) -> TestClient:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = FastAPI()
    app.state.engine = engine

    @app.middleware("http")
    async def _actor(request, call_next):
        request.state.authenticated_actor = "authenticated:test-adapter"
        return await call_next(request)

    app.include_router(ingest_router)
    return TestClient(app)


def test_daemon_remember_redacts_on_disk(engine_with_mock_deps, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    with _daemon_client(engine_with_mock_deps) as client:
        resp = client.post("/remember?wait=true", json={
            "content": _text("remember"), "idempotency_key": "pii-remember-1",
        })
        assert resp.status_code == 200, resp.text
    _assert_raw_absent(tmp_path)


def test_ingest_redacts_on_disk(engine_with_mock_deps, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    resp = _ingest_client(engine_with_mock_deps).post("/ingest", json={
        "content": _text("ingest"), "source_type": "gmail", "dedup_key": "pii-ingest-1",
    })
    assert resp.status_code == 200, resp.text
    _assert_raw_absent(tmp_path)


def test_engine_path_redacts_on_disk(engine_with_mock_deps, monkeypatch, tmp_path) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    canonical_store(
        engine_with_mock_deps, _text("engine"), source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True,
    )
    _assert_raw_absent(tmp_path)


def test_prebuilt_fact_redacts_request_and_payload(
    engine_with_mock_deps, monkeypatch, tmp_path,
) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store_fact,
        local_trusted_actor_id,
    )
    from superlocalmemory.storage.models import AtomicFact

    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    engine = engine_with_mock_deps
    fact = AtomicFact(
        fact_id="prebuilt-pii-1", memory_id="", profile_id=engine._profile_id,
        content=_text("prebuilt"),
    )
    original = fact.content
    canonical_store_fact(
        engine, fact, trusted_actor_id=local_trusted_actor_id("python-api"),
    )
    assert fact.content == original  # the caller's object is not mutated
    _assert_raw_absent(tmp_path)


def test_redaction_off_stores_the_text_as_written(
    engine_with_mock_deps, monkeypatch, tmp_path,
) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    resp = _ingest_client(engine_with_mock_deps).post("/ingest", json={
        "content": _text("off"), "source_type": "gmail", "dedup_key": "pii-off-1",
    })
    assert resp.status_code == 200, resp.text
    blob = _disk_bytes(tmp_path)
    assert EMAIL.encode() in blob
    assert b"[PII:EMAIL]" not in blob


def test_legacy_backfill_hands_over_redacted_text(monkeypatch) -> None:
    """Route-level check: the request built for the backfill is redacted."""
    from unittest.mock import MagicMock

    from superlocalmemory.daemon import materializer

    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    seen: list = []

    class _Cmd:
        def submit(self, request):
            seen.append(request)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
        lambda engine, **kw: _Cmd(),
    )
    engine = MagicMock()
    engine._profile_id = "default"
    engine._config.pii_redaction = False
    with pytest.raises(RuntimeError):
        materializer.legacy_item(
            engine, {"id": 1, "content": _text("legacy"), "profile_id": "default"},
            actor_id="t",
        )
    assert EMAIL not in seen[0].content and "[PII:EMAIL]" in seen[0].content


def test_observe_hands_over_redacted_text(monkeypatch) -> None:
    """Route-level check: the request built for auto-capture is redacted."""
    from unittest.mock import MagicMock

    from superlocalmemory.server.unified_daemon import ObserveBuffer

    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    seen: list = []

    class _Cmd:
        def submit(self, request):
            seen.append(request)
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        "superlocalmemory.core.engine_ingestion.build_engine_ingestion_command",
        lambda engine, **kw: _Cmd(),
    )
    monkeypatch.setattr(
        "superlocalmemory.hooks.auto_capture.AutoCapture.evaluate",
        lambda self, content: MagicMock(
            capture=True, category="decision", confidence=0.9, reason="",
        ),
    )
    engine = MagicMock()
    engine._profile_id = "default"
    engine._config.pii_redaction = False
    engine._config.scope.default_scope = "personal"
    buf = ObserveBuffer(debounce_sec=60)
    buf.set_engine(engine)
    try:
        buf.enqueue(_text("observe"), trusted_actor_id="t")
    except RuntimeError:
        pass
    finally:
        if buf._timer is not None:
            buf._timer.cancel()
    assert seen, "observe never built a request"
    assert EMAIL not in seen[0].content and "[PII:EMAIL]" in seen[0].content
