# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Only a true redaction duplicate is tolerated; remaining raw keys are closed."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core.ingestion_command import IngestionOperationRepository
from superlocalmemory.memory_core import effective_pii_redaction
from superlocalmemory.server.routes.ingest import router as ingest_router
from superlocalmemory.storage.migrations import M018_ingestion_operations

from .test_write_paths_redact import EMAIL, PHONE

CONTENT = f"Gmail message from {EMAIL} says the production plan was approved."


def _client(engine, actor: str = "authenticated:a") -> TestClient:
    with engine._db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
    app = FastAPI()
    app.state.engine = engine

    @app.middleware("http")
    async def _actor(request, call_next):
        request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(ingest_router)
    return TestClient(app)


def _post(client, key: str, label: str):
    return client.post("/ingest", json={
        "content": CONTENT, "source_type": "gmail", "dedup_key": key,
        "metadata": {"label": label},
    })


def test_different_metadata_is_still_a_conflict_with_redaction_on(
    engine_with_mock_deps, monkeypatch,
) -> None:
    client = _client(engine_with_mock_deps)
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    assert _post(client, "off-key", "a").status_code == 200
    off = _post(client, "off-key", "b")
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    assert _post(client, "on-key", "a").status_code == 200
    on = _post(client, "on-key", "b")
    assert off.status_code != 200 and on.status_code == off.status_code


def test_toggle_retry_with_changed_metadata_is_a_conflict(
    engine_with_mock_deps, monkeypatch,
) -> None:
    client = _client(engine_with_mock_deps)
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    assert _post(client, "toggle-meta", "a").status_code == 200
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    assert _post(client, "toggle-meta", "b").status_code != 200
    same = _post(client, "toggle-meta", "a")
    assert same.status_code == 200 and same.json()["reason"] == "already_ingested"


def test_toggle_retry_by_another_actor_is_a_conflict(
    engine_with_mock_deps, monkeypatch,
) -> None:
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    assert _post(_client(engine_with_mock_deps), "toggle-actor", "a").status_code == 200
    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    other = _client(engine_with_mock_deps, actor="authenticated:someone-else")
    assert _post(other, "toggle-actor", "a").status_code != 200


def test_effective_redaction_includes_the_deployment_config(monkeypatch) -> None:
    from superlocalmemory.core import config as config_module
    from superlocalmemory.core.config import SLMConfig

    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    monkeypatch.setattr(
        config_module, "load_deployment_config",
        lambda *a, **k: SimpleNamespace(pii_redaction=True),
    )
    assert effective_pii_redaction(SLMConfig()) is True
    monkeypatch.setattr(
        config_module, "load_deployment_config",
        lambda *a, **k: SimpleNamespace(pii_redaction=False),
    )
    assert effective_pii_redaction(SLMConfig()) is False

    def _boom(*a, **k):
        raise OSError("unreadable")

    monkeypatch.setattr(config_module, "load_deployment_config", _boom)
    assert effective_pii_redaction(SLMConfig()) is False


def test_engine_path_key_with_personal_data_is_replaced(
    engine_with_mock_deps, monkeypatch, tmp_path,
) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    monkeypatch.setenv("SLM_PII_REDACTION", "1")
    canonical_store(
        engine_with_mock_deps, "The platform review was scheduled for Thursday afternoon.",
        source_type="python-api", trusted_actor_id=local_trusted_actor_id("python-api"),
        idempotency_key=f"call-{PHONE}", require_complete=False,
    )
    keys = [o.idempotency_key for o in
            IngestionOperationRepository(engine_with_mock_deps._db).list_operations()]
    assert keys and all(PHONE not in k and k.startswith("redacted:") for k in keys)


def test_legacy_backfill_key_with_personal_data_is_replaced(monkeypatch) -> None:
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
    item = {
        "id": 3, "content": "A note about the review.", "profile_id": "default",
        "metadata": {"_slm_idempotency_key": f"k-{PHONE}"},
    }
    with pytest.raises(RuntimeError):
        materializer.legacy_item(engine, item, actor_id="t")
    assert PHONE not in seen[0].idempotency_key
    assert seen[0].idempotency_key.startswith("redacted:")


def test_prebuilt_scrub_resets_every_derived_vector() -> None:
    from superlocalmemory.core.engine_ingestion import _prepared_prebuilt
    from superlocalmemory.storage.models import AtomicFact

    engine = MagicMock()
    engine._config.pii_redaction = True

    def fact(text: str) -> AtomicFact:
        return AtomicFact(
            fact_id="f1", memory_id="", profile_id="default", content=text,
            embedding=[0.1], fisher_mean=[0.2], fisher_variance=[0.3],
            langevin_position=[0.4],
        )

    _, scrubbed = _prepared_prebuilt(engine, fact(f"mail {EMAIL}"))
    for name in ("embedding", "fisher_mean", "fisher_variance", "langevin_position"):
        assert scrubbed[name] is None
    _, clean = _prepared_prebuilt(engine, fact("nothing personal here"))
    assert clean["embedding"] == [0.1] and clean["langevin_position"] == [0.4]
