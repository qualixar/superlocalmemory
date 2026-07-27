"""HTTP boundary contracts for the daemon-owned canonical remember path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from starlette.requests import Request


def _remember_endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/remember"
    )


def _request(app) -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/remember",
        "headers": [],
        "app": app,
    })


def test_remember_runs_trust_hook_before_journal_runtime_and_ignores_wait(
    tmp_path, monkeypatch,
) -> None:
    """wait=true cannot revive the legacy inline materializer path."""
    from superlocalmemory.core.remember_admission import RememberReceipt
    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    sequence: list[str] = []

    class Hooks:
        def run_pre(self, operation, payload):
            assert operation == "store"
            assert payload["agent_id"] == "trusted-actor"
            sequence.append("hook")

    class Runtime:
        def remember(self, admission, actor, *, deadline_ms):
            assert sequence == ["hook"]
            assert admission.content == "canonical route witness"
            assert actor.principal_id == "trusted-actor"
            assert deadline_ms == 2_000
            sequence.append("journal-coordinator")
            return RememberReceipt({
                "operation_id": "operation-1",
                "pending_id": "operation-1",
                "fact_ids": ["fact-1"],
                "materialization_state": "queryable",
                "commit_sequence": 7,
            })

    monkeypatch.setattr(
        "superlocalmemory.server.write_identity.require_write_actor",
        lambda *_args, **_kwargs: "trusted-actor",
    )
    app = unified_daemon.create_app()
    app.state.engine = SimpleNamespace(
        _profile_id="default",
        _config=SimpleNamespace(scope=SimpleNamespace(default_scope="personal")),
        _hooks=Hooks(),
    )
    app.state.canonical_remember_runtime = Runtime()

    result = asyncio.run(_remember_endpoint(app)(
        unified_daemon.RememberRequest(content="canonical route witness"),
        _request(app),
        wait=True,
    ))

    assert sequence == ["hook", "journal-coordinator"]
    assert result["status"] == "queryable"
    assert result["wait_ignored"] is True
    assert result["commit_sequence"] == 7


def test_remember_returns_retryable_503_when_canonical_writer_is_unavailable(
    tmp_path, monkeypatch,
) -> None:
    """A missing writer never falls back to a direct engine write."""
    from fastapi import HTTPException

    from superlocalmemory.server import unified_daemon

    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "superlocalmemory.server.write_identity.require_write_actor",
        lambda *_args, **_kwargs: "trusted-actor",
    )
    app = unified_daemon.create_app()
    app.state.engine = SimpleNamespace(
        _profile_id="default",
        _config=SimpleNamespace(scope=SimpleNamespace(default_scope="personal")),
        _hooks=SimpleNamespace(run_pre=lambda *_args, **_kwargs: None),
    )
    app.state.canonical_remember_runtime = None

    try:
        asyncio.run(_remember_endpoint(app)(
            unified_daemon.RememberRequest(content="unavailable witness"),
            _request(app),
        ))
    except HTTPException as exc:
        assert exc.status_code == 503
    else:  # pragma: no cover - explicit fail-closed contract
        raise AssertionError("canonical remember unexpectedly fell back to an engine write")
