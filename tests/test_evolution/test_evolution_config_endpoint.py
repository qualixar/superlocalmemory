# v3.7.9: POST /api/evolution/config — dashboard write endpoint for evolution.
import asyncio
import inspect
import json

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from superlocalmemory.server.routes import evolution as evo_route
from superlocalmemory.server.routes.evolution import EvolutionConfigUpdate


def _call(body):
    request = Request({"type": "http", "app": FastAPI(), "headers": []})
    result = evo_route.evolution_config(request, body)
    return asyncio.run(result) if inspect.isawaitable(result) else result


@pytest.fixture(autouse=True)
def allow_manage(monkeypatch):
    """Unit tests focus on persistence; the authorization call is tested below."""
    monkeypatch.setattr(evo_route, "_require_manage", lambda request: None)


def test_valid_update_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(evo_route, "MEMORY_DIR", tmp_path)
    res = _call(EvolutionConfigUpdate(enabled=True, mutation_model="haiku"))
    assert res["ok"] is True
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["evolution"]["enabled"] is True
    assert saved["evolution"]["mutation_model"] == "haiku"


def test_invalid_model_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(evo_route, "MEMORY_DIR", tmp_path)
    res = _call(EvolutionConfigUpdate(mutation_model="gpt-5-ultra"))
    assert res["ok"] is False
    assert "must be one of" in res["error"]
    # nothing written
    assert not (tmp_path / "config.json").exists()


def test_auto_normalized_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(evo_route, "MEMORY_DIR", tmp_path)
    res = _call(EvolutionConfigUpdate(verify_model="auto"))
    assert res["ok"] is True
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["evolution"]["verify_model"] == ""


def test_partial_update_preserves_other_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(evo_route, "MEMORY_DIR", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(
        {"mode": "b", "evolution": {"enabled": True, "backend": "ollama"}}))
    res = _call(EvolutionConfigUpdate(mutation_model="sonnet"))
    assert res["ok"] is True
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["mode"] == "b"                          # untouched
    assert saved["evolution"]["backend"] == "ollama"     # untouched
    assert saved["evolution"]["mutation_model"] == "sonnet"


def test_post_config_requires_manage_permission(tmp_path, monkeypatch):
    """Evolution settings are an administrative daemon mutation."""
    monkeypatch.setattr(evo_route, "MEMORY_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(evo_route, "_require_manage", lambda request: calls.append(request))

    result = _call(EvolutionConfigUpdate(backend="auto"))

    assert result["ok"] is True
    assert len(calls) == 1
