# Copyright (c) 2026 Barry Gausden / GFO-X
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Regression tests for AIDEV-86 (broadened): dashboard config saves must not
wipe user-tuned config blocks that are absent from the dashboard payload.

The dashboard's /api/v3/mode/set and PUT /api/v3/mode endpoints only send the
fields visible in the UI. Previously, all three write-path endpoints called
for_mode() to build a fresh config and saved that — silently resetting every
block not in the payload (forgetting, injection, retrieval, consolidation,
scope, math, channel_weights, …) to hardcoded defaults.

Fix: load the full existing config first, then mutate only the fields the
dashboard sent. for_mode() is still used for first-time setup (no config.json)
and to supply mode-structural presets on a genuine mode switch.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import asyncio


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point SLMConfig at a tmp_path base_dir for the duration of the test."""
    monkeypatch.setattr(
        "superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path
    )
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    return tmp_path


def _write_config(base_dir: Path, payload: dict) -> None:
    (base_dir / "config.json").write_text(json.dumps(payload, indent=2))


def _read_config(base_dir: Path) -> dict:
    return json.loads((base_dir / "config.json").read_text())


def _make_request(body: dict):
    """Minimal async mock of a FastAPI Request with a json() coroutine."""
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    request.app.state = MagicMock(spec=[])  # no .engine attribute
    return request


# ── Helpers ──────────────────────────────────────────────────────────────────

_CUSTOM_FORGETTING = {
    "enabled": True,
    "half_life_days": 42,
    "min_importance": 0.1,
    "max_age_days": 999,
}

_CUSTOM_RETRIEVAL = {
    "use_cross_encoder": True,
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "cross_encoder_backend": "onnx",
    "semantic_top_k": 55,
    "agentic_max_rounds": 3,
}

_CUSTOM_EMBEDDING = {
    "model_name": "nomic-ai/nomic-embed-text-v1.5",
    "dimension": 768,
    "provider": "openai",
    "api_endpoint": "https://custom-embedding.example.com/v1",
    "api_key": "sk-custom-key",
    "deployment_name": "",
}

_BASE_CONFIG = {
    "mode": "c",
    "active_profile": "myprofile",
    "llm": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "api_key": "sk-existing",
        "base_url": "https://openrouter.example.com",
    },
    "embedding": _CUSTOM_EMBEDDING,
    "retrieval": _CUSTOM_RETRIEVAL,
    "forgetting": _CUSTOM_FORGETTING,
}


# ── set_full_config (POST /api/v3/mode/set) ──────────────────────────────────

class TestSetFullConfig:
    """POST /api/v3/mode/set must preserve all non-dashboard config blocks."""

    def test_embedding_preserved_when_omitted(self, isolated_config):
        """Embedding config survives a dashboard save that omits embedding fields."""
        _write_config(isolated_config, _BASE_CONFIG)

        dashboard_payload = {
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-new-key",
            "base_url": "https://openrouter.example.com",
            # embedding_* fields intentionally absent
        }

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request(dashboard_payload)))

        saved = _read_config(isolated_config)
        assert saved["embedding"]["provider"] == "openai"
        assert saved["embedding"]["api_endpoint"] == "https://custom-embedding.example.com/v1"
        assert saved["embedding"]["model_name"] == "nomic-ai/nomic-embed-text-v1.5"
        assert saved["embedding"]["dimension"] == 768

    def test_forgetting_preserved_when_omitted(self, isolated_config):
        """User-tuned forgetting config survives a dashboard settings save."""
        _write_config(isolated_config, _BASE_CONFIG)

        dashboard_payload = {
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-updated",
            "base_url": "https://openrouter.example.com",
        }

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request(dashboard_payload)))

        saved = _read_config(isolated_config)
        fg = saved.get("forgetting", {})
        assert fg.get("half_life_days") == 42, \
            f"forgetting.half_life_days was reset; got {fg}"
        assert fg.get("max_age_days") == 999

    def test_retrieval_cross_encoder_preserved_on_same_mode_save(self, isolated_config):
        """Retrieval cross-encoder settings (serialised by save()) survive a dashboard save."""
        _write_config(isolated_config, _BASE_CONFIG)

        dashboard_payload = {
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-updated",
            "base_url": "https://openrouter.example.com",
        }

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request(dashboard_payload)))

        saved = _read_config(isolated_config)
        rt = saved.get("retrieval", {})
        assert rt.get("cross_encoder_backend") == "onnx", \
            f"retrieval.cross_encoder_backend was reset; got {rt}"
        assert rt.get("use_cross_encoder") is True

    def test_active_profile_preserved(self, isolated_config):
        """active_profile survives a dashboard settings save."""
        _write_config(isolated_config, _BASE_CONFIG)

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request({
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-new",
            "base_url": "https://openrouter.example.com",
        })))

        saved = _read_config(isolated_config)
        assert saved.get("active_profile") == "myprofile"

    def test_explicit_embedding_fields_are_applied(self, isolated_config):
        """When the dashboard does send embedding fields, they are applied."""
        _write_config(isolated_config, _BASE_CONFIG)

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request({
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-new",
            "base_url": "https://openrouter.example.com",
            "embedding_provider": "openai",
            "embedding_endpoint": "https://new-embedding.example.com/v1",
            "embedding_key": "sk-emb-new",
            "embedding_model": "text-embedding-3-large",
            "embedding_dimension": 3072,
        })))

        saved = _read_config(isolated_config)
        assert saved["embedding"]["api_endpoint"] == "https://new-embedding.example.com/v1"
        assert saved["embedding"]["model_name"] == "text-embedding-3-large"
        assert saved["embedding"]["dimension"] == 3072

    def test_mode_switch_updates_mode_and_preserves_other_config(self, isolated_config):
        """Switching mode via set_full_config persists the new mode and preserves non-dashboard blocks."""
        _write_config(isolated_config, {**_BASE_CONFIG, "mode": "c"})

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request({
            "mode": "a",
            "provider": "",
            "model": "",
            "api_key": "",
            "base_url": "",
        })))

        saved = _read_config(isolated_config)
        assert saved["mode"] == "a"
        # Embedding and forgetting (persisted blocks) must still survive the mode switch
        assert saved["embedding"]["provider"] == "openai"
        fg = saved.get("forgetting", {})
        assert fg.get("half_life_days") == 42, \
            f"forgetting wiped on mode switch; got {fg}"

    def test_first_time_setup_no_existing_config(self, isolated_config):
        """Works correctly when no config.json exists yet (first-time setup)."""
        # No config.json written — isolated_config directory is empty
        assert not (isolated_config / "config.json").exists()

        from superlocalmemory.server.routes.v3_api import set_full_config

        asyncio.run(set_full_config(_make_request({
            "mode": "c",
            "provider": "openrouter",
            "model": "anthropic/claude-sonnet-4",
            "api_key": "sk-firsttime",
            "base_url": "https://openrouter.example.com",
        })))

        saved = _read_config(isolated_config)
        assert saved["mode"] == "c"
        assert saved["llm"]["provider"] == "openrouter"
        assert saved["llm"]["api_key"] == "sk-firsttime"


# ── set_mode (PUT /api/v3/mode) ──────────────────────────────────────────────

class TestSetMode:
    """PUT /api/v3/mode must apply mode structural presets but preserve user config."""

    def test_forgetting_preserved_on_mode_switch(self, isolated_config):
        """User-tuned forgetting survives a mode switch via PUT /api/v3/mode."""
        _write_config(isolated_config, {**_BASE_CONFIG, "mode": "b",
                                        "llm": {**_BASE_CONFIG["llm"], "api_key": "sk-b"}})

        from superlocalmemory.server.routes.v3_api import set_mode

        asyncio.run(set_mode(_make_request({"mode": "c"})))

        saved = _read_config(isolated_config)
        fg = saved.get("forgetting", {})
        assert fg.get("half_life_days") == 42, \
            f"forgetting config reset during mode switch; got {fg}"

    def test_embedding_preserved_on_mode_switch(self, isolated_config):
        """Custom embedding survives a mode switch via PUT /api/v3/mode."""
        _write_config(isolated_config, {**_BASE_CONFIG, "mode": "b",
                                        "llm": {**_BASE_CONFIG["llm"], "api_key": "sk-b"}})

        from superlocalmemory.server.routes.v3_api import set_mode

        asyncio.run(set_mode(_make_request({"mode": "c"})))

        saved = _read_config(isolated_config)
        assert saved["embedding"]["provider"] == "openai"
        assert saved["embedding"]["api_endpoint"] == "https://custom-embedding.example.com/v1"

    def test_mode_persisted_and_user_config_preserved_on_switch(self, isolated_config):
        """Mode switch via PUT /api/v3/mode persists the new mode and preserves user config."""
        _write_config(isolated_config, {**_BASE_CONFIG, "mode": "b",
                                        "llm": {**_BASE_CONFIG["llm"], "api_key": "sk-b"}})

        from superlocalmemory.server.routes.v3_api import set_mode

        asyncio.run(set_mode(_make_request({"mode": "c"})))

        saved = _read_config(isolated_config)
        assert saved["mode"] == "c"
        # Embedding and forgetting must survive the mode switch
        assert saved["embedding"]["provider"] == "openai"
        fg = saved.get("forgetting", {})
        assert fg.get("half_life_days") == 42, \
            f"forgetting wiped on mode switch; got {fg}"


# ── set_provider (PUT /api/v3/provider) ──────────────────────────────────────

class TestSetProvider:
    """PUT /api/v3/provider must preserve all non-LLM config blocks."""

    def test_embedding_preserved_on_provider_change(self, isolated_config):
        """Custom embedding survives a provider-only update."""
        _write_config(isolated_config, _BASE_CONFIG)

        from superlocalmemory.server.routes.v3_api import set_provider

        asyncio.run(set_provider(_make_request({
            "provider": "anthropic",
            "api_key": "sk-ant-new",
            "model": "claude-opus-4-7",
            "base_url": "",
        })))

        saved = _read_config(isolated_config)
        assert saved["embedding"]["provider"] == "openai"
        assert saved["embedding"]["api_endpoint"] == "https://custom-embedding.example.com/v1"
        assert saved["llm"]["provider"] == "anthropic"

    def test_forgetting_preserved_on_provider_change(self, isolated_config):
        """User-tuned forgetting survives a provider-only update."""
        _write_config(isolated_config, _BASE_CONFIG)

        from superlocalmemory.server.routes.v3_api import set_provider

        asyncio.run(set_provider(_make_request({
            "provider": "anthropic",
            "api_key": "sk-ant-new",
            "model": "claude-opus-4-7",
            "base_url": "",
        })))

        saved = _read_config(isolated_config)
        fg = saved.get("forgetting", {})
        assert fg.get("half_life_days") == 42, \
            f"forgetting config reset during provider change; got {fg}"


# ── for_mode baseline (unchanged contract) ───────────────────────────────────

class TestForModeContract:
    """Mode defaults preserve an explicit boundary for paid cloud embeddings."""

    def test_for_mode_c_empty_embedding_keeps_local_fallback(self):
        """Mode C requires an explicit endpoint before selecting cloud embeddings."""
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        config = SLMConfig.for_mode(
            Mode("c"),
            llm_provider="openrouter",
            llm_model="anthropic/claude-sonnet-4",
            llm_api_key="sk-key",
            llm_api_base="https://example.com",
            embedding_provider="",
            embedding_endpoint="",
            embedding_key="",
            embedding_model_name="",
            embedding_dimension=0,
        )

        assert config.embedding.model_name == "nomic-ai/nomic-embed-text-v1.5"
        assert config.embedding.dimension == 768
