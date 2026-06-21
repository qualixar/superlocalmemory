# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for LLM provider layer — Task 1 of V3 build."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from superlocalmemory.llm.backbone import LLMBackbone, _SUPPORTED_PROVIDERS
from superlocalmemory.core.config import SLMConfig, LLMConfig
from superlocalmemory.storage.models import Mode


def test_openrouter_in_supported_providers():
    assert "openrouter" in _SUPPORTED_PROVIDERS


def test_openai_provider_init():
    config = LLMConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-test")
    backbone = LLMBackbone(config)
    assert backbone.provider == "openai"
    assert backbone.model == "gpt-4.1-mini"
    assert backbone.is_available()


def test_openrouter_provider_init():
    config = LLMConfig(provider="openrouter", model="openai/gpt-4.1-mini", api_key="sk-or-test")
    backbone = LLMBackbone(config)
    assert backbone.provider == "openrouter"
    assert backbone.is_available()


def test_anthropic_provider_init():
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-ant-test")
    backbone = LLMBackbone(config)
    assert backbone.provider == "anthropic"
    assert backbone.is_available()


def test_anthropic_provider_respects_api_base():
    """api_base must be used as the request URL for the anthropic provider.

    Regression guard: _base_url is stored from config.api_base in __init__
    but _build_anthropic previously ignored it, always returning
    _ANTHROPIC_URL. This test verifies the built URL uses api_base when set.
    """
    proxy = "https://my-proxy.example.com"
    config = LLMConfig(
        provider="anthropic", model="claude-sonnet-4-6",
        api_key="sk-ant-test", api_base=proxy,
    )
    backbone = LLMBackbone(config)
    url, _, _ = backbone._build_anthropic("hi", "", 100, 0.0)
    assert url == proxy + "/v1/messages"


def test_anthropic_provider_default_url_without_api_base():
    from superlocalmemory.llm.backbone import _ANTHROPIC_URL
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-ant-test")
    backbone = LLMBackbone(config)
    url, _, _ = backbone._build_anthropic("hi", "", 100, 0.0)
    assert url == _ANTHROPIC_URL


def test_ollama_provider_no_key_needed():
    config = LLMConfig(provider="ollama", model="llama3.2")
    backbone = LLMBackbone(config)
    assert backbone.provider == "ollama"
    assert backbone.is_available()


def test_no_provider_is_not_available():
    config = LLMConfig()
    backbone = LLMBackbone(config)
    assert not backbone.is_available()


def test_unsupported_provider_raises():
    config = LLMConfig(provider="invalid_provider")
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMBackbone(config)


def test_config_load_default_when_no_file(tmp_path):
    config = SLMConfig.load(tmp_path / "nonexistent.json")
    assert config.mode == Mode.A
    assert config.llm.provider == ""


def test_config_save_and_reload(tmp_path):
    config = SLMConfig.for_mode(
        Mode.C,
        llm_provider="openrouter",
        llm_model="openai/gpt-4.1-mini",
        llm_api_key="sk-or-test123",
        llm_api_base="https://openrouter.ai/api/v1",
    )
    config_path = tmp_path / "config.json"
    config.save(config_path)

    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["mode"] == "c"
    assert data["llm"]["provider"] == "openrouter"

    reloaded = SLMConfig.load(config_path)
    assert reloaded.mode == Mode.C
    assert reloaded.llm.provider == "openrouter"
    assert reloaded.llm.model == "openai/gpt-4.1-mini"


def test_config_provider_presets():
    presets = SLMConfig.provider_presets()
    assert "openai" in presets
    assert "anthropic" in presets
    assert "ollama" in presets
    assert "openrouter" in presets
    assert presets["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"


def test_mode_a_config_has_no_llm():
    config = SLMConfig.for_mode(Mode.A)
    assert config.llm.provider == ""
    assert not config.llm.is_available
