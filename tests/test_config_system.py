"""Tests for Config System — Task 4 of V3 build."""
import json
import pytest
from pathlib import Path
from superlocalmemory.core.config import SLMConfig, LLMConfig
from superlocalmemory.storage.models import Mode


def test_default_is_mode_a():
    config = SLMConfig.default()
    assert config.mode == Mode.A
    assert config.llm.provider == ""


def test_default_base_dir(monkeypatch):
    # WP-07: SLMConfig.default() now resolves via slm_home() which honours
    # SLM_DATA_DIR. When SLM_DATA_DIR is absent the fallback is ~/.superlocalmemory.
    monkeypatch.delenv("SLM_DATA_DIR", raising=False)
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)
    config = SLMConfig.default()
    assert ".superlocalmemory" in str(config.base_dir)


def test_save_includes_active_profile(tmp_path):
    config = SLMConfig.for_mode(Mode.A)
    config.active_profile = "work"
    config.save(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["active_profile"] == "work"


def test_load_restores_active_profile(tmp_path):
    config = SLMConfig.for_mode(Mode.B, llm_provider="ollama", llm_model="llama3.2")
    config.active_profile = "personal"
    config.save(tmp_path / "config.json")
    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.active_profile == "personal"
    assert reloaded.mode == Mode.B


def test_load_missing_profile_defaults_to_default(tmp_path):
    # Write a config without active_profile field
    (tmp_path / "config.json").write_text('{"mode": "a", "llm": {"provider": ""}}')
    config = SLMConfig.load(tmp_path / "config.json")
    assert config.active_profile == "default"


def test_mode_c_with_openrouter(tmp_path):
    config = SLMConfig.for_mode(
        Mode.C,
        llm_provider="openrouter",
        llm_model="openai/gpt-4.1-mini",
        llm_api_key="sk-or-test",
        llm_api_base="https://openrouter.ai/api/v1",
    )
    config.save(tmp_path / "config.json")
    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.llm.provider == "openrouter"
    assert reloaded.llm.model == "openai/gpt-4.1-mini"


def test_mode_b_ollama_defaults():
    config = SLMConfig.for_mode(Mode.B)
    assert config.llm.provider == "ollama"
    assert "11434" in config.llm.api_base


def test_config_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "config.json"
    config = SLMConfig.default()
    config.save(nested)
    assert nested.exists()


def test_provider_presets_have_required_fields():
    presets = SLMConfig.provider_presets()
    for name, preset in presets.items():
        assert "base_url" in preset, f"{name} missing base_url"
        assert "model" in preset, f"{name} missing model"
        assert "env_key" in preset, f"{name} missing env_key"
