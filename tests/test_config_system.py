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
    # SLMConfig.default() now resolves via slm_home() which honours
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


def test_mode_c_default_embedding_is_local_nomic():
    """Mode C with no embedding overrides must default to a model the local
    sentence-transformers worker can actually load. The prior default
    'text-embedding-3-large' is an OpenAI cloud model name that fails on the
    HuggingFace hub (401/RepositoryNotFound) and brings the semantic channel
    down on every recall."""
    config = SLMConfig.for_mode(Mode.C)
    assert config.embedding.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert config.embedding.dimension == 768


def test_mode_c_honors_disk_embedding_model_name():
    """Mode C must honour embedding_model_name passed in by load() from
    config.json, instead of discarding it. Regression for the load path:
    the on-disk model_name was silently overwritten by the hardcoded default."""
    config = SLMConfig.for_mode(
        Mode.C,
        embedding_model_name="BAAI/bge-m3",
        embedding_dimension=1024,
    )
    assert config.embedding.model_name == "BAAI/bge-m3"
    assert config.embedding.dimension == 1024


def test_mode_c_roundtrip_preserves_local_embedding(tmp_path):
    """Save then load a Mode C config with the default local embedding and
    confirm the reloaded model_name matches what was on disk — not the
    Mode C hardcoded default."""
    config = SLMConfig.for_mode(Mode.C)
    config.save(tmp_path / "config.json")
    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.embedding.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert reloaded.embedding.dimension == 768


def test_provider_presets_have_required_fields():
    presets = SLMConfig.provider_presets()
    for name, preset in presets.items():
        assert "base_url" in preset, f"{name} missing base_url"
        assert "model" in preset, f"{name} missing model"
        assert "env_key" in preset, f"{name} missing env_key"


# --- save()/load() must round-trip math and channel_weights ---

def test_save_persists_math_and_channel_weights(tmp_path):
    """save() must serialise the math and channel_weights sections — they are
    mode-tunable structural fields consumed by engine wiring, not internal
    defaults. Regression test."""
    config = SLMConfig.for_mode(Mode.A)
    config.math.sheaf_contradiction_threshold = 0.61
    config.channel_weights.semantic = 2.0
    config.channel_weights.hopfield = 0.4

    config.save(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text())

    assert data["math"]["sheaf_contradiction_threshold"] == 0.61
    assert data["channel_weights"]["semantic"] == 2.0
    assert data["channel_weights"]["hopfield"] == 0.4


def test_load_restores_math_and_channel_weights(tmp_path):
    """A mode-switch preset written by save() must still be active after a
    restart (load()) — previously both sections reverted to dataclass
    defaults. Regression test."""
    config = SLMConfig.for_mode(Mode.B)
    config.math.sheaf_contradiction_threshold = 0.61
    config.math.fisher_temperature = 12.5
    config.channel_weights.semantic = 2.0
    config.save(tmp_path / "config.json")

    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.math.sheaf_contradiction_threshold == 0.61
    assert reloaded.math.fisher_temperature == 12.5
    assert reloaded.channel_weights.semantic == 2.0


def test_math_langevin_weight_range_survives_roundtrip(tmp_path):
    """langevin_weight_range is a tuple in-memory; JSON serialises it as a
    list. load() must coerce it back so downstream consumers see a tuple."""
    config = SLMConfig.for_mode(Mode.A)
    config.save(tmp_path / "config.json")

    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert isinstance(reloaded.math.langevin_weight_range, tuple)
    assert reloaded.math.langevin_weight_range == config.math.langevin_weight_range


def test_invalid_math_and_channel_weights_fall_back_to_defaults(tmp_path):
    """Corrupt section values must fail open to dataclass defaults, not brick
    every slm invocation."""
    path = tmp_path / "config.json"
    config = SLMConfig.for_mode(Mode.A)
    config.save(path)
    data = json.loads(path.read_text())
    data["math"] = ["not", "a", "dict"]
    data["channel_weights"] = "also not a dict"
    path.write_text(json.dumps(data))

    reloaded = SLMConfig.load(path)
    assert reloaded.math == SLMConfig.for_mode(Mode.A).math
    assert reloaded.channel_weights == SLMConfig.for_mode(Mode.A).channel_weights
