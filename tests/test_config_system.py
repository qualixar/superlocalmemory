"""Tests for Config System — Task 4 of V3 build."""
import json

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def test_default_is_mode_a():
    config = SLMConfig.default()
    assert config.mode == Mode.A
    assert config.llm.provider == ""


def test_default_base_dir(monkeypatch, tmp_path):
    # SLMConfig.default() now resolves via slm_home() which honours
    # SLM_DATA_DIR. When SLM_DATA_DIR is absent the fallback is ~/.superlocalmemory.
    # HOME points at a temporary folder: default() also reads config.json and
    # answer_check.json there, and under the parallel CI gate (-n 4) the real
    # home it fell back to was the guarded live data root (v4.2.5 gate, 14 Oct).
    monkeypatch.delenv("SLM_DATA_DIR", raising=False)
    monkeypatch.delenv("SL_MEMORY_PATH", raising=False)
    monkeypatch.delenv("SLM_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = SLMConfig.default()
    assert config.base_dir.resolve() == (tmp_path / ".superlocalmemory").resolve()


def test_save_includes_active_profile(tmp_path):
    config = SLMConfig.for_mode(Mode.A)
    config.active_profile = "work"
    config.save(tmp_path / "config.json")
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
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
    (tmp_path / "config.json").write_text('{"mode": "a", "llm": {"provider": ""}}', encoding="utf-8")
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


def test_migrate_to_3mode_defaults_a_fresh_install_to_mode_a(tmp_path):
    # Every CI runner, and every user's first boot, starts here: no
    # config.json has ever been written. This used to default to Mode B
    # pointed at http://localhost:11434 for both the LLM and embeddings,
    # with no reachability check before use (LLMBackbone.is_available()
    # always returns True for the "ollama" provider), so a machine with no
    # local model silently got Mode B anyway — breaking Mode A's own
    # promise and leaving enrichment unable to ever reach "complete" when
    # nothing answers on that port. This pins the fix: a fresh install
    # matches SLMConfig.default()'s own Mode A, everywhere else in this
    # module.
    assert not (tmp_path / "config.json").exists()
    migrated = SLMConfig.migrate_to_3mode(tmp_path)
    assert migrated is True
    config = SLMConfig.load(tmp_path / "config.json")
    assert config.mode == Mode.A
    assert config.llm.provider == ""
    assert SLMConfig.read_current_mode(tmp_path) == "a"


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
    config.json, instead of discarding it. Regression for AIDEV-86 load path:
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
