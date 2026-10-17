"""Documented provider commands must not require a terminal prompt."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def test_provider_set_named_provider_uses_noninteractive_path() -> None:
    from superlocalmemory.cli.commands import cmd_provider

    args = Namespace(action="set", provider="openrouter")
    with patch(
        "superlocalmemory.cli.setup_wizard.configure_provider",
    ) as configure_provider:
        cmd_provider(args)

    configure_provider.assert_called_once()
    _config, = configure_provider.call_args.args
    assert _config is not None
    # #112: cmd_provider also forwards the custom-endpoint fields
    # (endpoint/api_key/model/target_mode) — None here because `args` (built
    # by the CLI's own argparser) carries them as None when provider=custom
    # is not selected. configure_provider only consults them for provider="custom".
    assert configure_provider.call_args.kwargs == {
        "provider_name": "openrouter",
        "endpoint": None,
        "api_key": None,
        "model": None,
        "target_mode": None,
    }


def test_named_provider_preserves_existing_runtime_configuration(tmp_path, monkeypatch) -> None:
    """Changing a provider cannot reset a promoted scale-engine profile."""
    from superlocalmemory.cli.setup_wizard import configure_provider

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    config.graph_backend = "cozo"
    config.vector_backend = "lancedb"
    config.scale_engine_state = "promoted"
    config.retrieval.semantic_top_k = 73
    config.save(mode_change=True)

    configure_provider(config, provider_name="openrouter")

    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.mode is Mode.C
    assert reloaded.llm.provider == "openrouter"
    assert reloaded.graph_backend == "cozo"
    assert reloaded.vector_backend == "lancedb"
    assert reloaded.scale_engine_state == "promoted"
    assert reloaded.retrieval.semantic_top_k == 73


def test_provider_set_ollama_honours_mode_and_model(tmp_path, monkeypatch) -> None:
    """4.2.9: `slm provider set ollama --mode b --model gemma3:4b` saved Mode C
    with the preset model - --mode and --model were read only for 'custom'.
    The per-mode file kept the old model, so a later `slm mode b` restored it."""
    from superlocalmemory.cli.setup_wizard import configure_provider

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    config.save(mode_change=True)

    configure_provider(config, provider_name="ollama", model="qwen2.5:7b",
                       target_mode="b")

    reloaded = SLMConfig.load(tmp_path / "config.json")
    assert reloaded.mode is Mode.B
    assert (reloaded.llm.provider, reloaded.llm.model) == ("ollama", "qwen2.5:7b")
    assert SLMConfig.read_current_mode(tmp_path) == "b"
    profile = SLMConfig.load(SLMConfig._mode_config_path(tmp_path, Mode.B))
    assert profile.llm.model == "qwen2.5:7b"


def test_provider_set_ollama_defaults_to_mode_b(tmp_path) -> None:
    """A local model is Mode B unless the operator asks for C."""
    from superlocalmemory.cli.setup_wizard import configure_provider

    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    configure_provider(config, provider_name="ollama")
    assert SLMConfig.load(tmp_path / "config.json").mode is Mode.B


def test_provider_set_rejects_an_unknown_mode(tmp_path) -> None:
    import pytest
    from superlocalmemory.cli.setup_wizard import configure_provider

    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    with pytest.raises(ValueError, match="mode"):
        configure_provider(config, provider_name="ollama", target_mode="a")
