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
    assert configure_provider.call_args.kwargs == {"provider_name": "openrouter"}


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
