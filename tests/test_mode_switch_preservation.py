# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.43

"""Regression tests for `slm mode <X>` CLI preservation behavior.

v3.4.34 added `mode_change=True` protection for the ``mode`` field itself.
But the CLI handler still called ``SLMConfig.for_mode(...)`` passing only
``llm_*`` kwargs — silently re-deriving embedding / retrieval / evolution /
forgetting / math from mode defaults on every switch. The user's
customizations (e.g. tuned cross_encoder_model, custom forgetting
half-lives, custom embedding endpoint) were lost on `slm mode b`.

v3.4.43 fix: `cmd_mode` now mutates only ``config.mode`` (plus optional
LLM defaults if the user has no provider configured). Everything else
is preserved byte-for-byte.

These tests assert the preservation contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point SLMConfig at a tmp_path base_dir for the duration of the test."""
    monkeypatch.setattr(
        "superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path
    )
    # Also ensure SLMConfig.load() uses tmp_path
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    return tmp_path


def _write_config(base_dir: Path, payload: dict) -> None:
    """Write a config.json directly so we control the starting state."""
    (base_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_config(base_dir: Path) -> dict:
    return json.loads((base_dir / "config.json").read_text(encoding="utf-8"))


def _invoke_mode_cli(value: str | None = None) -> None:
    """Invoke `slm mode [value]` programmatically."""
    from argparse import Namespace

    from superlocalmemory.cli.commands import cmd_mode

    args = Namespace(value=value, json=False)
    cmd_mode(args)


# --------------------------------------------------------------------------
# Core regression: embedding + retrieval + forgetting preserved on mode switch
# --------------------------------------------------------------------------


class TestModeSwitchPreservation:
    """v3.4.43 contract: `slm mode <X>` preserves ALL non-mode settings."""

    def test_mode_switch_a_to_b_preserves_custom_embedding(self, isolated_config):
        """User's customized embedding model must survive a mode switch."""
        _write_config(isolated_config, {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": "", "api_key": "", "base_url": ""},
            "embedding": {
                "model_name": "custom/my-special-embed-v99",
                "dimension": 1024,
                "provider": "openai",
                "api_endpoint": "https://custom.example.com/v1",
                "api_key": "",
                "deployment_name": "",
            },
            "retrieval": {
                "use_cross_encoder": True,
                "cross_encoder_model": "user/custom-reranker",
                "cross_encoder_backend": "onnx",
            },
        })

        _invoke_mode_cli("b")

        result = _read_config(isolated_config)
        assert result["mode"] == "b"
        # Embedding MUST be unchanged
        assert result["embedding"]["model_name"] == "custom/my-special-embed-v99"
        assert result["embedding"]["dimension"] == 1024
        assert result["embedding"]["provider"] == "openai"
        assert result["embedding"]["api_endpoint"] == "https://custom.example.com/v1"
        # Retrieval MUST be unchanged
        assert result["retrieval"]["cross_encoder_model"] == "user/custom-reranker"
        assert result["retrieval"]["cross_encoder_backend"] == "onnx"

    def test_mode_switch_b_to_a_preserves_custom_retrieval(self, isolated_config):
        """Switching B→A must not reset the cross-encoder model."""
        _write_config(isolated_config, {
            "mode": "b",
            "active_profile": "default",
            "llm": {
                "provider": "ollama", "model": "llama3.2",
                "api_key": "", "base_url": "http://127.0.0.1:11434",
            },
            "embedding": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "dimension": 768,
                "provider": "sentence-transformers",
                "api_endpoint": "", "api_key": "", "deployment_name": "",
            },
            "retrieval": {
                "use_cross_encoder": True,
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "cross_encoder_backend": "",
            },
        })

        _invoke_mode_cli("a")

        result = _read_config(isolated_config)
        assert result["mode"] == "a"
        assert result["retrieval"]["cross_encoder_model"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"
        assert result["embedding"]["model_name"] == "nomic-ai/nomic-embed-text-v1.5"

    def test_mode_switch_preserves_existing_llm_provider(self, isolated_config):
        """If user has LLM provider set, mode switch must NOT overwrite it."""
        _write_config(isolated_config, {
            "mode": "b",
            "active_profile": "default",
            "llm": {
                "provider": "ollama",
                "model": "llama3.2",
                "api_key": "",
                "base_url": "http://127.0.0.1:11434",
            },
            "embedding": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "dimension": 768,
                "provider": "sentence-transformers",
                "api_endpoint": "", "api_key": "", "deployment_name": "",
            },
            "retrieval": {"use_cross_encoder": True,
                          "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                          "cross_encoder_backend": ""},
        })

        # Switch B→C — should NOT auto-grab any env vars or change llm.provider
        _invoke_mode_cli("c")

        result = _read_config(isolated_config)
        assert result["mode"] == "c"
        # llm provider untouched
        assert result["llm"]["provider"] == "ollama"
        assert result["llm"]["model"] == "llama3.2"
        assert result["llm"]["base_url"] == "http://127.0.0.1:11434"

    def test_mode_switch_populates_llm_defaults_when_empty(self, isolated_config):
        """If user has NO provider set and switches to B/C, populate defaults."""
        _write_config(isolated_config, {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": "", "api_key": "", "base_url": ""},
            "embedding": {"model_name": "nomic-ai/nomic-embed-text-v1.5",
                          "dimension": 768, "provider": "sentence-transformers",
                          "api_endpoint": "", "api_key": "", "deployment_name": ""},
            "retrieval": {"use_cross_encoder": True,
                          "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                          "cross_encoder_backend": ""},
        })

        _invoke_mode_cli("b")

        result = _read_config(isolated_config)
        assert result["mode"] == "b"
        # Defaults populated because user had nothing
        assert result["llm"]["provider"] != ""
        # But embedding is still the user's value (sentence-transformers, not re-derived)
        assert result["embedding"]["provider"] == "sentence-transformers"

    def test_mode_get_does_not_modify_config(self, isolated_config):
        """`slm mode` (no value) should READ only, never write."""
        original = {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": "", "api_key": "", "base_url": ""},
            "embedding": {"model_name": "marker", "dimension": 768,
                          "provider": "sentence-transformers",
                          "api_endpoint": "", "api_key": "", "deployment_name": ""},
            "retrieval": {"use_cross_encoder": True,
                          "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                          "cross_encoder_backend": ""},
        }
        _write_config(isolated_config, original)

        _invoke_mode_cli(None)  # read-only

        result = _read_config(isolated_config)
        # Embedding model_name unchanged — proves no write happened
        assert result["embedding"]["model_name"] == "marker"


class TestModeSwitchJsonOutput:
    """JSON path must also preserve everything."""

    def test_json_mode_switch_preserves_embedding(self, isolated_config, capsys):
        """JSON CLI path must preserve embedding identical to the human path.

        Uses an HF-style custom name (org/name) + provider=openai with
        api_endpoint, which is the path SLMConfig.load() preserves
        verbatim (per V3.4.24). Plain unknown names get coerced to mode
        defaults at load time — that's a load() concern, not the mode
        switch concern under test here.
        """
        from argparse import Namespace

        from superlocalmemory.cli.commands import cmd_mode

        _write_config(isolated_config, {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": "", "api_key": "", "base_url": ""},
            "embedding": {
                "model_name": "my-org/preserve-me-please-v2",
                "dimension": 1024,
                "provider": "openai",
                "api_endpoint": "https://custom.example.com/v1",
                "api_key": "",
                "deployment_name": "",
            },
            "retrieval": {"use_cross_encoder": True,
                          "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                          "cross_encoder_backend": ""},
        })

        args = Namespace(value="b", json=True)
        cmd_mode(args)

        result = _read_config(isolated_config)
        assert result["mode"] == "b"
        assert result["embedding"]["model_name"] == "my-org/preserve-me-please-v2"
        assert result["embedding"]["dimension"] == 1024
        assert result["embedding"]["provider"] == "openai"
        assert result["embedding"]["api_endpoint"] == "https://custom.example.com/v1"


class TestNoEmbeddingChangedWarning:
    """The pre-v3.4.43 'Embedding model changed' warning fired on every mode
    switch because for_mode re-derived. With preservation, it must not fire."""

    def test_no_embedding_warning_on_routine_mode_switch(self, isolated_config, capsys):
        _write_config(isolated_config, {
            "mode": "a",
            "active_profile": "default",
            "llm": {"provider": "", "model": "", "api_key": "", "base_url": ""},
            "embedding": {"model_name": "nomic-ai/nomic-embed-text-v1.5",
                          "dimension": 768, "provider": "sentence-transformers",
                          "api_endpoint": "", "api_key": "", "deployment_name": ""},
            "retrieval": {"use_cross_encoder": True,
                          "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                          "cross_encoder_backend": ""},
        })

        _invoke_mode_cli("a")  # noop switch — should produce zero warnings

        out = capsys.readouterr().out
        assert "Embedding model changed" not in out


# --------------------------------------------------------------------------
# 4.2.9: a setting saved in one mode survives a switch away and back
# --------------------------------------------------------------------------


class TestSavedSettingSurvivesModeRoundTrip:
    """Every writer (dashboard, CLI, re-index activation) saved config.json
    only. switch_mode() restores mode_<x>.json when it exists, so a model or
    embedder changed in Mode B came back as the old one after B -> C -> B."""

    def _start_in_b(self, base_dir: Path):
        import dataclasses

        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        config = SLMConfig.for_mode(Mode.B, base_dir=base_dir)
        config.llm = dataclasses.replace(config.llm, provider="ollama", model="gemma4:e2b")
        config.save(mode_change=True)
        SLMConfig.switch_mode("c", base_dir=base_dir)   # mode_b.json now holds gemma4
        SLMConfig.switch_mode("b", base_dir=base_dir)
        return SLMConfig.load(base_dir / "config.json")

    def test_llm_model_survives_b_c_b(self, isolated_config):
        import dataclasses

        from superlocalmemory.core.config import SLMConfig

        config = self._start_in_b(isolated_config)
        config.llm = dataclasses.replace(config.llm, model="gemma3:4b")
        config.save()   # what the dashboard and the wizard do

        SLMConfig.switch_mode("c", base_dir=isolated_config)
        back = SLMConfig.switch_mode("b", base_dir=isolated_config)
        assert back.llm.model == "gemma3:4b"
        assert _read_config(isolated_config)["llm"]["model"] == "gemma3:4b"

    def test_same_mode_switch_keeps_a_setting_saved_since(self, isolated_config):
        """`slm mode b` while already in B skipped saving config.json to
        mode_b.json but still loaded mode_b.json - reverting any change made
        since the last switch (the dashboard, the wizard, `slm provider set`)."""
        import dataclasses

        from superlocalmemory.core.config import SLMConfig

        config = self._start_in_b(isolated_config)
        config.llm = dataclasses.replace(config.llm, model="gemma3:4b")
        config.save()

        back = SLMConfig.switch_mode("b", base_dir=isolated_config)
        assert back.llm.model == "gemma3:4b"
        assert _read_config(isolated_config)["llm"]["model"] == "gemma3:4b"
        profile = SLMConfig.load(isolated_config / "mode_b.json")
        assert profile.llm.model == "gemma3:4b"
