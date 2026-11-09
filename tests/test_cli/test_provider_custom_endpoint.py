# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""`slm provider set custom` — tests for GitHub issue #112, part 2.

Before this, only the dashboard's Settings pane could point Mode B or Mode C
at a keyless, self-hosted OpenAI-compatible server; the CLI only offered
named cloud presets with a mandatory key. These tests prove the CLI path now
works end to end: saves the right config, applies the same endpoint-trust
rules as the remote reranker, and runs a real connection test against an
ephemeral stub server — matching ``POST /api/v3/provider/test``'s behavior.
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode
from tests._portable import child_env_base
from tests.fixtures.stub_llm_server import StubLLMServer


def _cfg(tmp_path, mode=Mode.A):
    config = SLMConfig.for_mode(mode, base_dir=tmp_path)
    config.save(mode_change=True)
    return config


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_endpoint_is_refused(self, tmp_path) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        with pytest.raises(ValueError, match="--endpoint"):
            configure_custom_endpoint_provider(
                config, endpoint=None, api_key=None, model=None,
                target_mode=None, interactive=False,
            )

    def test_bare_hostname_over_plain_http_is_refused(self, tmp_path) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        with pytest.raises(ValueError, match="HTTPS"):
            configure_custom_endpoint_provider(
                config, endpoint="http://my-llm-server.lan:8080/v1",
                api_key=None, model=None, target_mode="b", interactive=False,
            )

    def test_public_ip_over_plain_http_is_refused(self, tmp_path) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        with pytest.raises(ValueError, match="HTTPS"):
            configure_custom_endpoint_provider(
                config, endpoint="http://8.8.8.8/v1",
                api_key=None, model=None, target_mode="c", interactive=False,
            )

    def test_private_lan_over_plain_http_refused_when_trust_flag_is_false(
        self, tmp_path,
    ) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        config.retrieval.trust_plain_http_lan = False
        with pytest.raises(ValueError, match="trust_plain_http_lan"):
            configure_custom_endpoint_provider(
                config, endpoint="http://192.168.1.50:8041/v1",
                api_key=None, model=None, target_mode="b", interactive=False,
            )

    def test_invalid_mode_is_refused(self, tmp_path) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        with pytest.raises(ValueError, match="--mode"):
            configure_custom_endpoint_provider(
                config, endpoint="http://127.0.0.1:9/v1",
                api_key=None, model=None, target_mode="z", interactive=False,
            )


# ---------------------------------------------------------------------------
# Saving — Mode B and Mode C, keyless and keyed
# ---------------------------------------------------------------------------

class TestSavesTheRightConfig:
    def test_keyless_endpoint_saves_mode_b(self, tmp_path, capsys) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path, mode=Mode.A)
        with StubLLMServer() as stub:
            configure_custom_endpoint_provider(
                config,
                endpoint=f"{stub.url}/v1",
                api_key=None,
                model="local-model",
                target_mode="b",
                interactive=False,
            )

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.mode is Mode.B
        assert reloaded.llm.provider == "openai"
        assert reloaded.llm.api_key == ""
        assert reloaded.llm.model == "local-model"

        out = capsys.readouterr().out
        assert "Mode: B" in out
        assert "none (keyless)" in out
        assert "Connection test:" in out

    def test_endpoint_with_a_key_targets_mode_c_by_default(
        self, tmp_path, capsys,
    ) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path, mode=Mode.A)
        with StubLLMServer() as stub:
            configure_custom_endpoint_provider(
                config,
                endpoint=f"{stub.url}/v1",
                api_key="sk-local-test",
                model="local-model",
                target_mode=None,
                interactive=False,
            )

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.mode is Mode.C
        assert reloaded.llm.api_key == "sk-local-test"

        out = capsys.readouterr().out
        assert "Key: configured" in out

    def test_preserves_unrelated_runtime_configuration(self, tmp_path) -> None:
        """Mirrors test_provider_noninteractive_contract's preservation
        guarantee for named presets — the custom-endpoint path must not
        regress it."""
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
        config.graph_backend = "cozo"
        config.scale_engine_state = "promoted"
        config.retrieval.semantic_top_k = 73
        config.save(mode_change=True)

        with StubLLMServer() as stub:
            configure_custom_endpoint_provider(
                config, endpoint=f"{stub.url}/v1", api_key=None,
                model="local-model", target_mode="b", interactive=False,
            )

        reloaded = SLMConfig.load(tmp_path / "config.json")
        assert reloaded.graph_backend == "cozo"
        assert reloaded.scale_engine_state == "promoted"
        assert reloaded.retrieval.semantic_top_k == 73


# ---------------------------------------------------------------------------
# Connection test — matches POST /api/v3/provider/test's acceptance rule
# ---------------------------------------------------------------------------

class TestConnectionTest:
    def test_reachable_server_reports_success(self) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            test_custom_endpoint_connection,
        )

        with StubLLMServer() as stub:
            ok, message = test_custom_endpoint_connection(f"{stub.url}/v1", "", "local-model")

        assert ok is True
        assert "reachable" in message
        assert "200" in message

    def test_unreachable_server_reports_failure(self) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            test_custom_endpoint_connection,
        )

        # Port 1 is reserved/unassigned — connection is refused immediately.
        ok, message = test_custom_endpoint_connection(
            "http://127.0.0.1:1/v1", "", "local-model",
        )
        assert ok is False
        assert "cannot connect" in message.lower()

    def test_failure_is_visible_after_save(self, tmp_path, capsys) -> None:
        from superlocalmemory.cli.provider_custom_endpoint import (
            configure_custom_endpoint_provider,
        )

        config = _cfg(tmp_path)
        configure_custom_endpoint_provider(
            config, endpoint="http://127.0.0.1:1/v1", api_key=None,
            model="local-model", target_mode="b", interactive=False,
        )
        out = capsys.readouterr().out
        assert "⚠ Connection test:" in out


# ---------------------------------------------------------------------------
# CLI wiring — argparse and cmd_provider
# ---------------------------------------------------------------------------

class TestCliWiring:
    def test_main_py_provider_subparser_accepts_custom_flags(self, tmp_path) -> None:
        """``main()`` builds its parser inline (no standalone factory to
        import), so this subprocess drives the real CLI entry point and
        intercepts the single ``parser.parse_args()`` call before dispatch —
        but ``main()`` runs first-run init (``_ensure_initialized()``)
        *before* that call, so HOME/SLM_DATA_DIR are isolated to ``tmp_path``
        to keep that side effect off the real ``~/.superlocalmemory``."""
        import json
        import os
        import subprocess
        import sys

        script = (
            "import sys, json; sys.argv = ["
            "'slm', 'provider', 'set', 'custom', "
            "'--endpoint', 'http://192.168.1.50:8041/v1', "
            "'--key', 'sk-test', '--model', 'llama3.2', '--mode', 'b'"
            "]\n"
            "import argparse\n"
            "_orig = argparse.ArgumentParser.parse_args\n"
            "def _capture(self, *a, **k):\n"
            "    ns = _orig(self, *a, **k)\n"
            "    print('CAPTURED:' + json.dumps(vars(ns)), file=sys.stderr)\n"
            "    raise SystemExit(0)\n"
            "argparse.ArgumentParser.parse_args = _capture\n"
            "from superlocalmemory.cli.main import main\n"
            "main()\n"
        )
        env = dict(os.environ)
        env.update(child_env_base(tmp_path))  # HOME, and USERPROFILE on Windows
        env["SLM_DATA_DIR"] = str(tmp_path / "data")
        env["SLM_NON_INTERACTIVE"] = "1"
        env["CI"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30, env=env,
        )
        line = next(
            (l for l in result.stderr.splitlines() if l.startswith("CAPTURED:")),
            None,
        )
        assert line is not None, f"parse_args was never reached: {result.stderr}"
        parsed = json.loads(line[len("CAPTURED:"):])
        assert parsed["provider"] == "custom"
        assert parsed["endpoint"] == "http://192.168.1.50:8041/v1"
        assert parsed["api_key"] == "sk-test"
        assert parsed["model"] == "llama3.2"
        assert parsed["mode"] == "b"

    def test_cmd_provider_forwards_custom_fields(self) -> None:
        from superlocalmemory.cli.commands import cmd_provider

        args = Namespace(
            action="set", provider="custom",
            endpoint="http://192.168.1.50:8041/v1", api_key="",
            model="local-model", mode="b",
        )
        with patch("superlocalmemory.cli.setup_wizard.configure_provider") as cp:
            cmd_provider(args)

        cp.assert_called_once()
        assert cp.call_args.kwargs == {
            "provider_name": "custom",
            "endpoint": "http://192.168.1.50:8041/v1",
            "api_key": "",
            "model": "local-model",
            "target_mode": "b",
        }
