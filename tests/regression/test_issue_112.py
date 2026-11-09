# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Regression tests for GitHub issue #112, part 2 (mode vocabulary + the

Mode C quick-switch guard).

Root cause 1: ``PUT /api/v3/mode`` (the dashboard's one-click Mode C button)
refused the switch whenever no cloud API key was on file — even when the
operator had already configured a keyless custom endpoint (llama.cpp, vLLM,
LM Studio, any other self-hosted OpenAI-compatible server). The full
settings save (``POST /api/v3/mode/set``) already accepted that setup; the
one-click button disagreed with it.

Root cause 2: every surface that describes a mode to a user defined Mode B
by naming Ollama specifically, and Mode C as "a cloud provider" — making one
vendor the definition of "local AI" and excluding self-hosted deployments
from Mode C's own description, even though Mode C's actual capability
(``core/config.py::_mode_template``, ``llm/backbone.py``) already accepted
any OpenAI-compatible endpoint.

These tests FAIL on the unfixed code (guard refuses a keyless custom
endpoint; wording asserts "Ollama"/"cloud provider" as the mode's
definition) and PASS after.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.config import LLMConfig
from superlocalmemory.server.routes.v3_api import _mode_c_quick_switch_refusal

# ---------------------------------------------------------------------------
# The Mode C quick-switch guard (PUT /api/v3/mode)
# ---------------------------------------------------------------------------

class TestModeCQuickSwitchGuard:
    def test_keyless_custom_endpoint_is_allowed(self) -> None:
        """A deliberately configured llama.cpp/vLLM endpoint needs no key."""
        llm = LLMConfig(provider="openai", api_base="http://192.168.1.50:8041/v1", api_key="")
        assert _mode_c_quick_switch_refusal(llm) is None

    def test_cloud_key_is_still_allowed(self) -> None:
        """Existing behaviour: a real cloud key always works."""
        llm = LLMConfig(provider="anthropic", api_key="sk-existing")
        assert _mode_c_quick_switch_refusal(llm) is None

    def test_nothing_configured_is_refused_with_a_next_step(self) -> None:
        """A genuinely bare config is still refused — but the message names
        BOTH ways to fix it, not just "get a cloud key"."""
        refusal = _mode_c_quick_switch_refusal(LLMConfig())
        assert refusal is not None
        assert refusal["code"] == "mode_c_requires_api_key"
        assert "key" in refusal["error"].lower()
        assert "endpoint" in refusal["error"].lower()

    def test_ollama_default_endpoint_does_not_count_as_a_custom_endpoint(self) -> None:
        """Mode B's leftover Ollama config must not silently satisfy Mode C.

        Ollama's endpoint is Mode B's own local model, not an endpoint the
        operator chose for Mode C — a one-click switch with only this on
        file must still be refused.
        """
        llm = LLMConfig(provider="ollama", api_base="http://localhost:11434", api_key="")
        refusal = _mode_c_quick_switch_refusal(llm)
        assert refusal is not None
        assert refusal["code"] == "mode_c_requires_api_key"


# ---------------------------------------------------------------------------
# LLMConfig.has_custom_endpoint
# ---------------------------------------------------------------------------

class TestLLMConfigHasCustomEndpoint:
    def test_true_for_non_ollama_endpoint(self) -> None:
        llm = LLMConfig(provider="openai", api_base="http://localhost:8080/v1")
        assert llm.has_custom_endpoint is True

    def test_false_for_ollama(self) -> None:
        llm = LLMConfig(provider="ollama", api_base="http://localhost:11434")
        assert llm.has_custom_endpoint is False

    def test_false_when_blank(self) -> None:
        assert LLMConfig().has_custom_endpoint is False

    def test_false_without_a_base_url(self) -> None:
        llm = LLMConfig(provider="openai", api_base="")
        assert llm.has_custom_endpoint is False


# ---------------------------------------------------------------------------
# Mode wording — vendor-neutral, consistent across surfaces
# ---------------------------------------------------------------------------

class TestModeWordingIsVendorNeutral:
    """Modes are named by what the user gets, never by a vendor (#112):

    A = no language model, B = a model on this machine (Ollama by default,
    any local server works), C = your own endpoint or a cloud provider.
    """

    def test_mode_b_tagline_names_ollama_only_as_a_default(self) -> None:
        from superlocalmemory.core.modes import mode_tagline
        from superlocalmemory.storage.models import Mode

        text = mode_tagline(Mode.B)
        assert "any local" in text.lower()
        assert "server" in text.lower()

    def test_mode_c_tagline_is_not_cloud_only(self) -> None:
        from superlocalmemory.core.modes import mode_tagline
        from superlocalmemory.storage.models import Mode

        text = mode_tagline(Mode.C)
        assert "own endpoint" in text.lower()

    def test_mcp_mode_description_matches_core_modes_tagline(self) -> None:
        """tools_v3._base_mode_description must not carry its own, divergent
        copy of the mode text — it is the one place the old bug lived."""
        from superlocalmemory.core.modes import mode_short_name, mode_tagline
        from superlocalmemory.mcp.tools_v3 import _base_mode_description
        from superlocalmemory.storage.models import Mode

        for mode in (Mode.A, Mode.B, Mode.C):
            expected = f"{mode_short_name(mode)} — {mode_tagline(mode)}"
            assert _base_mode_description(mode.value) == expected

    @pytest.mark.parametrize(
        "path",
        [
            "src/superlocalmemory/mcp/tools_v3.py",
            "src/superlocalmemory/storage/models.py",
            "src/superlocalmemory/cli/setup_wizard.py",
            "src/superlocalmemory/core/modes.py",
            "src/superlocalmemory/ui/index.html",
            "src/superlocalmemory/ui/js/auto-settings.js",
            "src/superlocalmemory/ui/js/od-settings.js",
        ],
    )
    def test_no_surface_defines_mode_b_as_requiring_ollama(self, path: str) -> None:
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        text = (repo_root / path).read_text(encoding="utf-8")
        assert "requires ollama" not in text.lower(), (
            f"{path} still defines Mode B by requiring Ollama specifically; "
            "any local OpenAI-compatible server must work."
        )

    def test_cli_help_epilog_does_not_define_mode_b_as_requiring_ollama(self) -> None:
        """Scoped to the mode-definition epilog specifically — cli/main.py also
        documents an unrelated ``context --full`` flag that legitimately
        needs Ollama today; that is a different feature, not Mode B's
        definition, and must not make this check a false positive."""
        from superlocalmemory.cli.main import _HELP_EPILOG

        assert "requires ollama" not in _HELP_EPILOG.lower()
        assert "any local" in _HELP_EPILOG.lower()

    def test_slm_help_modes_topic_is_vendor_neutral(self) -> None:
        """``slm help modes`` — scoped specifically, since cli/commands.py
        (2900+ lines) also legitimately names Ollama elsewhere (doctor
        diagnostics, the self-heal topic, an unrelated fast-path comment)
        and a whole-file scan would false-positive on those."""
        from superlocalmemory.cli.commands import _HELP_TOPICS

        text = _HELP_TOPICS["modes"].lower()
        assert "requires ollama" not in text
        assert "any local" in text
        assert "own endpoint" in text or "custom endpoint" in text
        # Names must match every other surface's product names, not an
        # independent, drifted vocabulary ("On-device only" / "Cloud AI").
        assert "local guardian" in text
        assert "smart local" in text
        assert "full power" in text

    def test_slm_help_command_list_does_not_name_mode_b_as_ollama_only(self) -> None:
        from superlocalmemory.cli.commands import _COMMAND_GROUPS

        mode_entries = [
            desc
            for _group_name, entries in _COMMAND_GROUPS
            for name, desc in entries
            if name == "mode"
        ]
        assert mode_entries, "no 'mode' entry found in _COMMAND_GROUPS"
        for desc in mode_entries:
            assert "(Ollama)" not in desc

    @pytest.mark.parametrize(
        "path",
        [
            "src/superlocalmemory/mcp/tools_v3.py",
            "src/superlocalmemory/storage/models.py",
            "src/superlocalmemory/cli/setup_wizard.py",
            "src/superlocalmemory/core/modes.py",
        ],
    )
    def test_mode_c_surface_names_a_custom_endpoint_option(self, path: str) -> None:
        """Every Python surface that spells out Mode C's capability must also
        name the self-hosted/custom-endpoint path, not just "cloud"."""
        import pathlib

        repo_root = pathlib.Path(__file__).resolve().parents[2]
        text = (repo_root / path).read_text(encoding="utf-8").lower()
        assert "own endpoint" in text or "custom endpoint" in text, (
            f"{path} describes Mode C without naming the custom-endpoint option."
        )

    def test_cli_help_epilog_names_a_custom_endpoint_option_for_mode_c(self) -> None:
        from superlocalmemory.cli.main import _HELP_EPILOG

        text = _HELP_EPILOG.lower()
        assert "own endpoint" in text or "custom endpoint" in text


class TestDashboardDescribesTheModeNotTheVendor:
    """The dashboard's System panel showed "Smart Local — ollama": the mode
    named after a vendor, on the one surface #112's rewording missed."""

    def test_dashboard_payload_carries_the_mode_tagline(self) -> None:
        import inspect

        from superlocalmemory.server.routes import v3_api

        source = inspect.getsource(v3_api)
        assert '"mode_tagline": mode_tagline(config.mode)' in source

    def test_dashboard_script_shows_the_tagline_not_the_provider(self) -> None:
        from pathlib import Path

        js = (Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
              / "ui" / "js" / "dashboard.js").read_text(encoding="utf-8")
        line = next(l for l in js.splitlines() if "'dashboard-mode-desc'" in l)
        assert "mode_tagline" in line and "provider" not in line
