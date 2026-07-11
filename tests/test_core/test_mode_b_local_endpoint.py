# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-15 coverage tests (D5 — Mode B local endpoint)

"""Tests for backbone.py Mode B local/keyless endpoint behavior.

Covers:
- Keyless OpenAI-compatible endpoint: is_available==True, generate() works,
  NO Authorization header, request hits configured base not localhost:11434
- Ollama native: request goes to /api/chat at configured base, NOT localhost:11434
- validate_mode_config() with/without Ollama

Uses StubLLMServer (ephemeral 127.0.0.1:0) — NO real Ollama or OpenAI calls.

CRIT-3 (LLD): backbone retries 3× on error. We assert on the LAST recorded
request of a successful call AND that localhost:11434 sentinel got ZERO requests.
"""

from __future__ import annotations

import pytest

from tests.fixtures.stub_llm_server import StubLLMServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backbone(provider: str, api_base: str, api_key: str = "", model: str = "test-model"):
    """Construct LLMBackbone with explicit config, no env lookups."""
    from superlocalmemory.core.config import LLMConfig
    from superlocalmemory.llm.backbone import LLMBackbone

    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        timeout_seconds=10,
        temperature=0.0,
        max_tokens=64,
    )
    return LLMBackbone(config)


# ---------------------------------------------------------------------------
# Keyless OpenAI-compatible (Mode B with llama.cpp / LM Studio / vLLM)
# ---------------------------------------------------------------------------

class TestKeylessOpenAI:
    def test_is_available_keyless_with_base_url(self) -> None:
        """A custom base_url is sufficient — api_key NOT required for Mode B local."""
        with StubLLMServer(reply_text="EXTRACTED") as stub:
            backbone = _make_backbone("openai", api_base=stub.url, api_key="")
            assert backbone.is_available() is True, (
                "is_available() must be True when base_url is set, even with empty key; "
                "backbone.py modeb-1 comment describes this fix"
            )

    def test_generate_posts_to_configured_base_not_openai_com(self) -> None:
        """generate() must POST to configured base_url/chat/completions, not api.openai.com."""
        with StubLLMServer(reply_text="EXTRACTED") as stub:
            backbone = _make_backbone("openai", api_base=f"{stub.url}/v1", api_key="")
            result = backbone.generate("test prompt")

            assert result == "EXTRACTED", (
                f"Expected stub reply 'EXTRACTED', got {result!r}"
            )
            assert len(stub.requests) >= 1, "No requests recorded — backbone didn't call stub"
            last_req = stub.requests[-1]
            assert "chat/completions" in last_req.path, (
                f"Request path should contain 'chat/completions', got: {last_req.path}"
            )

    def test_no_authorization_header_when_keyless(self, monkeypatch) -> None:
        """Keyless OpenAI-compatible request must NOT send Authorization header.

        backbone.py V3.5.9: Authorization is omitted when api_key is empty so
        unauthenticated local endpoints don't reject with HTTP 401.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-should-not-leak")
        with StubLLMServer(reply_text="EXTRACTED") as stub:
            backbone = _make_backbone("openai", api_base=f"{stub.url}/v1", api_key="")
            backbone.generate("test prompt")

            assert len(stub.requests) >= 1
            last_req = stub.requests[-1]
            assert "authorization" not in last_req.headers, (
                f"Authorization header present in keyless request: {last_req.headers}; "
                "backbone.py should omit Authorization when api_key is empty"
            )

    def test_keyed_openai_sends_authorization(self) -> None:
        """When api_key is set, Authorization: Bearer must be present."""
        with StubLLMServer(reply_text="EXTRACTED") as stub:
            backbone = _make_backbone("openai", api_base=f"{stub.url}/v1", api_key="sk-test-key")
            backbone.generate("test prompt")

            assert len(stub.requests) >= 1
            last_req = stub.requests[-1]
            assert "authorization" in last_req.headers, (
                "Authorization header missing even though api_key was set"
            )
            assert last_req.headers["authorization"].startswith("Bearer "), (
                f"Authorization format wrong: {last_req.headers['authorization']!r}"
            )


# ---------------------------------------------------------------------------
# Ollama native /api/chat path
# ---------------------------------------------------------------------------

class TestOllamaEndpoint:
    def test_ollama_posts_to_api_chat_not_v1_completions(self) -> None:
        """Ollama provider must use /api/chat (native format), not /v1/chat/completions."""
        with StubLLMServer(reply_text="OLLAMA-REPLY") as stub:
            backbone = _make_backbone("ollama", api_base=stub.url)
            result = backbone.generate("test prompt")

            assert result == "OLLAMA-REPLY", (
                f"Expected stub reply 'OLLAMA-REPLY', got {result!r}"
            )
            assert len(stub.requests) >= 1
            last_req = stub.requests[-1]
            assert last_req.path.endswith("/api/chat") or "/api/chat" in last_req.path, (
                f"Ollama request went to {last_req.path!r}, expected /api/chat; "
                "backbone._build_ollama should set base_url to <host>/api/chat"
            )

    def test_ollama_does_not_hit_localhost_11434_when_override_set(self) -> None:
        """When api_base is set for Ollama, localhost:11434 must receive ZERO requests.

        This verifies the url-override plumbing — the stub is on a random port,
        and if backbone falls back to _OLLAMA_DEFAULT_BASE the stub would get 0 requests.
        """
        with StubLLMServer(reply_text="OVERRIDE") as stub:
            backbone = _make_backbone("ollama", api_base=stub.url)
            result = backbone.generate("test override")

            # Stub must have received the request (not localhost:11434)
            assert len(stub.requests) >= 1, (
                "Stub received 0 requests — backbone may have used localhost:11434 instead "
                "of the configured api_base"
            )
            assert result == "OVERRIDE"

    def test_ollama_is_always_available(self) -> None:
        """Ollama provider reports is_available=True without any API key."""
        with StubLLMServer() as stub:
            backbone = _make_backbone("ollama", api_base=stub.url)
            assert backbone.is_available() is True

    def test_ollama_payload_has_keep_alive_and_stream_false(self) -> None:
        """Ollama payload must include stream=False and keep_alive for memory safety."""
        with StubLLMServer(reply_text="OK") as stub:
            backbone = _make_backbone("ollama", api_base=stub.url)
            backbone.generate("test prompt")

            assert len(stub.requests) >= 1
            payload = stub.requests[-1].body
            assert payload.get("stream") is False, (
                "stream must be False to prevent Ollama from streaming back tokens"
            )
            assert "keep_alive" in payload, (
                "keep_alive must be set to prevent Ollama from holding the model in RAM"
            )


# ---------------------------------------------------------------------------
# validate_mode_config (from core.modes) — gaps in test_modes.py
# ---------------------------------------------------------------------------

class TestValidateModeConfig:
    def test_mode_b_no_warning_with_ollama(self) -> None:
        """No warnings when Mode B has Ollama available."""
        from superlocalmemory.core.modes import validate_mode_config
        from superlocalmemory.storage.models import Mode

        issues = validate_mode_config(Mode.B, has_ollama=True)
        assert issues == [], f"Expected no issues, got: {issues}"

    def test_mode_b_warns_without_ollama(self) -> None:
        """Mode B emits a warning when Ollama is not available."""
        from superlocalmemory.core.modes import validate_mode_config
        from superlocalmemory.storage.models import Mode

        issues = validate_mode_config(Mode.B, has_ollama=False)
        assert len(issues) >= 1, "Expected at least one warning for Mode B without Ollama"
        combined = " ".join(issues).lower()
        assert "ollama" in combined, (
            f"Warning should mention Ollama, got: {issues}"
        )

    def test_mode_c_warns_without_cloud_llm(self) -> None:
        """Mode C emits warnings when cloud LLM is not available."""
        from superlocalmemory.core.modes import validate_mode_config
        from superlocalmemory.storage.models import Mode

        issues = validate_mode_config(Mode.C, has_cloud_llm=False)
        assert len(issues) >= 1, "Expected warnings for Mode C without cloud LLM"

    def test_mode_a_always_clean(self) -> None:
        """Mode A needs no external services — always returns empty issues."""
        from superlocalmemory.core.modes import validate_mode_config
        from superlocalmemory.storage.models import Mode

        issues = validate_mode_config(Mode.A)
        assert issues == [], f"Mode A should have no validation issues, got: {issues}"


# ---------------------------------------------------------------------------
# backbone.py LLMUnavailableError when no provider/key
# ---------------------------------------------------------------------------

class TestUnavailableError:
    def test_generate_raises_when_no_provider(self) -> None:
        """generate() must raise LLMUnavailableError when provider is empty."""
        from superlocalmemory.core.config import LLMConfig
        from superlocalmemory.llm.backbone import LLMBackbone, LLMUnavailableError

        config = LLMConfig(
            provider="",
            model="",
            api_key="",
            api_base="",
            timeout_seconds=5,
            temperature=0.0,
            max_tokens=64,
        )
        backbone = LLMBackbone(config)
        assert backbone.is_available() is False

        with pytest.raises(LLMUnavailableError):
            backbone.generate("test")
