# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for LLM provider layer — Task 1 of V3 build."""
import json
import pytest
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


def test_ollama_provider_no_key_needed(monkeypatch):
    """No API key is required -- a separate question from whether anything
    is actually listening at the configured host, which is faked reachable
    here so this test does not depend on the machine running it."""
    from superlocalmemory.llm import ollama_reachability
    monkeypatch.setattr(ollama_reachability, "ollama_reachable", lambda *a, **k: True)
    config = LLMConfig(provider="ollama", model="llama3.2")
    backbone = LLMBackbone(config)
    assert backbone.provider == "ollama"
    assert backbone.is_available()


def test_ollama_provider_unavailable_when_unreachable(monkeypatch):
    """4.1.22: a saved Mode B with nothing listening must say so, not report
    available forever (the old bug this reachability check closes)."""
    from superlocalmemory.llm import ollama_reachability
    monkeypatch.setattr(ollama_reachability, "ollama_reachable", lambda *a, **k: False)
    config = LLMConfig(provider="ollama", model="llama3.2")
    backbone = LLMBackbone(config)
    assert not backbone.is_available()


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
    data = json.loads(config_path.read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# Issue #128 — thinking models (qwen3.x, DeepSeek-R1 class)
# ---------------------------------------------------------------------------

def _thinking_model_backbone() -> LLMBackbone:
    config = LLMConfig(provider="ollama", model="qwen3.5:9b")
    return LLMBackbone(config)


def test_ollama_extract_text_prefers_content():
    backbone = _thinking_model_backbone()
    data = {"message": {"content": '[{"text": "hi"}]'}}
    assert backbone._extract_text(data) == '[{"text": "hi"}]'


def test_ollama_extract_text_falls_back_to_thinking(caplog):
    backbone = _thinking_model_backbone()
    trace = 'Reasoning... [{"text": "Alice works at Google"}]'
    data = {"message": {"role": "assistant", "content": "", "thinking": trace}}
    # DEBUG, not INFO/WARNING: for thinking models this is the normal
    # shape, and generate() sits on the store hot path (#128 review).
    with caplog.at_level("DEBUG", logger="superlocalmemory.llm.backbone"):
        assert backbone._extract_text(data) == trace
    assert "message.thinking" in caplog.text


def test_ollama_prose_content_falls_through_to_thinking():
    backbone = _thinking_model_backbone()
    trace = 'Reasoning... [{"text": "Alice works at Google"}]'
    content = "Here are the facts:"
    data = {
        "message": {
            "role": "assistant",
            "content": content,
            "thinking": trace,
        }
    }
    # 4.1.14 audit: both fields feed the parser — prose content no longer
    # discards the thinking trace, bracket noise no longer discards content.
    # Thinking goes first so content wins schema-fit ties (it is the
    # model's primary answer channel).
    assert backbone._extract_text(data) == trace + "\n" + content


def test_ollama_content_with_brackets_wins():
    backbone = _thinking_model_backbone()
    content = '[{"text": "hi"}]'
    thinking = "other trace"
    data = {
        "message": {
            "content": content,
            "thinking": thinking,
        }
    }
    # 4.1.14 audit: both fields feed the parser — prose content no longer
    # discards the thinking trace, bracket noise no longer discards content.
    # Thinking goes first so content wins schema-fit ties (it is the
    # model's primary answer channel).
    assert backbone._extract_text(data) == thinking + "\n" + content


def test_ollama_extract_text_hardening():
    backbone = _thinking_model_backbone()
    assert backbone._extract_text({"message": None}) == ""
    assert backbone._extract_text({"message": ["not", "a", "dict"]}) == ""
    assert backbone._extract_text({"message": "a string"}) == ""
    # Non-string fields never crash the hot path.
    data = {"message": {"content": 42, "thinking": 7}}
    assert backbone._extract_text(data) == ""


def test_ollama_extract_text_empty_both_returns_empty():
    backbone = _thinking_model_backbone()
    assert backbone._extract_text({"message": {"content": "", "thinking": ""}}) == ""
    assert backbone._extract_text({}) == ""


def test_ollama_payload_disables_think_by_default(monkeypatch):
    """4.2.9: every SLM prompt asks for a short structured answer, so thinking
    is off unless asked for. Measured on gemma4:e2b: 406-423 tokens and ~5 s
    per context call with the model default, 73-77 tokens and 0.85 s with
    think:false - and a capped call left on the default spends the whole cap
    thinking and returns empty content."""
    monkeypatch.delenv("SLM_OLLAMA_ALLOW_THINK", raising=False)
    backbone = _thinking_model_backbone()
    _, _, payload = backbone._build_ollama("hi", "", 100, 0.0)
    assert payload["think"] is False


def test_ollama_payload_keeps_model_thinking_when_allowed(monkeypatch):
    """An operator who wants the model's own thinking back opts in."""
    monkeypatch.setenv("SLM_OLLAMA_ALLOW_THINK", "1")
    backbone = _thinking_model_backbone()
    _, _, payload = backbone._build_ollama("hi", "", 100, 0.0)
    assert "think" not in payload


def test_ollama_payload_disables_think_when_env_set(monkeypatch):
    monkeypatch.setenv("SLM_OLLAMA_DISABLE_THINK", "1")
    monkeypatch.setenv("SLM_OLLAMA_ALLOW_THINK", "1")
    backbone = _thinking_model_backbone()
    _, _, payload = backbone._build_ollama("hi", "", 100, 0.0)
    # The older, stronger switch still wins over the opt-in.
    assert payload["think"] is False


def test_generate_threads_think_flag_to_ollama(monkeypatch):
    monkeypatch.delenv("SLM_OLLAMA_ALLOW_THINK", raising=False)
    backbone = _thinking_model_backbone()
    _, _, default_payload = backbone._build_request("hi", "", 100, 0.0)
    assert default_payload["think"] is False
    _, _, off_payload = backbone._build_request("hi", "", 100, 0.0, False)
    assert off_payload["think"] is False
    # True is never sent (non-thinking models reject it): the model decides.
    _, _, on_payload = backbone._build_request("hi", "", 100, 0.0, True)
    assert "think" not in on_payload


def test_generate_remembers_think_unsupported_after_400(monkeypatch):
    """A server that rejects the field once is not asked again, so an old
    Ollama costs one extra request per process, not one per memory."""
    import httpx

    monkeypatch.delenv("SLM_OLLAMA_ALLOW_THINK", raising=False)
    backbone = _thinking_model_backbone()
    monkeypatch.setattr(backbone, "is_available", lambda: True)
    seen: list[bool] = []

    def _old_server(url, headers, payload):
        seen.append("think" in payload)
        if "think" in payload:
            request = httpx.Request("POST", url)
            raise httpx.HTTPStatusError(
                "bad request", request=request,
                response=httpx.Response(400, request=request),
            )
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(backbone, "_send", _old_server)
    assert backbone.generate("one") == "ok"
    assert backbone.generate("two") == "ok"
    assert seen == [True, False, False]


def test_generate_downgrades_think_once_on_400(monkeypatch):
    import httpx

    backbone = _thinking_model_backbone()
    monkeypatch.setattr(backbone, "is_available", lambda: True)
    seen_payloads: list[dict] = []

    def _flaky_send(url, headers, payload):
        seen_payloads.append(dict(payload))
        if "think" in payload:
            request = httpx.Request("POST", url)
            raise httpx.HTTPStatusError(
                "bad request", request=request,
                response=httpx.Response(400, request=request),
            )
        return {"message": {"content": "downgraded answer"}}

    monkeypatch.setattr(backbone, "_send", _flaky_send)
    assert backbone.generate("hi", think=False) == "downgraded answer"
    assert [("think" in payload) for payload in seen_payloads] == [True, False]


def test_generate_second_400_returns_empty(monkeypatch):
    import httpx

    backbone = _thinking_model_backbone()
    monkeypatch.setattr(backbone, "is_available", lambda: True)

    def _always_400(url, headers, payload):
        request = httpx.Request("POST", url)
        raise httpx.HTTPStatusError(
            "bad request", request=request,
            response=httpx.Response(400, request=request),
        )

    monkeypatch.setattr(backbone, "_send", _always_400)
    assert backbone.generate("hi", think=False) == ""
