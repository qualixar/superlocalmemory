# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""#303: what a mode says about where data goes follows the configuration.

Modes A and B accept a remote embedding endpoint (and B a hosted language
model), so "data stays local" cannot be a constant of the mode name. The
per-mode record stays the default; every reader asks the effective answer,
worked out from the endpoints actually configured, by literal host and with
no DNS lookup.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.core.config import EmbeddingConfig, LLMConfig, SLMConfig
from superlocalmemory.storage.models import Mode

REMOTE_EMBEDDING = "https://api.openai.com/v1/embeddings"
LOCAL_EMBEDDING = "http://127.0.0.1:8045/v1/embeddings"

_UI = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui"


def _openai(endpoint: str) -> EmbeddingConfig:
    return EmbeddingConfig(provider="openai", api_endpoint=endpoint,
                           model_name="m", dimension=1024)


# ---------------------------------------------------------------------------
# One classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://localhost:8045/v1/embeddings",
    "http://LOCALHOST/v1",
    "http://embed.localhost:9000/v1",
    "http://127.0.0.1:8045/v1",
    "http://127.3.4.5/v1",
    "http://[::1]:8045/v1",
    "http://10.0.0.5:8000/v1",
    "http://172.16.0.1/v1",
    "http://172.31.255.254/v1",
    "http://192.168.1.20:11434/v1",
    "http://169.254.10.10/v1",
    "http://[fd12:3456::1]/v1",
    "http://[fc00::1]/v1",
    "http://[fe80::1]/v1",
    "http://[::ffff:192.168.1.5]/v1",
    "localhost:8045/v1/embeddings",
])
def test_loopback_and_private_lan_hosts_are_local(url):
    from superlocalmemory.core.data_locality import is_local_endpoint

    assert is_local_endpoint(url) is True


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1/embeddings",
    "http://8.8.8.8/v1",
    "http://172.15.0.1/v1",
    "http://172.32.0.1/v1",
    "http://100.64.0.1/v1",
    "http://[2606:4700::1111]/v1",
    "http://my-box.lan:8000/v1",   # a name is not proof: no DNS at config time
    "http://localhost.evil.com/v1",
    "http://notlocalhost/v1",
    "",
    "   ",
    "not a url at all",
    "http://",
])
def test_everything_else_is_remote(url):
    from superlocalmemory.core.data_locality import is_local_endpoint

    assert is_local_endpoint(url) is False


def test_the_classifier_never_resolves_a_name(monkeypatch):
    import socket

    from superlocalmemory.core.data_locality import is_local_endpoint

    def _boom(*_a, **_k):
        raise AssertionError("DNS lookup at config time")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(socket, "gethostbyname", _boom)
    assert is_local_endpoint("https://some.unresolvable.example/v1") is False


# ---------------------------------------------------------------------------
# Effective locality
# ---------------------------------------------------------------------------

def test_mode_a_default_stays_local():
    from superlocalmemory.core.data_locality import effective_data_locality

    config = SLMConfig.for_mode(Mode.A)
    loc = effective_data_locality(config.mode, config.embedding, config.llm)
    assert loc.stays_local is True
    assert loc.label == "local-only"
    assert loc.remote_parts == ()


def test_mode_a_with_a_local_endpoint_stays_local():
    from superlocalmemory.core.data_locality import effective_data_locality

    loc = effective_data_locality(Mode.A, _openai(LOCAL_EMBEDDING), LLMConfig())
    assert loc.stays_local is True


def test_mode_a_with_a_remote_endpoint_does_not():
    from superlocalmemory.core.data_locality import effective_data_locality

    loc = effective_data_locality(Mode.A, _openai(REMOTE_EMBEDDING), LLMConfig())
    assert loc.stays_local is False
    assert loc.label == "provider-assisted"
    assert loc.remote_parts == ("embedding endpoint",)


def test_mode_b_with_ollama_stays_local_and_with_a_hosted_model_does_not():
    from superlocalmemory.core.data_locality import effective_data_locality

    local = SLMConfig.for_mode(Mode.B)
    assert effective_data_locality(local.mode, local.embedding, local.llm).stays_local is True

    for provider in ("openrouter", "openai", "anthropic"):
        hosted = SLMConfig.for_mode(Mode.B, llm_provider=provider, llm_model="m")
        loc = effective_data_locality(hosted.mode, hosted.embedding, hosted.llm)
        assert loc.stays_local is False, provider
        assert loc.remote_parts == ("language model",), provider


def test_mode_b_with_a_local_openai_compatible_server_stays_local():
    from superlocalmemory.core.data_locality import effective_data_locality

    llm = LLMConfig(provider="openai", model="m", api_base="http://127.0.0.1:8000/v1")
    assert effective_data_locality(Mode.B, EmbeddingConfig(), llm).stays_local is True


def test_mode_b_reports_both_remote_parts():
    from superlocalmemory.core.data_locality import effective_data_locality

    llm = LLMConfig(provider="openrouter", model="m")
    loc = effective_data_locality(Mode.B, _openai(REMOTE_EMBEDDING), llm)
    assert loc.stays_local is False
    assert loc.remote_parts == ("embedding endpoint", "language model")


def test_mode_a_ignores_a_language_model_it_never_calls():
    from superlocalmemory.core.data_locality import effective_data_locality

    llm = LLMConfig(provider="openrouter", model="m")
    assert effective_data_locality(Mode.A, EmbeddingConfig(), llm).stays_local is True


def test_mode_c_default_is_not_made_local_by_a_local_endpoint():
    from superlocalmemory.core.data_locality import effective_data_locality

    loc = effective_data_locality(Mode.C, _openai(LOCAL_EMBEDDING), LLMConfig(provider="ollama"))
    assert loc.stays_local is False


def test_missing_configuration_falls_back_to_the_mode_default():
    from superlocalmemory.core.data_locality import effective_data_locality

    assert effective_data_locality(Mode.A, None, None).stays_local is True
    assert effective_data_locality(Mode.C, None, None).stays_local is False


def test_an_ollama_embedder_pointed_at_a_remote_host_is_remote():
    from superlocalmemory.core.data_locality import effective_data_locality

    emb = EmbeddingConfig(provider="ollama", ollama_base_url="http://gpu.example.com:11434")
    assert effective_data_locality(Mode.A, emb, LLMConfig()).stays_local is False
    assert effective_data_locality(Mode.A, EmbeddingConfig(provider="ollama"), LLMConfig()).stays_local is True


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def test_validate_mode_config_warns_about_a_remote_endpoint_under_a_and_b():
    from superlocalmemory.core.modes import validate_mode_config

    for mode in (Mode.A, Mode.B):
        issues = validate_mode_config(
            mode, has_llm=True, embedding=_openai(REMOTE_EMBEDDING), llm=LLMConfig(provider="ollama"),
        )
        assert any("embedding endpoint" in i and "leaves this computer" in i for i in issues), issues


def test_validate_mode_config_is_quiet_for_local_endpoints_and_for_mode_c():
    from superlocalmemory.core.modes import validate_mode_config

    assert validate_mode_config(
        Mode.A, embedding=_openai(LOCAL_EMBEDDING), llm=LLMConfig()) == []
    c_issues = validate_mode_config(
        Mode.C, has_cloud_llm=True, embedding=_openai(REMOTE_EMBEDDING), llm=LLMConfig(provider="openrouter"))
    assert not any("leaves this computer" in i for i in c_issues)


def test_validate_mode_config_without_a_config_is_unchanged():
    from superlocalmemory.core.modes import validate_mode_config

    assert validate_mode_config(Mode.A) == []


def test_eu_ai_act_gets_the_effective_value():
    from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

    checker = EUAIActChecker()
    remote = checker.check_compliance(Mode.A, embedding=_openai(REMOTE_EMBEDDING), llm=LLMConfig())
    assert remote.data_stays_local is False
    assert any("embedding endpoint" in f for f in remote.findings)

    local = checker.check_compliance(Mode.A, embedding=_openai(LOCAL_EMBEDDING), llm=LLMConfig())
    assert local.data_stays_local is True


def test_eu_ai_act_mode_b_remote_embedding_with_a_local_model_is_not_called_cloud_gen_ai():
    from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

    report = EUAIActChecker().check_compliance(
        Mode.B, embedding=_openai(REMOTE_EMBEDDING), llm=LLMConfig(provider="ollama"))
    assert report.data_stays_local is False
    assert not any("cloud generative AI" in f for f in report.findings)


def test_eu_ai_act_mode_b_hosted_model_is_cloud_gen_ai():
    from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

    report = EUAIActChecker().check_compliance(
        Mode.B, embedding=EmbeddingConfig(), llm=LLMConfig(provider="openrouter", model="m"))
    assert report.data_stays_local is False
    assert any("cloud generative AI" in f for f in report.findings)


def test_eu_ai_act_without_a_config_keeps_the_mode_defaults():
    from superlocalmemory.compliance.eu_ai_act import EUAIActChecker

    reports = EUAIActChecker().verify_all_modes()
    assert [reports[m].data_stays_local for m in ("a", "b", "c")] == [True, True, False]


def test_dashboard_fields_follow_the_configuration():
    from superlocalmemory.core.modes import dashboard_mode_fields

    default = dashboard_mode_fields(Mode.A)
    assert default["data_stays_local"] is True
    assert default["data_locality_label"] == "local-only"

    remote = dashboard_mode_fields(Mode.A, embedding=_openai(REMOTE_EMBEDDING), llm=LLMConfig())
    assert remote["data_stays_local"] is False
    assert remote["data_locality_label"] == "provider-assisted"
    assert remote["data_remote_parts"] == ["embedding endpoint"]


def _dashboard(config, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from superlocalmemory.server.routes.v3_api import router

    app = FastAPI()
    app.state.config = config
    app.include_router(router)
    return TestClient(app).get("/api/v3/dashboard")


def test_dashboard_route_reports_the_effective_state(tmp_path):
    base = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    base.embedding = _openai(REMOTE_EMBEDDING)
    resp = _dashboard(base, tmp_path)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["data_stays_local"] is False
    assert body["data_locality_label"] == "provider-assisted"
    assert "api.openai.com" not in resp.text  # status names the part, never the host

    base.embedding = _openai(LOCAL_EMBEDDING)
    assert _dashboard(base, tmp_path).json()["data_stays_local"] is True


def test_embedding_config_route_reports_the_effective_state(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from superlocalmemory.server.routes.v3_api import router

    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.embedding = _openai(REMOTE_EMBEDDING)
    app = FastAPI()
    app.state.config = config
    app.include_router(router)
    body = TestClient(app).get("/api/v3/embedding/config").json()
    assert body["data_stays_local"] is False
    assert body["endpoint_is_local"] is False


def test_mcp_mode_description_stops_saying_nothing_leaves_when_something_does():
    from superlocalmemory.core.data_locality import effective_data_locality
    from superlocalmemory.mcp.tools_v3 import _mode_description

    plain = _mode_description("a")
    assert "nothing leaves this device" in plain

    loc = effective_data_locality(Mode.A, _openai(REMOTE_EMBEDDING), LLMConfig())
    text = _mode_description("a", locality=loc)
    assert "nothing leaves this device" not in text
    assert "embedding endpoint" in text

    local = effective_data_locality(Mode.A, _openai(LOCAL_EMBEDDING), LLMConfig())
    assert _mode_description("a", locality=local) == plain


# ---------------------------------------------------------------------------
# Dashboard Step 3
# ---------------------------------------------------------------------------

def test_step_3_says_a_non_local_endpoint_sends_memory_text_away():
    html = (_UI / "index.html").read_text(encoding="utf-8")
    assert 'id="settings-emb-locality-note"' in html
    assert "memory text" in html and "leaves this computer" in html


def test_the_mode_a_card_follows_the_effective_state():
    html = (_UI / "index.html").read_text(encoding="utf-8")
    js = (_UI / "js" / "auto-settings.js").read_text(encoding="utf-8")
    assert 'id="mode-a-card-note"' in html
    assert "mode-a-card-note" in js and "data_stays_local" in js
