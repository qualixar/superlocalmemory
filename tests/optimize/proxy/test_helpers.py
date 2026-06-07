"""LLD-01 §5.3 — _helpers tests (header redaction, hop-by-hop, fail-open)."""

from __future__ import annotations

from fastapi.requests import Request

from superlocalmemory.optimize.proxy._helpers import (
    _HOP_BY_HOP,
    _SENSITIVE_HEADER_KEYS,
    _body_has_tools,
    _build_forward_headers,
    _filter_response_headers,
    _redact_headers,
)


def test_redact_headers_authorization() -> None:
    h = _redact_headers({
        "authorization": "Bearer sk-secret-12345",
        "x-api-key": "sk-ant-secret",
        "x-goog-api-key": "goog-secret",
        "content-type": "application/json",
    })
    assert h["authorization"] == "[REDACTED]"
    assert h["x-api-key"] == "[REDACTED]"
    assert h["x-goog-api-key"] == "[REDACTED]"
    assert h["content-type"] == "application/json"


def test_redact_headers_does_not_mutate_input() -> None:
    src = {"authorization": "Bearer real-secret"}
    out = _redact_headers(src)
    assert out is not src
    assert src["authorization"] == "Bearer real-secret"  # original untouched


def test_body_has_tools_true() -> None:
    assert _body_has_tools({"tools": [{"name": "bash"}]}) is True


def test_body_has_tools_false_empty() -> None:
    assert _body_has_tools({"tools": []}) is False


def test_body_has_tools_false_missing() -> None:
    assert _body_has_tools({"messages": []}) is False


def test_filter_response_headers_strips_hop_by_hop() -> None:
    h = _filter_response_headers({
        "content-type": "application/json",
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
    })
    assert "content-type" in h
    assert "connection" not in h
    assert "transfer-encoding" not in h


def test_sensitive_header_keys_is_frozenset() -> None:
    assert isinstance(_SENSITIVE_HEADER_KEYS, frozenset)
    assert "authorization" in _SENSITIVE_HEADER_KEYS
    assert "x-api-key" in _SENSITIVE_HEADER_KEYS
    assert "x-goog-api-key" in _SENSITIVE_HEADER_KEYS


def test_hop_by_hop_strips_x_forwarded() -> None:
    """SEC-M-01: topology-revealing headers must not be forwarded."""
    assert "x-forwarded-for" in _HOP_BY_HOP
    assert "x-forwarded-host" in _HOP_BY_HOP
    assert "x-forwarded-proto" in _HOP_BY_HOP
    assert "x-real-ip" in _HOP_BY_HOP
