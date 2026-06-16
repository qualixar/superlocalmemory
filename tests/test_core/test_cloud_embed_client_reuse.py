# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Regression test: _cloud_embed_batch must reuse self._get_http_client().

WP-H: _cloud_embed_batch was creating a fresh httpx.Client inside the retry
loop (new TCP+TLS per call/retry) and calling resp.json() outside the
with-block (fragile body-buffering).  After the fix it must:

1. Call _get_http_client() to obtain the shared client (not construct a new one).
2. Use the SAME client instance across two separate embed_batch calls.
3. Call resp.json() while the response is still in scope (inside the try block).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cloud_config(*, dimension: int = 768) -> EmbeddingConfig:
    """Minimal EmbeddingConfig for a cloud (Azure) provider."""
    return EmbeddingConfig(
        model_name="text-embedding-ada-002",
        dimension=dimension,
        api_endpoint="https://fake-azure.openai.azure.com",
        api_key="test-api-key-1234",
        deployment_name="ada-002",
        api_version="2024-02-01",
        provider="azure",
    )


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    """Build a minimal fake httpx.Response whose .json() returns Azure format."""
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": data, "model": "ada-002"})
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCloudEmbedClientReuse:
    """_cloud_embed_batch must reuse the shared httpx client."""

    def test_get_http_client_called_not_fresh_httpx_client(self) -> None:
        """_cloud_embed_batch must obtain its client from _get_http_client(),
        NOT from constructing a new httpx.Client inside the method."""
        cfg = _make_cloud_config()
        svc = EmbeddingService(cfg)

        stub_client = MagicMock()
        vectors = [[0.1] * 768]
        stub_client.post.return_value = _fake_response(vectors)

        get_http_client_call_count = 0

        def tracking_get_http_client():
            nonlocal get_http_client_call_count
            get_http_client_call_count += 1
            return stub_client

        svc._get_http_client = tracking_get_http_client  # type: ignore[method-assign]

        # Patch httpx.Client at the real module level so we can detect direct use.
        # The old code does `import httpx` then `with httpx.Client(...)`.
        # We make httpx.Client raise if called, so the old code will error out
        # while the fixed code (using _get_http_client) will pass.
        with patch("httpx.Client", side_effect=AssertionError(
            "_cloud_embed_batch must NOT create a new httpx.Client; "
            "use self._get_http_client() instead"
        )):
            result = svc._cloud_embed_batch(["hello world"])

        assert result == vectors, f"Expected {vectors}, got {result}"
        assert get_http_client_call_count >= 1, (
            "_get_http_client() was never called — client not reused"
        )

    def test_same_client_instance_across_two_calls(self) -> None:
        """Two consecutive embed_batch calls must use the SAME client object
        (handshake-once behaviour — no repeated TCP+TLS per call)."""
        cfg = _make_cloud_config()
        svc = EmbeddingService(cfg)

        vectors_a = [[0.1] * 768]
        vectors_b = [[0.2] * 768]

        stub_client = MagicMock()
        stub_client.post.side_effect = [
            _fake_response(vectors_a),
            _fake_response(vectors_b),
        ]

        clients_seen: list[object] = []

        def tracking_get_http_client():
            clients_seen.append(stub_client)
            return stub_client

        svc._get_http_client = tracking_get_http_client  # type: ignore[method-assign]

        with patch("httpx.Client", side_effect=AssertionError(
            "Direct httpx.Client() construction detected — must use _get_http_client()"
        )):
            result_a = svc._cloud_embed_batch(["first text"])
            result_b = svc._cloud_embed_batch(["second text"])

        assert result_a == vectors_a
        assert result_b == vectors_b

        # Both calls must have gone through _get_http_client
        assert len(clients_seen) >= 2, (
            f"Expected _get_http_client() called >= 2 times, got {len(clients_seen)}"
        )
        # All returned instances must be identical
        assert all(c is stub_client for c in clients_seen), (
            "Different client instances used across calls — TCP connection NOT reused"
        )

    def test_resp_json_called_inside_try_block(self) -> None:
        """resp.json() must be called while the response is still valid.

        The old code called resp.json() AFTER exiting the with-block.  We
        verify correctness by ensuring the parsed data is returned intact and
        json() is called exactly once per batch call.
        """
        cfg = _make_cloud_config()
        svc = EmbeddingService(cfg)

        expected = [[float(i) / 768 for i in range(768)]]

        stub_client = MagicMock()
        resp = _fake_response(expected)
        stub_client.post.return_value = resp

        svc._get_http_client = lambda: stub_client  # type: ignore[method-assign]

        with patch("httpx.Client", side_effect=AssertionError(
            "Must not construct new httpx.Client"
        )):
            result = svc._cloud_embed_batch(["test sentence"])

        assert result == expected, "Embedding data not parsed / returned correctly"
        # resp.json() must have been called exactly once
        resp.json.assert_called_once()

    def test_retry_reuses_same_client(self) -> None:
        """On transient failure the retry attempt must use the same client,
        not spin up a new TCP connection."""
        import httpx as real_httpx

        cfg = _make_cloud_config()
        svc = EmbeddingService(cfg)

        vectors = [[0.5] * 768]

        stub_client = MagicMock()
        # First call raises a network error; second succeeds
        stub_client.post.side_effect = [
            real_httpx.ConnectError("transient"),
            _fake_response(vectors),
        ]

        get_http_client_call_count = 0

        def tracking_get_http_client():
            nonlocal get_http_client_call_count
            get_http_client_call_count += 1
            return stub_client

        svc._get_http_client = tracking_get_http_client  # type: ignore[method-assign]

        with patch("httpx.Client", side_effect=AssertionError(
            "Must not create new httpx.Client on retry"
        )):
            # Make sleep a no-op so test is fast
            with patch("superlocalmemory.core.embeddings.time.sleep"):
                result = svc._cloud_embed_batch(["retry text"])

        assert result == vectors
        assert get_http_client_call_count >= 1, (
            "_get_http_client() never called during retry path"
        )
