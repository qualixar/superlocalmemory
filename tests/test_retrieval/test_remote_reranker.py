# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Issue #105 — remote / OpenAI-compatible reranker endpoints.

The HTTP layer is stubbed with ``httpx.MockTransport``, NOT by mocking the
code under test. Every test drives the real ``RemoteReranker`` methods, the
real ``httpx.Client``, the real bounded body read, the real retry policy, and
the real response-schema gate — only the socket is fake. (PR #101's tests
mocked the function under test and therefore proved nothing; that is the
failure mode these tests are written against.)

No test performs real network I/O.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import patch

import httpx
import pytest

from superlocalmemory.retrieval.remote_reranker import (
    CROSS_ENCODER_API_KEY_ENV,
    RemoteReranker,
    RemoteRerankerConfigError,
    RemoteRerankerError,
    is_remote_cross_encoder_backend,
    normalize_rerank_endpoint,
    parse_rerank_response,
    redact_endpoint,
    validate_remote_reranker_config,
)
from superlocalmemory.storage.models import AtomicFact

ENDPOINT = "https://reranker.example.test/v1/rerank"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fact(fact_id: str, content: str) -> AtomicFact:
    return AtomicFact(fact_id=fact_id, memory_id="m0", content=content)


def _candidates() -> list[tuple[AtomicFact, float]]:
    """Three candidates in DESCENDING fusion order (f0 best)."""
    return [
        (_fact("f0", "文档零"), 0.9),
        (_fact("f1", "文档一"), 0.5),
        (_fact("f2", "文档二"), 0.1),
    ]


class _Recorder:
    """Captures every request the reranker actually put on the wire."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


@contextmanager
def stub_http(
    handler: Callable[[httpx.Request], httpx.Response],
    reranker: "RemoteReranker | None" = None,
):
    """Replace httpx's transport, keeping the real Client/stream/read code.

    ``reranker`` drops any client pooled by an earlier stub so a second
    context really does install a new transport (production holds exactly one
    transport for the process lifetime; tests do not).
    """
    if reranker is not None:
        reranker.unload()
    recorder = _Recorder()
    real_client = httpx.Client

    def _handler(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        return handler(request)

    def _factory(**kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    with patch("httpx.Client", _factory):
        yield recorder


def _ok_response(scores_by_index: dict[int, float]) -> Callable:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "results": [
                {"index": i, "relevance_score": s}
                for i, s in scores_by_index.items()
            ],
        })
    return handler


def _raises(exc: Exception) -> Callable:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc
    return handler


def _reranker(**kwargs: Any) -> RemoteReranker:
    kwargs.setdefault("model_name", "/root/model/reranker.gguf")
    kwargs.setdefault("endpoint", ENDPOINT)
    return RemoteReranker(kwargs.pop("model_name"), kwargs.pop("endpoint"), **kwargs)


# ---------------------------------------------------------------------------
# Backend token recognition
# ---------------------------------------------------------------------------

class TestBackendToken:

    @pytest.mark.parametrize("backend", ["openai", "remote", "OpenAI", " remote "])
    def test_remote_tokens_recognised(self, backend: str) -> None:
        assert is_remote_cross_encoder_backend(backend) is True

    @pytest.mark.parametrize("backend", ["", "onnx", "pytorch", "torch", None])
    def test_local_tokens_not_remote(self, backend: str) -> None:
        assert is_remote_cross_encoder_backend(backend) is False


# ---------------------------------------------------------------------------
# Config validation — the issue #103 leftover, made loud
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_valid_remote_pair_has_no_error(self) -> None:
        assert validate_remote_reranker_config("openai", ENDPOINT) is None

    def test_plain_local_config_has_no_error(self) -> None:
        assert validate_remote_reranker_config("", "") is None
        assert validate_remote_reranker_config("onnx", "") is None

    def test_remote_backend_without_endpoint_is_named(self) -> None:
        error = validate_remote_reranker_config("openai", "")
        assert error
        assert "cross_encoder_endpoint" in error
        assert "openai" in error

    def test_endpoint_against_local_backend_is_no_longer_silent(self) -> None:
        """The exact #103 leftover: accepted, ignored, never mentioned.

        Before 3.8.12 ``cross_encoder_endpoint`` was not a dataclass field, so
        ``SLMConfig.load`` dropped it without a word and the user believed
        reranking was remote while an English model scored their corpus.
        """
        error = validate_remote_reranker_config("", ENDPOINT)
        assert error
        assert "cross_encoder_endpoint" in error
        assert "cross_encoder_backend" in error
        assert "ignored" in error

    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://host/rerank",
        "gopher://host:70/",
    ])
    def test_non_http_schemes_are_rejected(self, url: str) -> None:
        error = validate_remote_reranker_config("openai", url)
        assert error
        assert "http" in error

    def test_url_without_host_is_rejected(self) -> None:
        error = validate_remote_reranker_config("openai", "http:///v1/rerank")
        assert error
        assert "host" in error

    def test_https_is_allowed(self) -> None:
        assert validate_remote_reranker_config(
            "openai", "https://rerank.internal/v1/rerank",
        ) is None

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8041/v1/rerank",
        "http://[::1]:8041/v1/rerank",
        "http://localhost:8041/v1/rerank",
    ])
    def test_plain_http_is_loopback_only(self, url: str) -> None:
        assert validate_remote_reranker_config("openai", url) is None

    def test_plain_http_off_host_is_rejected(self) -> None:
        error = validate_remote_reranker_config(
            "openai", "http://192.168.50.140:8041/v1/rerank",
        )
        assert error and "HTTPS" in error

    def test_query_strings_and_fragments_are_rejected(self) -> None:
        for suffix in ("?api_key=secret", "#secret"):
            error = validate_remote_reranker_config(
                "openai", ENDPOINT + suffix,
            )
            assert error and "query string or fragment" in error

    def test_constructor_rejects_invalid_config(self) -> None:
        with pytest.raises(RemoteRerankerConfigError):
            RemoteReranker("m", "file:///etc/passwd")
        with pytest.raises(RemoteRerankerConfigError):
            RemoteReranker("m", "")


class TestEndpointNormalization:

    @pytest.mark.parametrize("given,expected", [
        ("https://h:8041/v1/rerank", "https://h:8041/v1/rerank"),
        ("https://h:8041/v1/rerank/", "https://h:8041/v1/rerank"),
        ("https://h:8041/v1", "https://h:8041/v1/rerank"),
        ("https://h:8041", "https://h:8041/rerank"),
    ])
    def test_rerank_suffix_is_added_once(self, given: str, expected: str) -> None:
        assert normalize_rerank_endpoint(given) == expected

    def test_credentials_are_stripped_before_logging(self) -> None:
        safe = redact_endpoint("http://admin:hunter2@h:8041/v1/rerank")
        assert "hunter2" not in safe
        assert "admin" not in safe
        assert "h:8041/v1/rerank" in safe


# ---------------------------------------------------------------------------
# Response schema gate
# ---------------------------------------------------------------------------

class TestResponseParsing:

    def test_cohere_shape_is_returned_in_document_order(self) -> None:
        payload = {"results": [
            {"index": 1, "relevance_score": -5.94},
            {"index": 2, "relevance_score": -6.08},
            {"index": 0, "relevance_score": -6.34},
        ]}
        assert parse_rerank_response(payload, 3) == [-6.34, -5.94, -6.08]

    def test_bare_array_with_score_key_is_accepted(self) -> None:
        payload = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.2}]
        assert parse_rerank_response(payload, 2) == [0.9, 0.2]

    def test_score_count_mismatch_is_rejected(self) -> None:
        payload = {"results": [{"index": 0, "relevance_score": 1.0}]}
        with pytest.raises(RemoteRerankerError, match="refusing to guess"):
            parse_rerank_response(payload, 3)

    def test_missing_results_key_names_the_likely_misconfiguration(self) -> None:
        with pytest.raises(RemoteRerankerError, match="results"):
            parse_rerank_response({"data": [{"embedding": [0.1]}]}, 1)

    def test_non_object_payload_is_rejected(self) -> None:
        with pytest.raises(RemoteRerankerError):
            parse_rerank_response("not json object", 1)

    def test_duplicate_index_is_rejected(self) -> None:
        payload = {"results": [
            {"index": 0, "relevance_score": 1.0},
            {"index": 0, "relevance_score": 2.0},
        ]}
        with pytest.raises(RemoteRerankerError, match="duplicate"):
            parse_rerank_response(payload, 2)

    def test_out_of_range_index_is_rejected(self) -> None:
        payload = {"results": [
            {"index": 0, "relevance_score": 1.0},
            {"index": 7, "relevance_score": 2.0},
        ]}
        with pytest.raises(RemoteRerankerError, match="out-of-range"):
            parse_rerank_response(payload, 2)

    def test_non_integer_index_is_rejected(self) -> None:
        payload = {"results": [{"index": "0", "relevance_score": 1.0}]}
        with pytest.raises(RemoteRerankerError, match="non-integer"):
            parse_rerank_response(payload, 1)

    def test_non_numeric_score_is_rejected(self) -> None:
        payload = {"results": [{"index": 0, "relevance_score": "high"}]}
        with pytest.raises(RemoteRerankerError, match="non-numeric"):
            parse_rerank_response(payload, 1)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_score_is_rejected(self, bad: float) -> None:
        payload = {"results": [{"index": 0, "relevance_score": bad}]}
        with pytest.raises(RemoteRerankerError, match="non-finite"):
            parse_rerank_response(payload, 1)

    def test_missing_score_key_is_rejected(self) -> None:
        payload = {"results": [{"index": 0, "rank": 1}]}
        with pytest.raises(RemoteRerankerError, match="relevance_score"):
            parse_rerank_response(payload, 1)

    def test_result_item_that_is_not_an_object_is_rejected(self) -> None:
        with pytest.raises(RemoteRerankerError, match="expected an object"):
            parse_rerank_response({"results": [0.9]}, 1)


# ---------------------------------------------------------------------------
# Happy path over a real httpx client with a fake transport
# ---------------------------------------------------------------------------

class TestRerankHappyPath:

    def test_remote_scores_reorder_the_candidates(self) -> None:
        # Fusion order is f0 > f1 > f2; the remote reranker disagrees.
        handler = _ok_response({0: -6.34, 1: -5.94, 2: -6.08})
        rr = _reranker()
        with stub_http(handler):
            results, applied, status = rr.rerank_with_status(
                "测试重排速度", _candidates(), top_k=3,
            )
        assert applied is True
        assert status == "applied"
        assert [f.fact_id for f, _ in results] == ["f1", "f2", "f0"]
        assert results[0][1] == pytest.approx(-5.94)

    def test_request_body_matches_the_rerank_contract(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0})) as rec:
            rr.rerank("测试", _candidates(), top_k=2)
        assert len(rec.requests) == 1
        body = rec.bodies[0]
        assert body["model"] == "/root/model/reranker.gguf"
        assert body["query"] == "测试"
        assert body["documents"] == ["文档零", "文档一", "文档二"]
        assert str(rec.requests[0].url) == ENDPOINT

    def test_request_body_is_redacted_before_network_egress(self) -> None:
        rr = _reranker()
        candidates = [
            (_fact("f0", "Contact alice@example.com with sk-" + "a" * 24), 1.0),
        ]
        with stub_http(_ok_response({0: 1.0})) as rec:
            rr.rerank("SSN 123-45-6789", candidates, top_k=1)
        serialized = json.dumps(rec.bodies[0])
        assert "alice@example.com" not in serialized
        assert "123-45-6789" not in serialized
        assert "sk-" + "a" * 24 not in serialized
        assert "PII" in serialized

    def test_top_k_truncates_after_reranking(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 0.1, 1: 0.9, 2: 0.5})):
            results, applied, _ = rr.rerank_with_status(
                "q", _candidates(), top_k=2,
            )
        assert applied is True
        assert [f.fact_id for f, _ in results] == ["f1", "f2"]

    def test_empty_candidates_short_circuit_without_a_request(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({})) as rec:
            results, applied, status = rr.rerank_with_status("q", [], top_k=5)
        assert results == []
        assert applied is False
        assert status == "no_candidates"
        assert rec.requests == []

    def test_score_pair_returns_the_remote_score(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 4.25})):
            assert rr.score_pair("q", "doc") == pytest.approx(4.25)

    def test_warmup_probe_reports_ready(self, caplog) -> None:
        rr = _reranker()
        with caplog.at_level(logging.INFO), stub_http(_ok_response({0: 0.5})):
            assert rr.warmup_sync() is True
        assert "Remote reranker ready" in caplog.text


# ---------------------------------------------------------------------------
# Failure paths — visible degradation, never silent, never local substitution
# ---------------------------------------------------------------------------

class TestRerankFailurePaths:

    def _assert_degraded(self, rr: RemoteReranker, handler, caplog) -> None:
        with caplog.at_level(logging.ERROR), stub_http(handler) as rec:
            results, applied, status = rr.rerank_with_status(
                "q", _candidates(), top_k=3,
            )
        assert applied is False, "a failed rerank must not claim it applied"
        assert status == "remote_unavailable"
        # Degrades to fusion order — NOT to a locally-scored English model.
        assert [f.fact_id for f, _ in results] == ["f0", "f1", "f2"]
        assert "Remote reranker unavailable" in caplog.text
        assert rec.requests, "the reranker must actually have tried"

    def test_endpoint_down_degrades_visibly(self, caplog) -> None:
        self._assert_degraded(
            _reranker(), _raises(httpx.ConnectError("connection refused")),
            caplog,
        )

    def test_timeout_degrades_visibly(self, caplog) -> None:
        self._assert_degraded(
            _reranker(), _raises(httpx.ReadTimeout("timed out")), caplog,
        )

    def test_malformed_response_degrades_instead_of_scoring_garbage(
        self, caplog,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": "very relevant"},
            ]})
        self._assert_degraded(_reranker(), handler, caplog)

    def test_non_json_body_degrades(self, caplog) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")
        self._assert_degraded(_reranker(), handler, caplog)

    def test_error_response_body_is_never_logged(self, caplog) -> None:
        canary = "echoed-private-memory-canary"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text=canary)

        self._assert_degraded(_reranker(), handler, caplog)
        assert canary not in caplog.text

    def test_malformed_success_payload_values_and_keys_are_never_logged(
        self, caplog,
    ) -> None:
        value_canary = "remote-schema-value-secret-canary"
        key_canary = "remote-schema-key-secret-canary"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "results": [{
                    "index": 0,
                    "score": value_canary,
                    key_canary: "present",
                }],
            })

        self._assert_degraded(_reranker(), handler, caplog)
        assert value_canary not in caplog.text
        assert key_canary not in caplog.text

    def test_partial_results_are_refused(self, caplog) -> None:
        """A ``top_n``-style partial answer must not leave documents unscored."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 1.0},
            ]})
        self._assert_degraded(_reranker(), handler, caplog)

    def test_redirects_are_not_followed(self, caplog) -> None:
        """A rerank endpoint must not be able to pivot the request elsewhere."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "169.254.169.254" in str(request.url):
                return httpx.Response(200, json={"results": [
                    {"index": i, "relevance_score": 99.0} for i in range(3)
                ]})
            return httpx.Response(302, headers={
                "location": "http://169.254.169.254/latest/meta-data/",
            })
        rr = _reranker()
        with caplog.at_level(logging.ERROR), stub_http(handler) as rec:
            _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is False, "a redirect must never be honoured as a result"
        # Exactly one request: the redirect target was never fetched.
        assert len(rec.requests) == 1
        assert all("169.254.169.254" not in str(r.url) for r in rec.requests)
        assert "redirect" in caplog.text.lower()

    def test_oversized_response_is_aborted(self, caplog) -> None:
        from superlocalmemory.retrieval import remote_reranker as mod

        big = b'{"results": [' + b"x" * 4096 + b"]}"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big)

        rr = _reranker()
        with patch.object(mod, "_MAX_RESPONSE_BYTES", 1024):
            with caplog.at_level(logging.ERROR), stub_http(handler):
                _, applied, status = rr.rerank_with_status("q", _candidates())
        assert applied is False
        assert status == "remote_unavailable"
        assert "exceeded" in caplog.text

    def test_score_pair_returns_zero_on_failure(self) -> None:
        rr = _reranker()
        with stub_http(_raises(httpx.ConnectError("down"))):
            assert rr.score_pair("q", "doc") == 0.0

    def test_warmup_probe_failure_is_logged_as_an_error(self, caplog) -> None:
        rr = _reranker()
        with caplog.at_level(logging.ERROR):
            with stub_http(_raises(httpx.ConnectError("down"))):
                assert rr.warmup_sync() is False
        assert "Remote reranker probe failed" in caplog.text
        assert "WITHOUT reranking" in caplog.text

    def test_failed_probe_does_not_disable_later_reranking(self) -> None:
        """A booting endpoint must not cost the whole process its reranker."""
        rr = _reranker()
        with stub_http(_raises(httpx.ConnectError("still booting")), rr):
            assert rr.warmup_sync() is False
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0}), rr):
            _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is True


class TestRetryPolicy:

    def test_server_error_is_retried_once(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="model loading")

        rr = _reranker()
        with stub_http(handler) as rec:
            _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is False
        assert len(rec.requests) == 2, "5xx is transient — exactly one retry"

    def test_client_error_is_not_retried(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="no such route")

        rr = _reranker()
        with stub_http(handler) as rec:
            _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is False
        assert len(rec.requests) == 1, "4xx is a config error — no retry"

    def test_malformed_body_is_not_retried(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"nope": 1})

        rr = _reranker()
        with stub_http(handler) as rec:
            rr.rerank_with_status("q", _candidates())
        assert len(rec.requests) == 1

    def test_transient_transport_fault_recovers_on_the_retry(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("stale pooled connection")
            return httpx.Response(200, json={"results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]})

        rr = _reranker()
        with stub_http(handler) as rec:
            results, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is True
        assert len(rec.requests) == 2
        assert [f.fact_id for f, _ in results] == ["f1", "f2", "f0"]


class TestFailureVisibility:

    def test_repeat_failures_are_rate_limited_but_the_first_always_logs(
        self, caplog,
    ) -> None:
        rr = _reranker()
        with caplog.at_level(logging.ERROR):
            with stub_http(_raises(httpx.ConnectError("down"))):
                for _ in range(5):
                    rr.rerank_with_status("q", _candidates())
        errors = [r for r in caplog.records if "Remote reranker unavailable" in r.message]
        assert len(errors) == 1, "flooding the log is not visibility"
        assert rr._consecutive_failures == 5

    def test_recovery_is_announced(self, caplog) -> None:
        rr = _reranker()
        with stub_http(_raises(httpx.ConnectError("down")), rr):
            rr.rerank_with_status("q", _candidates())
        with caplog.at_level(logging.INFO):
            with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0}), rr):
                rr.rerank_with_status("q", _candidates())
        assert "Remote reranker recovered" in caplog.text
        assert rr._consecutive_failures == 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:

    def test_no_auth_header_when_no_key_configured(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0})) as rec:
            rr.rerank_with_status("q", _candidates())
        assert "authorization" not in rec.requests[0].headers

    def test_configured_key_becomes_a_bearer_token(self) -> None:
        rr = _reranker(api_key="cfg-token")
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0})) as rec:
            rr.rerank_with_status("q", _candidates())
        assert rec.requests[0].headers["authorization"] == "Bearer cfg-token"

    def test_environment_key_overrides_the_config_field(
        self, monkeypatch,
    ) -> None:
        monkeypatch.setenv(CROSS_ENCODER_API_KEY_ENV, "env-token")
        rr = _reranker(api_key="cfg-token")
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0})) as rec:
            rr.rerank_with_status("q", _candidates())
        assert rec.requests[0].headers["authorization"] == "Bearer env-token"

    def test_bearer_token_never_reaches_the_logs_on_failure(self, caplog) -> None:
        rr = _reranker(api_key="super-secret-token")
        with caplog.at_level(logging.DEBUG):
            with stub_http(_raises(httpx.ConnectError("down")), rr):
                rr.rerank_with_status("q", _candidates())
            with stub_http(_raises(httpx.ConnectError("down")), rr):
                rr.warmup_sync()
        assert "super-secret-token" not in caplog.text
        assert "Bearer" not in caplog.text

    def test_credentials_in_the_url_are_refused_outright(self) -> None:
        """httpx logs request URLs in full — a password must never get in one."""
        url = "https://admin:hunter2@reranker.example.test/v1/rerank"

        error = validate_remote_reranker_config("openai", url)
        assert error
        assert "credentials" in error
        assert "SLM_CROSS_ENCODER_API_KEY" in error
        assert "hunter2" not in error

        with pytest.raises(RemoteRerankerConfigError):
            RemoteReranker("m", url)


# ---------------------------------------------------------------------------
# Payload bounds
# ---------------------------------------------------------------------------

class TestPayloadBounds:

    def test_document_count_is_capped(self, caplog) -> None:
        from superlocalmemory.retrieval import remote_reranker as mod

        candidates = [
            (_fact(f"f{i}", f"doc {i}"), 1.0 - i / 100.0) for i in range(10)
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            sent = json.loads(request.content)["documents"]
            return httpx.Response(200, json={"results": [
                {"index": i, "relevance_score": float(i)}
                for i in range(len(sent))
            ]})

        rr = _reranker()
        with patch.object(mod, "_MAX_DOCUMENTS", 4):
            with caplog.at_level(logging.WARNING), stub_http(handler) as rec:
                results, applied, _ = rr.rerank_with_status(
                    "q", candidates, top_k=10,
                )
        assert applied is True
        assert len(json.loads(rec.requests[0].content)["documents"]) == 4
        assert len(results) == 4
        assert "exceeds the 4-document request cap" in caplog.text


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_shutdown_stops_serving_without_a_request(self) -> None:
        rr = _reranker()
        rr.shutdown()
        with stub_http(_ok_response({0: 1.0})) as rec:
            results, applied, status = rr.rerank_with_status(
                "q", _candidates(), top_k=3,
            )
        assert applied is False
        assert status == "shutdown"
        assert [f.fact_id for f, _ in results] == ["f0", "f1", "f2"]
        assert rec.requests == []

    def test_is_available_reflects_the_endpoint(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 1.0}), rr):
            assert rr.is_available is True
        with stub_http(_raises(httpx.ConnectError("down")), rr):
            assert rr.is_available is False

    def test_unload_closes_the_client_but_keeps_the_object_usable(self) -> None:
        rr = _reranker()
        with stub_http(_ok_response({0: 1.0, 1: 2.0, 2: 3.0})):
            rr.rerank_with_status("q", _candidates())
            rr.unload()
            _, applied, _ = rr.rerank_with_status("q", _candidates())
        assert applied is True

    def test_rerank_with_status_is_defined_on_the_type(self) -> None:
        """RetrievalEngine only uses the rich contract when the TYPE has it."""
        assert hasattr(RemoteReranker, "rerank_with_status")
