# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Remote (OpenAI-compatible) cross-encoder reranker.

v3.8.12 (issue #105). The mirror image of the remote EMBEDDING endpoint that
shipped in v3.4.24 (issue #16): when ``retrieval.cross_encoder_backend`` is
``"openai"`` (or ``"remote"``) and ``retrieval.cross_encoder_endpoint`` is set,
reranking is an HTTP POST to that endpoint instead of a local subprocess.

WHY THIS EXISTS
    The bundled cross-encoder, ``cross-encoder/ms-marco-MiniLM-L-12-v2``, is
    English-only. A Chinese, Japanese, or Arabic corpus was being scored by a
    model that cannot read it — a silent relevance regression with no error to
    look at. Bringing your own multilingual reranker (bge-reranker-v2-m3, a
    Qwen reranker, …) is the same escape hatch embeddings already had.

WHY IT LIVES IN THE PARENT PROCESS
    ``CrossEncoderReranker`` spawns a subprocess to keep torch/ONNX out of the
    parent. The remote path imports neither, so a subprocess would buy nothing
    and cost a fork, a PID-file singleton, a warmup handshake, and the JSON
    pipe. Issue #103's reporter hit exactly that: a machine-wide worker
    singleton blocking a reranker that was never local to begin with. The
    remote path never spawns, never touches the PID file, and never warms up a
    model. This also mirrors the embedding side, where the OpenAI-compatible
    call lives in ``core/embeddings.py`` (parent), not ``embedding_worker.py``.

WIRE PROTOCOL (Cohere-shaped ``/v1/rerank``; llama-server, TEI, Infinity, …)
    Request : {"model": "...", "query": "...", "documents": ["...", ...]}
    Response: {"results": [{"index": 0, "relevance_score": -5.94}, ...]}

    Bare-list responses (``[{"index": 0, "score": 0.9}, ...]``) are accepted
    too. Anything else is REJECTED with a precise error rather than coerced
    into plausible-looking scores — issue #103 was a lesson in what silent
    degradation costs.

FAILURE POLICY
    An unreachable, slow, or malformed endpoint degrades to fusion-score
    ordering and logs an error. It does NOT fall back to the local
    cross-encoder: a user who configured a multilingual reranker asked for it
    precisely because the local English model is wrong for their corpus, and
    quietly substituting it would recreate the bug this feature fixes.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import ipaddress
import logging
import math
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

# Backend tokens that select the remote path. "openai" is what issue #105
# asked for and matches ``embedding.provider == "openai"``, the established
# repo token for "any OpenAI-compatible HTTP endpoint". It is a slight misnomer
# — OpenAI has no rerank API and these endpoints are usually llama-server or
# TEI — so "remote" is accepted as a truthful alias.
REMOTE_CROSS_ENCODER_BACKENDS = ("openai", "remote")

# Environment override for the bearer token. Preferred over the config field:
# ``config.json`` is world-readable in many installs and is copied around.
CROSS_ENCODER_API_KEY_ENV = "SLM_CROSS_ENCODER_API_KEY"

_CONNECT_TIMEOUT_S = 5.0
_DEFAULT_READ_TIMEOUT_S = 15.0

# A rerank response is a small array of floats. Anything past this is either a
# misconfigured URL pointing at something that is not a reranker, or a hostile
# endpoint trying to exhaust memory. Bounded read, hard stop.
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Candidate pools are 50-200 in practice (semantic_top_k/bm25_top_k are 50).
# This cap only guards against a pathological pool inflating one HTTP body.
_MAX_DOCUMENTS = 512

# Only transport faults and 5xx are retried, and only once: recall is
# interactive, so a second failure must surface fast rather than pay a
# backoff sleep on the user's latency budget.
_MAX_ATTEMPTS = 2

# Consecutive failures re-log at most this often. The first failure always
# logs; the operator must never have to guess whether reranking is running.
_FAILURE_RELOG_INTERVAL_S = 60.0

class RemoteRerankerError(RuntimeError):
    """A remote rerank request failed (transport, status, or schema)."""


class RemoteRerankerConfigError(ValueError):
    """The remote reranker configuration is unusable as written."""


# ---------------------------------------------------------------------------
# Configuration (pure functions — no I/O, directly testable)
# ---------------------------------------------------------------------------

def is_remote_cross_encoder_backend(backend: str) -> bool:
    """True when ``backend`` selects the remote reranker."""
    return (backend or "").strip().lower() in REMOTE_CROSS_ENCODER_BACKENDS


def validate_remote_reranker_config(backend: str, endpoint: str) -> str | None:
    """Return an actionable error string, or None when the pair is coherent.

    Covers the issue-#103 leftover directly: an endpoint configured against a
    LOCAL backend used to be dropped on the floor by ``SLMConfig.load``. It now
    produces a named error naming both keys and the exact edit to make.
    """
    backend = (backend or "").strip()
    endpoint = (endpoint or "").strip()
    remote = is_remote_cross_encoder_backend(backend)

    if remote and not endpoint:
        return (
            f"retrieval.cross_encoder_backend={backend!r} selects the remote "
            f"reranker but retrieval.cross_encoder_endpoint is empty. Set the "
            f"endpoint (e.g. \"http://127.0.0.1:8041/v1/rerank\"), or set "
            f"cross_encoder_backend to \"\" (PyTorch) / \"onnx\" to rerank "
            f"locally."
        )
    if endpoint and not remote:
        return (
            f"retrieval.cross_encoder_endpoint is set to {endpoint!r} but "
            f"retrieval.cross_encoder_backend={backend!r} is a LOCAL backend, "
            f"so the endpoint would be ignored. Set cross_encoder_backend to "
            f"\"openai\" to use the endpoint, or remove cross_encoder_endpoint "
            f"to rerank locally."
        )
    if not remote:
        return None
    return _validate_endpoint_url(endpoint)


def _validate_endpoint_url(endpoint: str) -> str | None:
    """Scheme/host allow-listing for the operator-supplied rerank URL."""
    try:
        parsed = urlparse(endpoint)
    except ValueError as exc:
        return f"retrieval.cross_encoder_endpoint is not a valid URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return (
            f"retrieval.cross_encoder_endpoint must use http or https, got "
            f"{parsed.scheme or '(none)'!r}. SuperLocalMemory will not open "
            f"file/ftp/other schemes for reranking."
        )
    if not parsed.hostname:
        return (
            "retrieval.cross_encoder_endpoint has no host; expected something "
            "like \"http://127.0.0.1:8041/v1/rerank\"."
        )
    if parsed.query or parsed.fragment:
        return (
            "retrieval.cross_encoder_endpoint must not include a query string "
            "or fragment. Put bearer credentials in "
            "SLM_CROSS_ENCODER_API_KEY and configure a clean endpoint URL."
        )
    if parsed.username or parsed.password:
        # httpx logs "HTTP Request: POST <url>" at INFO using str(url), which
        # renders an embedded password in full. This module never logs the raw
        # URL, but it does not own the httpx logger — so credentials are
        # refused at the door instead of being trusted to stay redacted.
        return (
            "retrieval.cross_encoder_endpoint must not embed credentials "
            "(user:password@host) — the HTTP client logs request URLs in "
            "full. Put the token in SLM_CROSS_ENCODER_API_KEY (preferred) or "
            "retrieval.cross_encoder_api_key; it is sent as a Bearer header "
            "and never logged."
        )
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        return (
            "retrieval.cross_encoder_endpoint must use HTTPS for non-loopback "
            "hosts because recall queries and candidate memory text cross this "
            "connection. Plain HTTP is allowed only for localhost/loopback."
        )
    return None


def _is_loopback_host(hostname: str) -> bool:
    """Return True only for literal loopback names/addresses (no DNS trust)."""
    host = (hostname or "").rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalize_rerank_endpoint(endpoint: str) -> str:
    """Append ``/rerank`` when the URL stops at the API root.

    Mirrors the embedding path's ``/embeddings`` suffixing so a user can paste
    either ``http://host:8041/v1`` or ``http://host:8041/v1/rerank``.
    """
    url = (endpoint or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.path.endswith("/rerank"):
        return url
    return f"{url}/rerank"


def redact_endpoint(endpoint: str) -> str:
    """Drop any ``user:password@`` userinfo before an endpoint reaches a log.

    Defence in depth. ``_validate_endpoint_url`` already refuses credentialed
    URLs, so this should never have anything to strip in a configured install
    — it exists so that any future caller constructing a reranker directly
    still cannot put a password in the log.
    """
    try:
        parsed = urlparse(endpoint)
    except ValueError:
        return "<unparseable endpoint>"
    if not parsed.hostname:
        return endpoint
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username or parsed.password:
        netloc = f"***@{netloc}"
    return urlunparse(parsed._replace(netloc=netloc, query="", fragment=""))


def _redact_remote_text(text: str) -> str:
    """Remove recognized secrets and PII before a remote trust boundary."""
    from superlocalmemory.core.pii import redact_pii_text
    from superlocalmemory.core.security_primitives import redact_secrets

    return redact_pii_text(redact_secrets(str(text), aggression="high"))


# ---------------------------------------------------------------------------
# Response parsing (pure — the schema gate)
# ---------------------------------------------------------------------------

def parse_rerank_response(payload: Any, expected: int) -> list[float]:
    """Validate a ``/v1/rerank`` payload and return scores in document order.

    Raises ``RemoteRerankerError`` on ANY deviation. A rerank response that is
    not understood must abort reranking, never yield partly-invented scores:
    a wrong score silently reorders a user's memory, and nothing downstream
    can tell that apart from a good one.
    """
    results = _extract_results_array(payload)
    if len(results) != expected:
        raise RemoteRerankerError(
            f"rerank endpoint returned {len(results)} results for "
            f"{expected} documents; refusing to guess the missing scores"
        )

    scores: list[float | None] = [None] * expected
    for position, item in enumerate(results):
        if not isinstance(item, dict):
            raise RemoteRerankerError(
                f"rerank result #{position} is {type(item).__name__}, "
                f"expected an object with 'index' and 'relevance_score'"
            )
        index = _coerce_index(item, position, expected)
        if scores[index] is not None:
            raise RemoteRerankerError(
                "rerank endpoint returned a duplicate document index"
            )
        scores[index] = _coerce_score(item, index)

    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise RemoteRerankerError(
            "rerank endpoint omitted one or more document scores"
        )
    return [float(s) for s in scores]  # type: ignore[arg-type]


def _extract_results_array(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload  # text-embeddings-inference style bare array
    if not isinstance(payload, dict):
        raise RemoteRerankerError(
            f"rerank endpoint returned {type(payload).__name__}, expected a "
            f"JSON object with a 'results' array"
        )
    results = payload.get("results")
    if results is None:
        raise RemoteRerankerError(
            "rerank response has no 'results' array. Is "
            "cross_encoder_endpoint pointing at a "
            f"rerank route and not, say, /v1/embeddings?"
        )
    if not isinstance(results, list):
        raise RemoteRerankerError(
            f"rerank response 'results' is {type(results).__name__}, "
            f"expected an array"
        )
    return results


def _coerce_index(item: dict, position: int, expected: int) -> int:
    raw = item.get("index", position)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RemoteRerankerError(
            f"rerank result #{position} has a non-integer index"
        )
    if not 0 <= raw < expected:
        raise RemoteRerankerError(
            f"rerank result #{position} has an out-of-range index "
            f"for a {expected}-document request"
        )
    return raw


def _coerce_score(item: dict, index: int) -> float:
    for key in ("relevance_score", "score"):
        if key in item:
            raw = item[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RemoteRerankerError(
                    f"rerank result for index {index} has non-numeric "
                    f"{key}"
                )
            value = float(raw)
            if not math.isfinite(value):
                raise RemoteRerankerError(
                    f"rerank result for index {index} has non-finite "
                    f"{key}"
                )
            return value
    raise RemoteRerankerError(
        f"rerank result for index {index} has neither 'relevance_score' nor "
        "'score'"
    )


# ---------------------------------------------------------------------------
# The reranker
# ---------------------------------------------------------------------------

class RemoteReranker:
    """Rerank candidates via an OpenAI-compatible ``/v1/rerank`` endpoint.

    Public surface is interchangeable with ``CrossEncoderReranker`` so
    ``RetrievalEngine`` never learns which one it holds.

    Args:
        model_name: Model identifier passed to the endpoint (llama-server
            wants the served path, e.g. ``/root/model/reranker.gguf``).
        endpoint: Base or full rerank URL. ``/rerank`` is appended when absent.
        api_key: Optional bearer token. ``SLM_CROSS_ENCODER_API_KEY`` wins.
        backend: The configured backend token, for validation + logs.
        timeout_seconds: Per-request read budget.

    Raises:
        RemoteRerankerConfigError: the backend/endpoint pair is unusable.
    """

    def __init__(
        self,
        model_name: str,
        endpoint: str,
        *,
        api_key: str = "",
        backend: str = "openai",
        timeout_seconds: float = _DEFAULT_READ_TIMEOUT_S,
    ) -> None:
        error = validate_remote_reranker_config(backend, endpoint)
        if error:
            raise RemoteRerankerConfigError(error)

        self._model_name = model_name
        self._backend = backend
        self._endpoint = normalize_rerank_endpoint(endpoint)
        self.safe_endpoint = redact_endpoint(self._endpoint)
        self._api_key = os.environ.get(CROSS_ENCODER_API_KEY_ENV, "") or api_key
        try:
            self._read_timeout = max(1.0, float(timeout_seconds))
        except (TypeError, ValueError):
            self._read_timeout = _DEFAULT_READ_TIMEOUT_S

        self._client: Any = None
        self._client_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._consecutive_failures = 0
        self._last_failure_log = 0.0
        self._probe_ok = False

    # -- lifecycle ---------------------------------------------------------

    def warmup_sync(self, timeout: float = _DEFAULT_READ_TIMEOUT_S) -> bool:
        """Probe the endpoint once so startup states reachability out loud.

        Diagnostic only: a failed probe never disables reranking, because an
        endpoint that is still booting will serve the next real recall fine.
        """
        if self._shutdown.is_set():
            return False
        try:
            self._request_scores("ping", ["SuperLocalMemory reranker probe"])
        except RemoteRerankerError as exc:
            self._probe_ok = False
            logger.error(
                "Remote reranker probe failed for %s (model=%s): %s. Recall "
                "will run WITHOUT reranking until the endpoint answers. "
                "Verify retrieval.cross_encoder_endpoint and that the service "
                "is up.",
                self.safe_endpoint, self._model_name, exc,
            )
            return False
        self._probe_ok = True
        logger.info(
            "Remote reranker ready: %s (model=%s, backend=%s)",
            self.safe_endpoint, self._model_name, self._backend,
        )
        return True

    def unload(self) -> None:
        """Release the pooled HTTP connections; the object stays usable."""
        self._close_client()

    def shutdown(self, timeout: float = 3.0) -> None:  # noqa: ARG002 - parity
        """Stop serving and close the HTTP client."""
        self._shutdown.set()
        self._close_client()

    def __del__(self) -> None:
        try:
            self._close_client()
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Whether the endpoint answers a probe right now."""
        if self._shutdown.is_set():
            return False
        try:
            self._request_scores("ping", ["SuperLocalMemory reranker probe"])
        except RemoteRerankerError:
            return False
        return True

    # -- public reranking --------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[tuple[AtomicFact, float]],
        top_k: int = 10,
    ) -> list[tuple[AtomicFact, float]]:
        """Rerank ``candidates``; fusion order is returned when the endpoint fails."""
        results, _, _ = self.rerank_with_status(query, candidates, top_k=top_k)
        return results

    def rerank_with_status(
        self,
        query: str,
        candidates: list[tuple[AtomicFact, float]],
        top_k: int = 10,
    ) -> tuple[list[tuple[AtomicFact, float]], bool, str]:
        """Return results plus whether remote reranking actually ran."""
        if not candidates:
            return [], False, "no_candidates"
        if self._shutdown.is_set():
            return self._fusion_order(candidates)[:top_k], False, "shutdown"

        ranked = self._fusion_order(candidates)
        if len(ranked) > _MAX_DOCUMENTS:
            # Unreachable with stock config (semantic_top_k/bm25_top_k are 50).
            # RetrievalEngine keeps every fused result and assigns the batch
            # minimum to any fact absent from the rerank map, so the excluded
            # tail — already the lowest-fusion candidates — is demoted, not
            # lost.
            logger.warning(
                "Remote reranker: %d candidates exceeds the %d-document "
                "request cap; reranking the top %d by fusion score and "
                "dropping the rest",
                len(ranked), _MAX_DOCUMENTS, _MAX_DOCUMENTS,
            )
            ranked = ranked[:_MAX_DOCUMENTS]

        try:
            scores = self._request_scores(
                query, [fact.content for fact, _ in ranked],
            )
        except RemoteRerankerError as exc:
            self._note_failure(exc)
            return ranked[:top_k], False, "remote_unavailable"

        self._note_success()
        scored = [
            (fact, float(score))
            for (fact, _), score in zip(ranked, scores)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k], True, "applied"

    def score_pair(self, query: str, document: str) -> float:
        """Score one (query, document) pair; 0.0 when the endpoint fails."""
        if self._shutdown.is_set():
            return 0.0
        try:
            return self._request_scores(query, [document])[0]
        except RemoteRerankerError as exc:
            self._note_failure(exc)
            return 0.0

    # -- HTTP --------------------------------------------------------------

    def _request_scores(self, query: str, documents: list[str]) -> list[float]:
        """POST one rerank request, retrying only genuinely transient faults."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            # Never logged: no error path in this module formats `headers`.
            headers["Authorization"] = f"Bearer {self._api_key}"
        body = {
            "model": self._model_name,
            "query": _redact_remote_text(query),
            "documents": [_redact_remote_text(document) for document in documents],
        }

        last_error: RemoteRerankerError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                payload = self._post(headers, body)
            except _RetryableRemoteError as exc:
                last_error = RemoteRerankerError(str(exc))
                if attempt < _MAX_ATTEMPTS - 1:
                    continue
                break
            return parse_rerank_response(payload, len(documents))
        raise RemoteRerankerError(
            f"remote reranker at {self.safe_endpoint} failed after "
            f"{_MAX_ATTEMPTS} attempts: {last_error}"
        )

    def _post(self, headers: dict[str, str], body: dict[str, Any]) -> Any:
        """Send the request and return parsed JSON, with a bounded body read."""
        import httpx

        client = self._get_client()
        try:
            with client.stream(
                "POST", self._endpoint, headers=headers, json=body,
            ) as resp:
                raw = _read_bounded(resp)
                if 300 <= resp.status_code < 400:
                    # Redirects are not followed: a rerank endpoint that
                    # bounces us elsewhere is either misconfigured or an
                    # attempt to pivot this outbound request at a host the
                    # operator never approved.
                    raise RemoteRerankerError(
                        f"rerank endpoint {self.safe_endpoint} replied HTTP "
                        f"{resp.status_code} (redirect). Redirects are not "
                        f"followed — configure the final URL directly."
                    )
                if resp.status_code >= 400:
                    message = (
                        f"HTTP {resp.status_code} from {self.safe_endpoint}; "
                        "response body suppressed"
                    )
                    if resp.status_code >= 500:
                        raise _RetryableRemoteError(message)
                    raise RemoteRerankerError(message)
        except httpx.TransportError as exc:
            raise _RetryableRemoteError(
                f"cannot reach {self.safe_endpoint}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RemoteRerankerError(
                f"rerank endpoint {self.safe_endpoint} returned non-JSON "
                f"({exc})"
            ) from exc

    def _get_client(self) -> Any:
        import httpx

        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(
                        connect=_CONNECT_TIMEOUT_S,
                        read=self._read_timeout,
                        write=10.0,
                        pool=5.0,
                    ),
                    follow_redirects=False,
                )
            return self._client

    def _close_client(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # -- failure visibility ------------------------------------------------

    def _note_failure(self, exc: Exception) -> None:
        """Make a degraded reranker impossible to miss, without log flooding."""
        self._consecutive_failures += 1
        now = time.time()
        if (
            self._consecutive_failures == 1
            or now - self._last_failure_log >= _FAILURE_RELOG_INTERVAL_S
        ):
            logger.error(
                "Remote reranker unavailable (%d consecutive failures): %s. "
                "Recall is returning fusion-ranked results with NO reranking. "
                "SuperLocalMemory will not silently substitute the local "
                "English cross-encoder for your configured model.",
                self._consecutive_failures, exc,
            )
            self._last_failure_log = now

    def _note_success(self) -> None:
        if self._consecutive_failures:
            logger.info(
                "Remote reranker recovered after %d consecutive failures (%s)",
                self._consecutive_failures, self.safe_endpoint,
            )
            self._consecutive_failures = 0

    @staticmethod
    def _fusion_order(
        candidates: list[tuple[AtomicFact, float]],
    ) -> list[tuple[AtomicFact, float]]:
        return sorted(candidates, key=lambda pair: pair[1], reverse=True)


class _RetryableRemoteError(RemoteRerankerError):
    """Internal marker: this failure is worth exactly one more attempt."""


def _read_bounded(resp: Any) -> bytes:
    """Read a streaming response body, refusing to buffer past the cap."""
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes():
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise RemoteRerankerError(
                f"rerank response exceeded {_MAX_RESPONSE_BYTES} bytes; "
                f"aborting the read. Is cross_encoder_endpoint pointing at a "
                f"rerank route?"
            )
        chunks.append(chunk)
    return b"".join(chunks)
