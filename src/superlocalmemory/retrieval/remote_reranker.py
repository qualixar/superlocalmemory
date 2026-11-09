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
import logging
import math
import os
import threading
import time
from typing import Any

from superlocalmemory.retrieval.remote_rerank_guard import (
    MAX_IN_FLIGHT,
    CircuitBreaker,
    DeadlineExceeded,
    TooManyInFlight,
    call_within,
)
from superlocalmemory.storage.models import AtomicFact

logger = logging.getLogger(__name__)

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

# Only fast transport faults (refused, reset, a stale pooled connection) and
# 5xx are retried, once, and only while the call's deadline has time left. A
# TIMEOUT is never retried: it has already spent the budget, and in 4.1.18 the
# retry is what turned one 15 s wait into two (30.5 s per recall, measured).
_MAX_ATTEMPTS = 2

# The most one rerank call may add to a recall, retries included, enforced as a
# wall clock (remote_rerank_guard.call_within). The recall ceiling is 2.0 s for
# the WHOLE answer; this leaves the other half for retrieval itself.
# ``cross_encoder_timeout_seconds`` (15 s by default) can only make it tighter.
# Quality cost, stated: an endpoint that needs longer than this per batch never
# reranks -- its recalls return fusion order, reported as remote_unavailable.
# Someone who knowingly accepts slower recalls can raise it with the env var.
_RECALL_DEADLINE_S = 1.0
RECALL_DEADLINE_ENV = "SLM_REMOTE_RERANK_DEADLINE_S"
_MIN_DEADLINE_S = 0.05

# Consecutive failures re-log at most this often. The first failure always
# logs; the operator must never have to guess whether reranking is running.
_FAILURE_RELOG_INTERVAL_S = 60.0

class RemoteRerankerError(RuntimeError):
    """A remote rerank request failed (transport, status, or schema)."""


class RemoteRerankerTimeout(RemoteRerankerError):
    """The endpoint did not answer inside the call's deadline."""


class _RemoteRerankerBusy(RemoteRerankerError):
    """Every request slot is still held; the endpoint was not called."""


def _positive_seconds(value: Any, fallback: float) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return fallback
    return seconds if math.isfinite(seconds) and seconds > 0 else fallback


def _recall_deadline_s(configured_timeout_s: float) -> float:
    """The per-call wall-clock budget: the recall budget, or tighter."""
    budget = _RECALL_DEADLINE_S
    raw = os.environ.get(RECALL_DEADLINE_ENV, "").strip()
    if raw:
        try:
            value = float(raw)
            if math.isfinite(value) and value > 0:
                budget = value
        except ValueError:
            logger.warning(
                "%s=%r is not a number of seconds; using %.1fs",
                RECALL_DEADLINE_ENV, raw, budget,
            )
    return max(_MIN_DEADLINE_S, min(budget, configured_timeout_s))


class RemoteRerankerConfigError(ValueError):
    """The remote reranker configuration is unusable as written."""


# Configuration (pure functions) lives in remote_reranker_config; re-exported
# here because callers and tests have always imported it from this module.
from superlocalmemory.retrieval.remote_reranker_config import (  # noqa: E402,F401
    REMOTE_CROSS_ENCODER_BACKENDS,
    _is_loopback_host,
    _is_private_lan_host,
    _validate_endpoint_url,
    is_remote_cross_encoder_backend,
    normalize_rerank_endpoint,
    redact_endpoint,
    validate_remote_reranker_config,
)


def _redact_remote_text(text: str) -> str:
    """Remove recognized secrets and PII before a remote trust boundary.

    The hosted screen: every credential becomes ``[redacted]`` with no type
    and no last four characters (the high-aggression scan alone kept both).
    """
    from superlocalmemory.core.pii import redact_pii_text
    from superlocalmemory.retrieval.hosted_redaction import redact_for_hosted_judge

    return redact_pii_text(redact_for_hosted_judge(str(text)))


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
            "rerank route and not, say, /v1/embeddings?"
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
        trust_plain_http_lan: bool = True,
        deadline_seconds: float | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        error = validate_remote_reranker_config(
            backend, endpoint, trust_plain_http_lan=trust_plain_http_lan,
        )
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
        if deadline_seconds is not None:
            self.deadline_seconds = max(_MIN_DEADLINE_S, float(deadline_seconds))
        else:
            self.deadline_seconds = _recall_deadline_s(
                _positive_seconds(timeout_seconds, self._read_timeout),
            )
        self._breaker = breaker or CircuitBreaker()
        self._slots = threading.BoundedSemaphore(MAX_IN_FLIGHT)

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

        scores = self._guarded_scores(query, [fact.content for fact, _ in ranked])
        if scores is None:
            return ranked[:top_k], False, "remote_unavailable"

        scored = [
            (fact, float(score))
            for (fact, _), score in zip(ranked, scores)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].fact_id))
        return scored[:top_k], True, "applied"

    def score_pair(self, query: str, document: str) -> float:
        """Score one (query, document) pair; 0.0 when the endpoint fails."""
        if self._shutdown.is_set():
            return 0.0
        scores = self._guarded_scores(query, [document])
        return scores[0] if scores else 0.0

    # -- the guard: breaker + one deadline per call ------------------------

    def _guarded_scores(self, query: str, documents: list[str]) -> list[float] | None:
        """Scores, or None when the endpoint was skipped or failed (logged).

        A paused endpoint is not called at all; when its pause has run out,
        this starts the single background probe and still answers at once.
        """
        decision = self._breaker.decide()
        if decision.start_probe:
            self._start_probe()
        if not decision.call:
            return None
        try:
            scores = self._request_scores(query, documents)
        except _RemoteRerankerBusy as exc:
            # Not the endpoint's fault yet: the calls holding the slots will
            # each succeed or fail on their own, inside their deadline.
            logger.debug("Remote reranker skipped for this recall: %s", exc)
            return None
        except RemoteRerankerError as exc:
            self._note_failure(exc)
            return None
        self._note_success()
        return scores

    def _start_probe(self) -> None:
        """Check a paused endpoint off the recall path; one probe at a time."""
        def _probe() -> None:
            ok = False
            try:
                if not self._shutdown.is_set():
                    self._request_scores("ping", ["SuperLocalMemory reranker probe"])
                    ok = True
            except RemoteRerankerError as exc:
                logger.debug("Remote reranker probe: still unavailable: %s", exc)
            except Exception as exc:  # noqa: BLE001 -- a probe must always finish
                logger.debug("Remote reranker probe raised: %s", exc)
            finally:
                self._breaker.probe_finished(ok)
            if ok:
                logger.info(
                    "Remote reranker %s answers again; the next recall tries it",
                    self.safe_endpoint,
                )

        try:
            threading.Thread(target=_probe, name="remote-rerank-probe",
                             daemon=True).start()
        except Exception as exc:  # noqa: BLE001 -- e.g. no threads at shutdown
            self._breaker.probe_finished(False)
            logger.debug("Remote reranker probe not started: %s", exc)

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

        expected = len(documents)
        try:
            return call_within(
                lambda deadline_at: self._attempts(headers, body, expected, deadline_at),
                self.deadline_seconds,
                self._slots,
            )
        except DeadlineExceeded as exc:
            raise RemoteRerankerTimeout(
                f"remote reranker at {self.safe_endpoint} did not answer within "
                f"{self.deadline_seconds:g}s"
            ) from exc
        except TooManyInFlight as exc:
            raise _RemoteRerankerBusy(str(exc)) from exc

    def _attempts(
        self, headers: dict[str, str], body: dict[str, Any], expected: int,
        deadline_at: float,
    ) -> list[float]:
        """The request and its one permitted retry, inside ``deadline_at``."""
        last_error: RemoteRerankerError | None = None
        attempts = 0
        while attempts < _MAX_ATTEMPTS:
            if deadline_at - time.monotonic() <= 0:
                break
            attempts += 1
            try:
                payload = self._post(headers, body, deadline_at)
            except RemoteRerankerTimeout:
                raise  # never retried: it already spent the budget
            except _RetryableRemoteError as exc:
                last_error = RemoteRerankerError(str(exc))
                continue
            return parse_rerank_response(payload, expected)
        raise RemoteRerankerError(
            f"remote reranker at {self.safe_endpoint} failed after "
            f"{attempts} attempt(s): {last_error}"
        )

    def _post(
        self, headers: dict[str, str], body: dict[str, Any], deadline_at: float,
    ) -> Any:
        """Send the request and return parsed JSON, with a bounded body read.

        Every transport phase is bounded by the time left, so the thread doing
        this ends soon after the deadline even when nobody waits for it.
        """
        import httpx

        client = self._get_client()
        left = max(_MIN_DEADLINE_S, deadline_at - time.monotonic())
        timeout = httpx.Timeout(
            connect=min(_CONNECT_TIMEOUT_S, left), read=left, write=left, pool=left,
        )
        try:
            with client.stream(
                "POST", self._endpoint, headers=headers, json=body,
                timeout=timeout,
            ) as resp:
                raw = _read_bounded(resp, deadline_at)
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
        except httpx.TimeoutException as exc:
            raise RemoteRerankerTimeout(
                f"{self.safe_endpoint} timed out: {type(exc).__name__}"
            ) from exc
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

        from superlocalmemory.core.outbound_http import GatedClient

        with self._client_lock:
            if self._client is None:
                # Per-request timeouts (see _post) override these; they are
                # only the ceiling for a request that somehow sets none.
                # The gated client never follows a redirect, screens the body
                # on the final URL and never sends a local endpoint's request
                # through a proxy.
                bound = self.deadline_seconds
                self._client = GatedClient(
                    timeout=httpx.Timeout(
                        connect=min(_CONNECT_TIMEOUT_S, bound),
                        read=bound, write=bound, pool=bound,
                    ),
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
        if self._breaker.record_failure(
            timed_out=isinstance(exc, RemoteRerankerTimeout),
        ):
            logger.error(
                "Remote reranker unavailable (%d consecutive failures): %s. "
                "Pausing it for %.0fs: recall no longer waits on it and returns "
                "fusion-ranked results with NO reranking until a background "
                "check finds it answering again. SuperLocalMemory will not "
                "silently substitute the local English cross-encoder for your "
                "configured model.",
                self._consecutive_failures, exc, self._breaker.cooldown_s,
            )
            self._last_failure_log = time.time()
            return
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
        self._breaker.record_success()
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
        return sorted(candidates, key=lambda pair: (-pair[1], pair[0].fact_id))


class _RetryableRemoteError(RemoteRerankerError):
    """Internal marker: this failure is worth exactly one more attempt."""


def _read_bounded(resp: Any, deadline_at: float | None = None) -> bytes:
    """Read a streaming response body, refusing to buffer past the cap.

    ``deadline_at`` also ends a reply that trickles in: each chunk arriving
    inside the read timeout would otherwise keep the read going for ever.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes():
        if deadline_at is not None and time.monotonic() > deadline_at:
            raise RemoteRerankerTimeout(
                "rerank response was still arriving at the deadline"
            )
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise RemoteRerankerError(
                f"rerank response exceeded {_MAX_RESPONSE_BYTES} bytes; "
                f"aborting the read. Is cross_encoder_endpoint pointing at a "
                f"rerank route?"
            )
        chunks.append(chunk)
    return b"".join(chunks)
