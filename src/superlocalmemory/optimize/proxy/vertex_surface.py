"""vertex_surface.py — Vertex AI passthrough proxy surface (WP-11).

Transparent passthrough: forward Authorization bearer untouched (AC-2),
cache by body content never by token (SEC), no SSE — always single JSON.

Key design decisions (per LLD §5 STAGE-5 RESOLUTIONS):
  CRIT-1: Route by FastAPI PATH /v1/projects/{vertex_path:path}, NOT hostname.
           Upstream host is reconstructed from /locations/{region}/ in the path.
  CRIT-2: Vertex bodies have no model/messages/system (model in PATH, prompts
           under 'contents'). The provider=='vertex' branch in CacheManager.build_key
           extracts these correctly — preventing all-requests-same-key collision.
  D2: Pin upstream host to https://{location}-aiplatform.googleapis.com from path.
      IGNORE providers.vertex.base_url — honoring it is an SSRF surface.
  SECURITY (AC-3): bearer token structurally excluded from cache key, value,
           logs, and stored ProxyRequest.headers (redacted via _redact_headers).

WP-11a (gemini-native fix) lives in _helpers.py:_GEMINI_NATIVE_FORWARD_HEADERS.
"""

from __future__ import annotations

import json
import logging
import re

from fastapi.requests import Request
from fastapi.responses import Response

from superlocalmemory.optimize.proxy._helpers import (
    _VERTEX_FORWARD_HEADERS,
    _fail_open_forward,
    _filter_response_headers,
    _redact_headers,
    _safe_cache_check,
    _safe_cache_hit_callbacks,
    _safe_cache_store,
    capture_passthrough_forward,
)
from superlocalmemory.optimize.proxy.capture import capture_enabled
from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest

logger = logging.getLogger("slm.optimize.proxy.vertex")

# ---------------------------------------------------------------------------
# SSRF guard — LLD §5
# ---------------------------------------------------------------------------

# Accepts ONLY:
#   {project}/locations/{location}/publishers/google/models/{model}:{method}
# Rejects:
#   - ../ traversal (no dots allowed in segments via character classes)
#   - unknown methods (countTokens, EXEC, etc.)
#   - overlong segments
_VERTEX_PATH_RE = re.compile(
    r"^(?P<project>[a-zA-Z0-9_\-]{1,63})"
    r"/locations/(?P<location>[a-z0-9\-]{1,40})"
    r"/publishers/google/models/(?P<model>[a-zA-Z0-9._\-]{1,128})"
    r":(?P<method>generateContent|streamGenerateContent)$"
)


def _validate_vertex_path(path: str) -> bool:
    """Return True iff path matches the Vertex AI path pattern (SSRF guard).

    Rejects ../, traversal characters, unknown methods, and overlong segments.
    """
    return bool(_VERTEX_PATH_RE.match(path))


def _parse_vertex_path(
    path: str,
) -> tuple[str, str, str, str] | None:
    """Parse a validated Vertex path into (project, location, model, method).

    Returns None on parse failure (caller should return 400).
    """
    m = _VERTEX_PATH_RE.match(path)
    if m is None:
        return None
    return (
        m.group("project"),
        m.group("location"),
        m.group("model"),
        m.group("method"),
    )


def _build_vertex_upstream_url(location: str, full_path: str) -> str:
    """Build the Vertex upstream URL.

    D2 (LOCKED): pin host to https://{location}-aiplatform.googleapis.com
    from the path — IGNORE providers.vertex.base_url (honoring it = SSRF).
    """
    host = f"https://{location}-aiplatform.googleapis.com"
    return f"{host}{full_path}"


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

async def handle_vertex_generative(
    proxy: object,
    request: Request,
    vertex_path: str,
) -> Response:
    """Handle POST /v1/projects/{vertex_path:path}.

    Full pipeline: validate → parse → cache check → POST upstream → cache store.
    Bearer token is NEVER in the cache key (structurally — build_key reads only body).
    Always returns a single JSON Response — no SSE/StreamingResponse (LLD §1).
    Fail-open on unexpected exceptions.
    """
    request_id = await proxy.next_request_id()

    # ── SSRF guard — validate path BEFORE any upstream contact (AC-3 / LLD §4) ──
    if not _validate_vertex_path(vertex_path):
        logger.warning(
            "[%s] handle_vertex_generative: rejected invalid path=%r (SSRF guard)",
            request_id, vertex_path,
        )
        return Response(
            content=b'{"error":{"code":400,"message":"Invalid Vertex path",'
                    b'"status":"INVALID_ARGUMENT"}}',
            status_code=400,
            media_type="application/json",
        )

    parsed = _parse_vertex_path(vertex_path)
    if parsed is None:
        # Should not reach here after _validate_vertex_path, but be defensive.
        logger.error("[%s] handle_vertex_generative: parse failed after validation", request_id)
        return Response(
            content=b'{"error":{"code":400,"message":"Path parse error",'
                    b'"status":"INVALID_ARGUMENT"}}',
            status_code=400,
            media_type="application/json",
        )

    _project, location, _model, _method = parsed
    upstream_url = _build_vertex_upstream_url(location, str(request.url.path))

    try:
        body_bytes = await request.body()

        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        # v3.6.10 shadow-capture: pure passthrough + corpus record.
        if capture_enabled():
            return await capture_passthrough_forward(
                proxy, request,
                provider="vertex",
                upstream_url=upstream_url,
                allowed_headers=_VERTEX_FORWARD_HEADERS,
                request_id=request_id,
                model_hint=_model,
                sse_parser=None,
                is_stream=False,
            )

        # SECURITY (AC-3): headers stored in ProxyRequest are redacted.
        # Bearer token is structurally excluded from the cache key because
        # build_key reads only body-derived fields (key_builder.py:90-105).
        ctx = ProxyRequest(
            provider="vertex",
            method="POST",
            path=str(request.url.path),
            headers=_redact_headers(dict(request.headers)),  # token → [REDACTED]
            body=body,
            body_bytes=body_bytes,
            request_id=request_id,
            stream=False,
            has_tools=False,
        )

        # Build forward headers — Authorization byte-identical (AC-2).
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in _VERTEX_FORWARD_HEADERS
        }

        # ── Cache check ────────────────────────────────────────────────────
        cache_result = None
        if proxy.hooks.cache:
            cache_result = await _safe_cache_check(proxy.hooks, ctx)
            if cache_result and cache_result.hit and cache_result.data:
                logger.debug(
                    "[%s] Vertex cache HIT key=%s",
                    request_id, cache_result.cache_key,
                )
                await _safe_cache_hit_callbacks(
                    proxy.hooks, ctx, cache_result.data, tokens_saved=0
                )
                return Response(
                    content=cache_result.data,
                    status_code=200,
                    media_type="application/json",
                )

        fwd_headers["content-length"] = str(len(body_bytes))

        # ── POST upstream (single JSON — no SSE) ──────────────────────────
        upstream_resp = await proxy.http_client.post(
            upstream_url, content=body_bytes, headers=fwd_headers,
        )
        resp_bytes = upstream_resp.content

        # ── Cache store on 200 only ────────────────────────────────────────
        if (
            upstream_resp.status_code == 200
            and proxy.hooks.cache
            and cache_result is not None
            and cache_result.cache_key
        ):
            prov_resp = ProviderResponse(
                modified=False,
                body={},
                body_bytes=resp_bytes,
                tokens_before=0,
                tokens_after=0,
                strategy="none",
            )
            await _safe_cache_store(proxy.hooks, ctx, prov_resp)

        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )

    except Exception as exc:
        logger.error(
            "[%s] handle_vertex_generative exc=%r — fail-open", request_id, exc
        )
        return await _fail_open_forward(proxy, request, upstream_url)
