"""gemini_surface.py — Gemini native and OpenAI-compat surfaces.

BUG-FIX (v3.6.4): Both surfaces were pure pass-throughs with zero cache or
compress operations.  Fixed:

  - handle_gemini_native():  full cache check → compress → stream-and-cache
    pipeline, matching the pattern in anthropic_surface.py.  Streaming uses
    _parse_gemini_sse_to_json / _gemini_sse_from_cached_json helpers that
    understand Gemini's SSE format (no 'event:' prefix; final chunk carries
    'finishReason').  Non-streaming response is cached verbatim.

  - handle_gemini_openai_compat(): cache check → compress → forward pipeline
    for POST requests.  The /v1beta/openai/chat/completions endpoint speaks
    standard OpenAI JSON, so _parse_openai_sse_to_json and the OpenAI SSE
    replay helper are reused after importing from openai_surface.

Gemini CLI and AGY/Antigravity (when using Google models) both hit the native
surface.  Codex/Antigravity in OpenAI-compat mode hit the compat surface.

NOTE ON GOOGLE GENAI ENV VARS (v3.6.4):
  Set GOOGLE_GENAI_BASE_URL=http://127.0.0.1:8765 for google-genai SDK >=0.6
  Set GOOGLE_API_BASE=http://127.0.0.1:8765  for older google-generativeai SDK
  Gemini CLI (≤2025-06-18): GEMINI_API_HOST=http://127.0.0.1:8765
  AGY (Antigravity): GOOGLE_GENAI_BASE_URL=http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse

from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse

from superlocalmemory.optimize.proxy._helpers import (
    _GEMINI_NATIVE_FORWARD_HEADERS,
    _GEMINI_OPENAI_COMPAT_FORWARD_HEADERS,
    _body_has_tools,
    _derive_tenant_id,
    _fail_open_forward,
    _filter_response_headers,
    _redact_headers,
    _safe_cache_check,
    _safe_cache_hit_callbacks,
    _safe_cache_store,
    _safe_compress,
    _stream_and_cache_forward,
    _stream_forward,
    capture_passthrough_forward,
)
from superlocalmemory.optimize.proxy.capture import capture_enabled
from superlocalmemory.optimize.proxy.openai_surface import _parse_openai_sse_to_json
from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest

logger = logging.getLogger("slm.optimize.proxy.gemini")

_GEMINI_UPSTREAM_BASE = "https://generativelanguage.googleapis.com"

# SSRF guard: accept BOTH `models/<name>:<method>` AND `<name>:<method>`
# (FastAPI's `:path` converter may strip the `models/` segment depending on
# route declaration; we normalize on the server side either way).
_GEMINI_PATH_RE = re.compile(
    r"^(?:models/)?[a-zA-Z0-9._\-]{1,128}:(generateContent|streamGenerateContent|countTokens)$"
)

_GEMINI_ALLOWED_QUERY_PARAMS = frozenset(["pagesize", "pagetoken"])


def _validate_gemini_path(model_and_method: str) -> bool:
    return bool(_GEMINI_PATH_RE.match(model_and_method))


# ---------------------------------------------------------------------------
# Gemini native SSE helpers
# ---------------------------------------------------------------------------

def _parse_gemini_sse_to_json(sse_bytes: bytes) -> bytes | None:
    """Assemble a Gemini SSE stream into a single generateContent JSON response.

    Gemini streaming format (no 'event:' prefix — just 'data:' lines):
        data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}],"usageMetadata":{...}}
        data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{...}}

    Completeness gate: returns None if no candidate ever emits a non-null
    finishReason — ensures incomplete streams are not cached.  This is the
    Gemini equivalent of checking 'message_stop' (Anthropic) or '[DONE]' (OpenAI).
    """
    parts_by_idx: dict[int, list[str]] = {}
    finish_reasons: dict[int, str] = {}
    usage_metadata: dict = {}
    model_version = ""

    for raw_line in sse_bytes.decode("utf-8", errors="replace").split("\n"):
        line = raw_line.rstrip("\r")
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if "usageMetadata" in chunk:
            usage_metadata = chunk["usageMetadata"]
        if "modelVersion" in chunk:
            model_version = chunk["modelVersion"]

        for candidate in chunk.get("candidates", []):
            idx = candidate.get("index", 0)
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text")
                if text is not None:
                    parts_by_idx.setdefault(idx, []).append(text)
            fr = candidate.get("finishReason")
            # Gemini emits finishReason as a string on the final chunk.
            # Intermediate chunks omit it or include None / "null".
            if fr and fr not in (None, "null", "FINISH_REASON_UNSPECIFIED"):
                finish_reasons[idx] = fr

    # Completeness gate
    if not finish_reasons:
        return None

    candidates = []
    for idx in sorted(parts_by_idx.keys()):
        candidates.append({
            "content": {
                "parts": [{"text": "".join(parts_by_idx.get(idx, []))}],
                "role": "model",
            },
            "finishReason": finish_reasons.get(idx, "STOP"),
            "index": idx,
        })

    if not candidates:
        return None

    result: dict = {"candidates": candidates, "usageMetadata": usage_metadata}
    if model_version:
        result["modelVersion"] = model_version

    return json.dumps(result, separators=(",", ":")).encode("utf-8")


def _gemini_sse_from_cached_json(cached_bytes: bytes) -> "StreamingResponse | None":
    """Replay a stored generateContent JSON as a Gemini SSE stream.

    Emits the text in 100-char chunks followed by a final chunk containing
    finishReason + usageMetadata, matching what the real API would send.
    """
    try:
        resp = json.loads(cached_bytes)
    except (json.JSONDecodeError, ValueError):
        return None

    if "candidates" not in resp:
        return None

    async def _generate():
        for candidate in resp.get("candidates", []):
            idx = candidate.get("index", 0)
            text = "".join(
                p.get("text", "")
                for p in candidate.get("content", {}).get("parts", [])
            )
            finish_reason = candidate.get("finishReason", "STOP")

            # Emit text in chunks
            chunk_size = 100
            for start in range(0, max(len(text), 1), chunk_size):
                piece = text[start: start + chunk_size]
                chunk = {
                    "candidates": [{
                        "content": {"parts": [{"text": piece}], "role": "model"},
                        "index": idx,
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            # Final chunk with finishReason and usageMetadata
            final: dict = {
                "candidates": [{
                    "content": {"parts": [{"text": ""}], "role": "model"},
                    "finishReason": finish_reason,
                    "index": idx,
                }],
                "usageMetadata": resp.get("usageMetadata", {}),
            }
            if "modelVersion" in resp:
                final["modelVersion"] = resp["modelVersion"]
            yield f"data: {json.dumps(final)}\n\n".encode()

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_gemini_native(
    proxy: object,
    request: Request,
    model_and_method: str,
) -> Response:
    """Handle /v1beta/models/{model}:{method} — native Gemini API.

    BUG-FIX (v3.6.4): Was a pure pass-through.  Now applies cache check,
    optional compression, and post-stream cache store for streaming responses.
    Non-streaming responses are cached verbatim.

    Streaming: _parse_gemini_sse_to_json validates completeness via finishReason.
    Non-streaming: response JSON stored directly.
    """
    request_id = await proxy.next_request_id()

    if not _validate_gemini_path(model_and_method):
        logger.warning(
            "[%s] handle_gemini_native: rejected invalid path param=%r (SSRF guard)",
            request_id, model_and_method,
        )
        return Response(
            content=b'{"error":{"code":400,"message":"Invalid model/method path",'
                    b'"status":"INVALID_ARGUMENT"}}',
            status_code=400,
            media_type="application/json",
        )

    upstream_url = f"{_GEMINI_UPSTREAM_BASE}/v1beta/{model_and_method}"

    try:
        body_bytes = await request.body()

        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        stream = "streamGenerateContent" in model_and_method
        has_tools = _body_has_tools(body)

        # v3.6.10 shadow-capture (plan §7): pure passthrough + corpus record.
        if capture_enabled():
            cap_model = model_and_method.split(":", 1)[0].replace("models/", "")
            cap_url = upstream_url
            if stream:
                _allowed = {
                    k: v for k, v in request.query_params.items()
                    if k.lower() in _GEMINI_ALLOWED_QUERY_PARAMS
                }
                _allowed["alt"] = "sse"
                cap_url = f"{upstream_url}?{urllib.parse.urlencode(_allowed)}"
            return await capture_passthrough_forward(
                proxy, request, provider="gemini", upstream_url=cap_url,
                allowed_headers=_GEMINI_NATIVE_FORWARD_HEADERS, request_id=request_id,
                model_hint=cap_model,
                sse_parser=_parse_gemini_sse_to_json, is_stream=stream,
            )

        # SECURITY (WP-D): derive tenant BEFORE _redact_headers strips the key.
        # Gemini uses x-goog-api-key; OAuth uses authorization bearer.
        _raw_key = (
            request.headers.get("x-goog-api-key")
            or request.headers.get("authorization")
        )
        _tenant_id = _derive_tenant_id("gemini", _raw_key)

        ctx = ProxyRequest(
            provider="gemini",
            method="POST",
            path=f"/v1beta/{model_and_method}",
            headers=_redact_headers(dict(request.headers)),
            body=body,
            body_bytes=body_bytes,
            request_id=request_id,
            stream=stream,
            has_tools=has_tools,
        )

        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in _GEMINI_NATIVE_FORWARD_HEADERS
        }

        if stream:
            # ── 1. Cache check ──────────────────────────────────────────────
            if proxy.hooks.cache:
                cache_result = await _safe_cache_check(proxy.hooks, ctx, tenant_id=_tenant_id)
                if cache_result and cache_result.hit and cache_result.data:
                    logger.debug(
                        "[%s] Gemini native streaming cache HIT key=%s",
                        request_id, cache_result.cache_key,
                    )
                    await _safe_cache_hit_callbacks(
                        proxy.hooks, ctx, cache_result.data, tokens_saved=0
                    )
                    sse_resp = _gemini_sse_from_cached_json(cache_result.data)
                    if sse_resp is not None:
                        return sse_resp

            # ── 2. Compression ──────────────────────────────────────────────
            outbound_bytes = body_bytes
            if proxy.hooks.compress:
                compress_result = await _safe_compress(proxy.hooks, ctx)
                if compress_result.body_bytes != body_bytes:
                    outbound_bytes = compress_result.body_bytes

            fwd_headers["content-length"] = str(len(outbound_bytes))

            # ── 3. Build upstream streaming URL ─────────────────────────────
            allowed = {
                k: v for k, v in request.query_params.items()
                if k.lower() in _GEMINI_ALLOWED_QUERY_PARAMS
            }
            allowed["alt"] = "sse"
            upstream_stream_url = f"{upstream_url}?{urllib.parse.urlencode(allowed)}"

            # ── 4. Stream + accumulate → cache store ─────────────────────────
            store_callback = None
            if proxy.hooks.cache:
                _hooks = proxy.hooks
                _ctx = ctx
                _tid = _tenant_id

                async def _store_from_gemini_sse(sse_bytes: bytes) -> None:
                    parsed = _parse_gemini_sse_to_json(sse_bytes)
                    if parsed is None:
                        return  # incomplete stream — skip
                    prov = ProviderResponse(
                        modified=False, body={}, body_bytes=parsed,
                        tokens_before=0, tokens_after=0, strategy="none",
                    )
                    await _safe_cache_store(_hooks, _ctx, prov, tenant_id=_tid)

                store_callback = _store_from_gemini_sse

            return await _stream_and_cache_forward(
                proxy, request_id, fwd_headers, outbound_bytes, upstream_stream_url,
                on_complete=store_callback,
            )

        # ── Non-streaming path ──────────────────────────────────────────────
        cache_result = None
        if proxy.hooks.cache:
            cache_result = await _safe_cache_check(proxy.hooks, ctx, tenant_id=_tenant_id)
            if cache_result.hit and cache_result.data:
                await _safe_cache_hit_callbacks(
                    proxy.hooks, ctx, cache_result.data, tokens_saved=0
                )
                return Response(
                    content=cache_result.data,
                    status_code=200,
                    media_type="application/json",
                )

        outbound_bytes = body_bytes
        if proxy.hooks.compress:
            compress_result = await _safe_compress(proxy.hooks, ctx)
            if compress_result.body_bytes != body_bytes:
                outbound_bytes = compress_result.body_bytes

        fwd_headers["content-length"] = str(len(outbound_bytes))

        upstream_resp = await proxy.http_client.post(
            upstream_url, content=outbound_bytes, headers=fwd_headers,
        )
        resp_bytes = upstream_resp.content

        if (
            upstream_resp.status_code == 200
            and proxy.hooks.cache
            and cache_result is not None
            and cache_result.cache_key
        ):
            prov_resp = ProviderResponse(
                modified=False, body={}, body_bytes=resp_bytes,
                tokens_before=0, tokens_after=0, strategy="none",
            )
            await _safe_cache_store(proxy.hooks, ctx, prov_resp, tenant_id=_tenant_id)

        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )

    except Exception as exc:
        logger.error("[%s] handle_gemini_native exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)


async def handle_gemini_openai_compat(proxy: object, request: Request) -> Response:
    """Handle /v1beta/openai/chat/completions and /v1beta/openai/models.

    BUG-FIX (v3.6.4): POST requests (chat completions) now go through the
    cache + compress pipeline.  The /v1beta/openai surface accepts standard
    OpenAI JSON, so we reuse OpenAI format parsing from openai_surface.py
    for streaming SSE detection and replay.

    GET requests (models list) are pass-through — no caching needed.
    """
    local_path = request.url.path
    upstream_url = f"{_GEMINI_UPSTREAM_BASE}{local_path}"
    request_id = await proxy.next_request_id()

    try:
        body_bytes = await request.body()

        # GET (models) — pass-through
        if request.method == "GET" or not body_bytes:
            fwd_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() in _GEMINI_OPENAI_COMPAT_FORWARD_HEADERS
            }
            upstream_resp = await proxy.http_client.request(
                method=request.method,
                url=upstream_url,
                content=body_bytes if body_bytes else None,
                headers=fwd_headers,
            )
            return Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type="application/json",
                headers=_filter_response_headers(dict(upstream_resp.headers)),
            )

        # POST (chat completions) — cache + compress
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return await _fail_open_forward(proxy, request, upstream_url)

        has_tools = _body_has_tools(body)
        stream = bool(body.get("stream", False))

        # v3.6.10 shadow-capture (plan §7): pure passthrough + corpus record.
        if capture_enabled():
            return await capture_passthrough_forward(
                proxy, request, provider="gemini-openai-compat",
                upstream_url=upstream_url,
                allowed_headers=_GEMINI_OPENAI_COMPAT_FORWARD_HEADERS,
                request_id=request_id, model_hint=str(body.get("model", "")),
                sse_parser=_parse_openai_sse_to_json, is_stream=stream,
            )

        # SECURITY (WP-D): derive tenant BEFORE _redact_headers strips the key.
        _raw_key_compat = (
            request.headers.get("x-goog-api-key")
            or request.headers.get("authorization")
        )
        _tenant_id_compat = _derive_tenant_id("gemini-openai-compat", _raw_key_compat)

        ctx = ProxyRequest(
            provider="gemini-openai-compat",
            method="POST",
            path=local_path,
            headers=_redact_headers(dict(request.headers)),
            body=body,
            body_bytes=body_bytes,
            request_id=request_id,
            stream=stream,
            has_tools=has_tools,
        )

        # Cache check
        cache_result = None
        if proxy.hooks.cache:
            cache_result = await _safe_cache_check(proxy.hooks, ctx, tenant_id=_tenant_id_compat)
            if cache_result.hit and cache_result.data:
                await _safe_cache_hit_callbacks(
                    proxy.hooks, ctx, cache_result.data, tokens_saved=0
                )
                return Response(
                    content=cache_result.data,
                    status_code=200,
                    media_type="application/json",
                )

        # Compress
        outbound_bytes = body_bytes
        if proxy.hooks.compress:
            compress_result = await _safe_compress(proxy.hooks, ctx)
            if compress_result.body_bytes != body_bytes:
                outbound_bytes = compress_result.body_bytes

        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in _GEMINI_OPENAI_COMPAT_FORWARD_HEADERS
        }
        fwd_headers["content-length"] = str(len(outbound_bytes))

        upstream_resp = await proxy.http_client.request(
            method="POST",
            url=upstream_url,
            content=outbound_bytes,
            headers=fwd_headers,
        )
        resp_bytes = upstream_resp.content

        if (
            upstream_resp.status_code == 200
            and proxy.hooks.cache
            and cache_result is not None
            and cache_result.cache_key
        ):
            prov_resp = ProviderResponse(
                modified=False, body={}, body_bytes=resp_bytes,
                tokens_before=0, tokens_after=0, strategy="none",
            )
            await _safe_cache_store(proxy.hooks, ctx, prov_resp, tenant_id=_tenant_id_compat)

        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )

    except Exception as exc:
        logger.error(
            "[%s] handle_gemini_openai_compat exc=%r — fail-open", request_id, exc
        )
        return await _fail_open_forward(proxy, request, upstream_url)
