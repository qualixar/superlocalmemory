"""openai_surface.py — OpenAI /v1/chat/completions and /v1/embeddings.

BUG-FIXES (v3.6.3):
- Streaming path previously bypassed cache AND had _safe_compress not imported
  (NameError on any non-streaming compressed call). Same set of bugs as
  anthropic_surface.py pre-v3.6.3. Fixed with the same pattern:
  cache check → _stream_and_cache_forward → post-stream store.
- _safe_compress added to imports (was missing; NameError on compress path).
- Streaming cache: accumulate SSE, parse to JSON, store so future identical
  calls are served from cache. OpenAI SSE format differs from Anthropic's
  so a dedicated _parse_openai_sse_to_json / _openai_sse_from_cached helper
  is used.
"""

from __future__ import annotations

import json
import logging

from fastapi.requests import Request
from fastapi.responses import Response, StreamingResponse

from superlocalmemory.optimize.proxy._helpers import (
    _OPENAI_FORWARD_HEADERS,
    _body_has_tools,
    _build_forward_headers,
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
from superlocalmemory.optimize.proxy.lifecycle import ProviderResponse, ProxyRequest

logger = logging.getLogger("slm.optimize.proxy.openai")

_UPSTREAM_BASE = "https://api.openai.com"


# ---------------------------------------------------------------------------
# OpenAI SSE helpers (streaming cache hit/store)
# ---------------------------------------------------------------------------

def _parse_openai_sse_to_json(sse_bytes: bytes) -> bytes | None:
    """Parse an accumulated OpenAI SSE stream into a single chat.completion JSON.

    OpenAI streaming format:
        data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[
                 {"index":0,"delta":{"role":"assistant","content":"Hello"},...}]}
        ...
        data: [DONE]

    Assembles into the non-streaming chat.completion format so it can be stored
    in the cache and replayed as SSE on the next identical call.

    BUG-FIX (v3.6.4): Previously returned None for responses containing
    tool_calls, meaning OpenAI-compatible tool-bearing clients (Codex CLI,
    Antigravity) were NEVER cached. Now accumulates tool_calls by index and
    includes them in the stored JSON so _openai_sse_from_cached_json can
    replay them correctly.

    Returns None only for incomplete streams (no [DONE], missing id).
    """
    completion_id = ""
    model = ""
    created = 0
    text_acc: dict[int, str] = {}  # choice_index → accumulated content
    finish_reasons: dict[int, str] = {}
    # tool_calls_acc[choice_idx][tc_idx] = {id, type, function: {name, args_parts}}
    tool_calls_acc: dict[int, dict[int, dict]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    done_seen = False

    for raw_line in sse_bytes.decode("utf-8", errors="replace").split("\n"):
        line = raw_line.rstrip("\r")
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            done_seen = True
            continue
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if not completion_id:
            completion_id = chunk.get("id", "")
        if not model:
            model = chunk.get("model", "")
        if not created:
            created = chunk.get("created", 0)

        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            delta = choice.get("delta", {})

            content = delta.get("content")
            if content:
                text_acc[idx] = text_acc.get(idx, "") + content

            for tc in delta.get("tool_calls", []):
                tc_idx = tc.get("index", 0)
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {}
                if tc_idx not in tool_calls_acc[idx]:
                    tool_calls_acc[idx][tc_idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments_parts": []},
                    }
                entry = tool_calls_acc[idx][tc_idx]
                if tc.get("id"):
                    entry["id"] = tc["id"]
                if tc.get("type"):
                    entry["type"] = tc["type"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    entry["function"]["name"] = fn["name"]
                if fn.get("arguments") is not None:
                    entry["function"]["arguments_parts"].append(fn["arguments"])

            fr = choice.get("finish_reason")
            if fr:
                finish_reasons[idx] = fr

        usage = chunk.get("usage") or {}
        if usage:
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)

    if not done_seen or not completion_id:
        return None

    all_choice_indices = sorted(set(list(text_acc.keys()) + list(tool_calls_acc.keys())))
    if not all_choice_indices:
        return None

    choices = []
    for i in all_choice_indices:
        finish_reason = finish_reasons.get(i, "stop")
        message: dict = {"role": "assistant"}

        if i in tool_calls_acc:
            tool_calls = [
                {
                    "id": tool_calls_acc[i][ti]["id"],
                    "type": tool_calls_acc[i][ti].get("type", "function"),
                    "function": {
                        "name": tool_calls_acc[i][ti]["function"]["name"],
                        "arguments": "".join(
                            tool_calls_acc[i][ti]["function"]["arguments_parts"]
                        ),
                    },
                }
                for ti in sorted(tool_calls_acc[i].keys())
            ]
            message["content"] = None
            message["tool_calls"] = tool_calls
        else:
            message["content"] = text_acc.get(i, "")

        choices.append({
            "index": i,
            "message": message,
            "logprobs": None,
            "finish_reason": finish_reason,
        })

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return json.dumps(result, separators=(",", ":")).encode("utf-8")


def _openai_sse_from_cached_json(cached_bytes: bytes) -> "StreamingResponse | None":
    """Convert a stored chat.completion JSON back to an OpenAI SSE stream.

    Used when an identical streaming request hits the cache — we replay the
    stored JSON as the SSE events the OpenAI API would have emitted, preserving
    the streaming contract with the client (e.g. Codex CLI, Antigravity).

    BUG-FIX (v3.6.4): Handles tool_calls in the cached message, replaying
    them as proper OpenAI SSE delta events with index/id/name/arguments chunks.

    Returns None if the bytes are not a parseable chat.completion object.
    """
    try:
        resp = json.loads(cached_bytes)
    except (json.JSONDecodeError, ValueError):
        return None

    if resp.get("object") != "chat.completion":
        return None

    async def _generate():
        completion_id = resp.get("id", "")
        model = resp.get("model", "")
        created = resp.get("created", 0)

        for choice in resp.get("choices", []):
            idx = choice.get("index", 0)
            message = choice.get("message", {})
            tool_calls = message.get("tool_calls")
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason", "stop")

            if tool_calls:
                # First chunk: role + tool_call headers (id, type, name, empty args)
                first_delta = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": ti,
                            "id": tc["id"],
                            "type": tc.get("type", "function"),
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": "",
                            },
                        }
                        for ti, tc in enumerate(tool_calls)
                    ],
                }
                first_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": idx, "delta": first_delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(first_chunk)}\n\n".encode()

                # Argument chunks per tool call (50-char pieces)
                chunk_size = 50
                for ti, tc in enumerate(tool_calls):
                    args = tc.get("function", {}).get("arguments", "")
                    for start in range(0, max(len(args), 1), chunk_size):
                        piece = args[start: start + chunk_size]
                        arg_chunk = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{
                                "index": idx,
                                "delta": {"tool_calls": [{"index": ti, "function": {"arguments": piece}}]},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(arg_chunk)}\n\n".encode()

                # Finish chunk
                finish_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": idx, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n".encode()

            else:
                # Text response replay
                role_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": idx, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n".encode()

                chunk_size = 100
                for start in range(0, max(len(content), 1), chunk_size):
                    piece = content[start: start + chunk_size]
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": idx, "delta": {"content": piece}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n".encode()

                finish_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": idx, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(finish_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_chat_completions(proxy: object, request: Request) -> Response:
    request_id = await proxy.next_request_id()
    upstream_url = f"{_UPSTREAM_BASE}/v1/chat/completions"
    try:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            return await _fail_open_forward(proxy, request, upstream_url)

        stream = bool(body.get("stream", False))
        has_tools = _body_has_tools(body)

        # v3.6.10 shadow-capture (plan §7): pure passthrough + corpus record.
        if capture_enabled():
            return await capture_passthrough_forward(
                proxy, request, provider="openai", upstream_url=upstream_url,
                allowed_headers=_OPENAI_FORWARD_HEADERS, request_id=request_id,
                model_hint=str(body.get("model", "")),
                sse_parser=_parse_openai_sse_to_json, is_stream=stream,
            )

        ctx = ProxyRequest(
            provider="openai", method="POST", path="/v1/chat/completions",
            headers=_redact_headers(dict(request.headers)),
            body=body, body_bytes=body_bytes,
            request_id=request_id, stream=stream, has_tools=has_tools,
        )

        if stream:
            # BUG-FIX (v3.6.3): streaming path previously bypassed cache
            # entirely — same bug as anthropic_surface.py pre-fix. OpenAI
            # clients (Codex CLI, Antigravity, openai-python) use stream=True
            # by default, so savings were permanently 0.

            # 1. Cache check
            if proxy.hooks.cache:
                cache_result = await _safe_cache_check(proxy.hooks, ctx)
                if cache_result and cache_result.hit and cache_result.data:
                    logger.debug(
                        "[%s] OpenAI streaming cache HIT key=%s",
                        request_id, cache_result.cache_key,
                    )
                    await _safe_cache_hit_callbacks(
                        proxy.hooks, ctx, cache_result.data, tokens_saved=0
                    )
                    sse_resp = _openai_sse_from_cached_json(cache_result.data)
                    if sse_resp is not None:
                        return sse_resp

            # 2. Compression on request body
            outbound_bytes = body_bytes
            if proxy.hooks.compress:
                compress_result = await _safe_compress(proxy.hooks, ctx)
                if compress_result.body_bytes != body_bytes:
                    outbound_bytes = compress_result.body_bytes

            fwd_headers = _build_forward_headers(request, _OPENAI_FORWARD_HEADERS)
            fwd_headers["content-length"] = str(len(outbound_bytes))

            # 3. Stream + accumulate for cache store
            store_callback = None
            if proxy.hooks.cache:
                _hooks = proxy.hooks
                _ctx = ctx
                async def _store_from_openai_sse(sse_bytes: bytes) -> None:
                    parsed = _parse_openai_sse_to_json(sse_bytes)
                    if parsed is None:
                        return
                    prov = ProviderResponse(
                        modified=False, body={}, body_bytes=parsed,
                        tokens_before=0, tokens_after=0, strategy="none",
                    )
                    await _safe_cache_store(_hooks, _ctx, prov)
                store_callback = _store_from_openai_sse

            return await _stream_and_cache_forward(
                proxy, request_id, fwd_headers, outbound_bytes, upstream_url,
                on_complete=store_callback,
            )

        # --- Non-streaming path ---
        cache_result = None
        if proxy.hooks.cache:
            cache_result = await _safe_cache_check(proxy.hooks, ctx)
            if cache_result.hit and cache_result.data:
                await _safe_cache_hit_callbacks(
                    proxy.hooks, ctx, cache_result.data, 0
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

        fwd_headers = _build_forward_headers(request, _OPENAI_FORWARD_HEADERS)
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
            _prov_resp = ProviderResponse(
                modified=False, body={}, body_bytes=resp_bytes,
                tokens_before=0, tokens_after=0, strategy="none",
            )
            await _safe_cache_store(proxy.hooks, ctx, _prov_resp)

        return Response(
            content=resp_bytes,
            status_code=upstream_resp.status_code,
            media_type="application/json",
            headers=_filter_response_headers(dict(upstream_resp.headers)),
        )

    except Exception as exc:
        logger.error("[%s] handle_chat_completions exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)


async def handle_embeddings(proxy: object, request: Request) -> Response:
    request_id = await proxy.next_request_id()
    upstream_url = f"{_UPSTREAM_BASE}/v1/embeddings"
    try:
        body_bytes = await request.body()
        fwd_headers = _build_forward_headers(request, _OPENAI_FORWARD_HEADERS)
        fwd_headers["content-length"] = str(len(body_bytes))
        upstream_resp = await proxy.http_client.post(
            upstream_url, content=body_bytes, headers=fwd_headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )
    except Exception as exc:
        logger.error("[%s] handle_embeddings exc=%r — fail-open", request_id, exc)
        return await _fail_open_forward(proxy, request, upstream_url)
