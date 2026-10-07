"""Private relay v1, matching the TypeScript codec; public MCP bytes are opaque."""

from __future__ import annotations

import base64
import binascii
import json
import re

MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_HEADERS = frozenset(
    {"content-type", "accept", "mcp-protocol-version", "mcp-method", "mcp-name"}
)
RESPONSE_HEADERS = frozenset({"content-type", "mcp-protocol-version", "retry-after"})
_COMMON = {"v", "kind", "id", "generation"}


class FrameError(ValueError):
    """Bounded error code; never includes frame contents or credentials."""


def compact(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _integer(value: object, minimum: int) -> bool:
    return type(value) is int and minimum <= value <= 2**53 - 1


def _headers(value: object, allowed: frozenset[str]) -> bool:
    if not isinstance(value, list) or len(value) > 32:
        return False
    seen = set()
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, str) for x in pair)
            or pair[0].lower() not in allowed
            or pair[0].lower() in seen
            or len(pair[1]) > 8192
            or any(not 32 <= ord(c) <= 126 for c in pair[1])
        ):
            return False
        seen.add(pair[0].lower())
    return True


def _body(value: object, cap: int) -> bool:
    if not isinstance(value, str) or len(value) % 4 or len(value) > ((cap + 2) // 3) * 4:
        return False
    try:
        body = base64.b64decode(value, validate=True)
        return len(body) <= cap and base64.b64encode(body).decode("ascii") == value
    except (ValueError, binascii.Error):
        return False


def decode_frame(value: str | bytes, *, param_headers: tuple[str, ...] = ()) -> dict:
    if (
        not isinstance(param_headers, (tuple, list))
        or len(param_headers) > 32
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"mcp-param-[a-z0-9_.-]{1,64}", name, re.I)
            for name in param_headers
        )
    ):
        raise FrameError("INVALID_CONFIGURATION")
    try:
        if not isinstance(value, (str, bytes)):
            raise FrameError("INVALID_FRAME")
        if len(value) > MAX_FRAME_BYTES:
            raise FrameError("FRAME_TOO_LARGE")
        text = value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        if len(text.encode("utf-8")) > MAX_FRAME_BYTES:
            raise FrameError("FRAME_TOO_LARGE")
        frame = json.loads(text)
    except (UnicodeError, ValueError, RecursionError) as error:
        if isinstance(error, FrameError):
            raise
        raise FrameError("INVALID_FRAME") from None
    if (
        not isinstance(frame, dict)
        or type(frame.get("v")) is not int
        or frame["v"] != 1
        or not isinstance(frame.get("id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", frame["id"])
        or not _integer(frame.get("generation"), 1)
    ):
        raise FrameError("INVALID_FRAME")
    kind = frame.get("kind")
    if not isinstance(kind, str):
        raise FrameError("INVALID_FRAME")
    if kind == "cancel":
        valid = set(frame) == _COMMON
    elif kind in {"request", "response"}:
        request = kind == "request"
        specific = "deadlineAt" if request else "status"
        allowed = (
            REQUEST_HEADERS | frozenset(name.lower() for name in param_headers)
            if request
            else RESPONSE_HEADERS
        )
        valid = (
            set(frame) == _COMMON | {specific, "headers", "bodyBase64"}
            and _integer(frame.get(specific), 0 if request else 200)
            and (request or frame[specific] <= 599)
            and _headers(frame.get("headers"), allowed)
            and _body(frame.get("bodyBase64"), MAX_REQUEST_BYTES if request else MAX_RESPONSE_BYTES)
        )
    else:
        valid = False
    if not valid:
        raise FrameError("INVALID_FRAME")
    if compact(frame) != text:
        raise FrameError("NON_CANONICAL_FRAME")
    return frame


def encode_frame(frame: dict, *, param_headers: tuple[str, ...] = ()) -> str:
    try:
        encoded = compact(frame)
    except (ValueError, TypeError, RecursionError):
        raise FrameError("INVALID_FRAME") from None
    decode_frame(encoded, param_headers=param_headers)
    return encoded
