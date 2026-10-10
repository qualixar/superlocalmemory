"""The signed per-connection grant carried on a relayed request.

The gateway signs, for each forwarded frame, which web app is calling and what
that app was allowed to do. This laptop verifies the signature before anything
downstream may rely on it. The wire format is shared byte for byte with the
TypeScript signer: ``v1.`` + base64url(payload) + ``.`` + base64url(mac), with
``mac = HMAC-SHA256(key, "slm-grant-v1." + base64url(payload))``.

Stdlib only. An error carries a bounded code and never the header value.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass

GRANT_HEADER = "x-slm-grant"
SCOPE_ORDER = ("slm:read", "slm:write", "slm:session", "slm:mesh", "slm:media")
CLOCK_SKEW_TOLERANCE_MS = 5000
MAX_SEEN_FRAMES = 4096
_SAFE_INT_MAX = 2**53 - 1
_SHAPE = re.compile(r"v1\.([A-Za-z0-9_-]{1,8000})\.([A-Za-z0-9_-]{43})")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,256}")
_KEYS = ("v", "kid", "cid", "aid", "ver", "app", "scp", "fv", "fid", "gen", "dl")


class GrantError(ValueError):
    """``args[0]`` is one of: malformed, unknown_kid, bad_mac, mismatch, expired, replay."""


@dataclass(frozen=True)
class RemoteGrant:
    connection_id: str
    authorization_id: str
    authorization_version: int
    app: str
    scopes: frozenset[str]
    folders_visible: bool
    key_version: int


@dataclass(frozen=True)
class GrantKeys:
    current: tuple[int, bytes] | None = None
    #: (version, key, valid_until_s): accepted only until that wall-clock second.
    previous: tuple[int, bytes, float] | None = None

    def __repr__(self) -> str:  # never print key material
        cur = self.current[0] if self.current else None
        prev = self.previous[0] if self.previous else None
        return f"GrantKeys(current={cur}, previous={prev})"


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError):
        raise GrantError("malformed") from None
    if _b64e(raw) != text:
        raise GrantError("malformed")
    return raw


def _mac(key: bytes, segment: str) -> bytes:
    return hmac.new(key, ("slm-grant-v1." + segment).encode("ascii"), hashlib.sha256).digest()


def sign_grant(
    key: bytes, *, connection_id: str, authorization_id: str, authorization_version: int,
    app: str, scopes: Iterable[str], folders_visible: bool, frame_id: str, generation: int,
    deadline_at_ms: int, key_version: int,
) -> str:
    """Produce a grant exactly as the gateway does (used by tests)."""
    chosen = set(scopes)
    payload = {
        "v": 1, "kid": key_version, "cid": connection_id, "aid": authorization_id,
        "ver": authorization_version, "app": app,
        "scp": [name for name in SCOPE_ORDER if name in chosen], "fv": folders_visible,
        "fid": frame_id, "gen": generation, "dl": deadline_at_ms,
    }
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    segment = _b64e(text.encode("utf-8"))
    return f"v1.{segment}.{_b64e(_mac(key, segment))}"


def _int(value: object, minimum: int) -> bool:
    return type(value) is int and minimum <= value <= _SAFE_INT_MAX


def _pairs(items: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for name, value in items:
        if name in result:
            raise GrantError("malformed")
        result[name] = value
    return result


def _parse(segment: str) -> dict:
    try:
        claims = json.loads(_b64d(segment).decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise GrantError("malformed") from None
    if not isinstance(claims, dict) or tuple(sorted(claims)) != tuple(sorted(_KEYS)):
        raise GrantError("malformed")
    scopes, app = claims["scp"], claims["app"]
    valid = (
        claims["v"] == 1 and type(claims["v"]) is int
        and _int(claims["kid"], 1) and _int(claims["ver"], 1)
        and _int(claims["gen"], 1) and _int(claims["dl"], 0)
        and all(isinstance(claims[k], str) and _ID.fullmatch(claims[k]) for k in ("cid", "aid", "fid"))
        and isinstance(app, str) and 1 <= len(app) <= 2048
        and not any(ord(ch) < 32 or ord(ch) == 127 for ch in app)
        and isinstance(scopes, list) and len(set(scopes)) == len(scopes)
        and all(isinstance(s, str) and s in SCOPE_ORDER for s in scopes)
        and "slm:read" in scopes and type(claims["fv"]) is bool
    )
    if not valid:
        raise GrantError("malformed")
    return claims


def _key_for(keys: GrantKeys, kid: int, now_ms: float) -> bytes:
    if keys.current is not None and keys.current[0] == kid:
        return keys.current[1]
    if keys.previous is not None and keys.previous[0] == kid and now_ms / 1000 <= keys.previous[2]:
        return keys.previous[1]
    raise GrantError("unknown_kid")


class ReplayGuard:
    """Frame ids already used, in memory, pruned once their grant has expired."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, int] = {}

    def check_and_add(self, frame_id: str, expires_at_ms: int, now_ms: float) -> bool:
        with self._lock:
            if self._seen:
                for old in [f for f, until in self._seen.items() if until < now_ms]:
                    del self._seen[old]
            if frame_id in self._seen or len(self._seen) >= MAX_SEEN_FRAMES:
                return False
            self._seen[frame_id] = expires_at_ms
            return True


def verify_grant(
    value: str, *, keys: GrantKeys, connection_id: str, frame_id: str, generation: int,
    deadline_at_ms: int, now_ms: float, seen: ReplayGuard,
) -> RemoteGrant:
    shape = _SHAPE.fullmatch(value) if isinstance(value, str) else None
    if shape is None:
        raise GrantError("malformed")
    segment, mac = shape.groups()
    claims = _parse(segment)
    key = _key_for(keys, claims["kid"], now_ms)
    if not hmac.compare_digest(_mac(key, segment), _b64d(mac)):
        raise GrantError("bad_mac")
    if (claims["cid"], claims["fid"], claims["gen"], claims["dl"]) != (
            connection_id, frame_id, generation, deadline_at_ms):
        raise GrantError("mismatch")
    expiry = claims["dl"] + CLOCK_SKEW_TOLERANCE_MS
    if now_ms > expiry:
        raise GrantError("expired")
    if not seen.check_and_add(frame_id, expiry, now_ms):
        raise GrantError("replay")
    return RemoteGrant(
        connection_id=claims["cid"], authorization_id=claims["aid"],
        authorization_version=claims["ver"], app=claims["app"],
        scopes=frozenset(claims["scp"]), folders_visible=claims["fv"],
        key_version=claims["kid"])


def peer_ref(connection_id: str, authorization_id: str) -> str:
    """The stable peer id of a web app: ``w_`` + 24 hex of SHA-256(cid:aid)."""
    digest = hashlib.sha256(f"{connection_id}:{authorization_id}".encode()).hexdigest()
    return "w_" + digest[:24]


__all__ = [
    "CLOCK_SKEW_TOLERANCE_MS", "GRANT_HEADER", "GrantError", "GrantKeys", "RemoteGrant",
    "ReplayGuard", "SCOPE_ORDER", "peer_ref", "sign_grant", "verify_grant",
]
