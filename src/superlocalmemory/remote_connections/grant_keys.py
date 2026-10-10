"""Per-connection grant keys, held only in the OS secure store.

Same secure-store rule as the connector credential: if the keyring is
unavailable there is simply no grant (remote mesh and media stay refused), and
a key is never written to a file or a log. The value stored under
``grant:<connection_id>`` is JSON ``{"v":1,"version","key","previous"}``.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
from collections.abc import Callable

from superlocalmemory.remote_connections.credentials import SERVICE, SecureBackend
from superlocalmemory.remote_connections.grant import GrantKeys

PREVIOUS_KEY_WINDOW_S = 120.0
_CONNECTION = re.compile(r"^[a-f0-9]{32}$")
_KEY = re.compile(r"^[A-Za-z0-9_-]{43}$")
_MAX_VERSION = 2**53 - 1


def _decode(text: object) -> bytes | None:
    if not isinstance(text, str) or not _KEY.match(text):
        return None
    try:
        raw = base64.urlsafe_b64decode(text + "=")
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == 32 else None


def _version(value: object) -> bool:
    return type(value) is int and 1 <= value <= _MAX_VERSION


class GrantKeyStore:
    """``backend_factory`` opens the secure store lazily, so nothing touches the
    keyring before the owner has opted in to a connection."""

    def __init__(self, backend_factory: Callable[[], SecureBackend], *,
                 clock: Callable[[], float] = time.time) -> None:
        self._factory = backend_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: dict[str, dict | None] = {}

    @staticmethod
    def _name(connection_id: str) -> str:
        if not isinstance(connection_id, str) or not _CONNECTION.match(connection_id):
            raise ValueError("invalid_grant_binding")
        return "grant:" + connection_id

    def _read(self, connection_id: str) -> dict | None:
        """The stored record, or ``None``. Callers hold the lock."""
        if connection_id in self._cache:
            return self._cache[connection_id]
        try:
            raw = self._factory().get_password(SERVICE, self._name(connection_id))
            record = self._parse(raw)
        except Exception:
            return None  # unreadable now: no grant; try again next time
        self._cache[connection_id] = record
        return record

    @staticmethod
    def _parse(raw: object) -> dict | None:
        if raw is None:
            return None
        try:
            record = json.loads(raw) if isinstance(raw, str) and len(raw) <= 4096 else None
        except ValueError:
            return None
        if (not isinstance(record, dict) or record.get("v") != 1
                or not _version(record.get("version")) or _decode(record.get("key")) is None):
            return None
        previous = record.get("previous")
        if previous is not None and not (
                isinstance(previous, dict) and _version(previous.get("version"))
                and _decode(previous.get("key")) is not None
                and isinstance(previous.get("until_s"), (int, float))
                and not isinstance(previous.get("until_s"), bool)):
            record = dict(record, previous=None)
        return record

    def load(self, connection_id: str) -> GrantKeys:
        with self._lock:
            record = self._read(connection_id)
        if record is None:
            return GrantKeys(None, None)
        previous = record.get("previous")
        old = None
        if previous is not None and previous["until_s"] > self._clock():
            old = (previous["version"], _decode(previous["key"]), float(previous["until_s"]))
        return GrantKeys((record["version"], _decode(record["key"])), old)

    def store_new(self, connection_id: str, version: int, key_b64url: str) -> None:
        """Make ``version`` current; the one it replaces stays valid for two minutes."""
        name = self._name(connection_id)
        if not _version(version) or _decode(key_b64url) is None:
            raise ValueError("invalid_grant_key")
        with self._lock:
            held = self._read(connection_id)
            if held is not None and held["version"] == version and held["key"] == key_b64url:
                return
            if held is not None and version <= held["version"]:
                raise ValueError("stale_grant_key")
            previous = None
            if held is not None:
                previous = {"version": held["version"], "key": held["key"],
                            "until_s": self._clock() + PREVIOUS_KEY_WINDOW_S}
            record = {"v": 1, "version": version, "key": key_b64url, "previous": previous}
            self._write(connection_id, name, record)

    def _write(self, connection_id: str, name: str, record: dict) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        try:
            backend = self._factory()
            backend.set_password(SERVICE, name, encoded)
            if backend.get_password(SERVICE, name) != encoded:
                raise ValueError("verification_failed")
        except Exception:
            self._cache.pop(connection_id, None)
            raise ValueError("grant_key_store_unavailable") from None
        self._cache[connection_id] = record

    def forget(self, connection_id: str) -> None:
        """Remove the key (the connection was cancelled or revoked)."""
        name = self._name(connection_id)
        with self._lock:
            self._cache[connection_id] = None
            try:
                backend = self._factory()
                delete = getattr(backend, "delete_password", None)
                if callable(delete):
                    delete(SERVICE, name)
                else:
                    backend.set_password(SERVICE, name, json.dumps({"v": 1, "forgotten": True}))
            except Exception:
                return  # nothing more to do; the cache already says no grant


__all__ = ["GrantKeyStore", "PREVIOUS_KEY_WINDOW_S"]
