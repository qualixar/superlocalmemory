"""Optional native OAuth state. Secrets are stored only through OS keyring."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from superlocalmemory.remote_connections.credentials import SecureBackend, _native_backend
from superlocalmemory.remote_connections.journal import _identity
from superlocalmemory.remote_connections.private_state import protect_directory
from superlocalmemory.remote_connections.proof import DeviceSigner

SERVICE = "SuperLocalMemory.NativeOAuth.v1"


@dataclass(frozen=True)
class PendingEnrollment:
    installation_id: str
    owner: str
    profile: str
    connection_id: str
    redirect_uri: str
    state: str = field(repr=False)
    verifier: str = field(repr=False)
    private_key: str = field(repr=False)
    intent_json: str
    expires_at_ms: int
    client_id: str = ""
    access_token: str = field(default="", repr=False)
    refresh_token: str = field(default="", repr=False)
    access_expires_ms: int = 0
    origin_key: str = field(default="", repr=False)
    completed: bool = False


class NativeEnrollmentStore:
    def __init__(
        self,
        root: Path,
        *,
        backend: SecureBackend | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("unsafe_enrollment_path")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        protect_directory(self.root)
        self._backend, self._clock = backend, clock

    @property
    def backend(self) -> SecureBackend:
        if self._backend is None:
            self._backend = _native_backend()
        return self._backend

    @staticmethod
    def _connection_key(connection: str) -> str:
        if not isinstance(connection, str) or not re.fullmatch(r"[a-f0-9]{32}", connection):
            raise ValueError("invalid_enrollment_identity")
        return "connection:" + connection

    @staticmethod
    def _state_key(state: str) -> str:
        if not isinstance(state, str) or not re.fullmatch(r"[-A-Za-z0-9_]{43,128}", state):
            raise ValueError("invalid_enrollment_state")
        return "state:" + hashlib.sha256(state.encode()).hexdigest()

    def _read(self, key: str) -> dict | None:
        try:
            raw = self.backend.get_password(SERVICE, key)
            if raw is None:
                return None
            if len(raw) > 16384:
                raise ValueError()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except Exception:
            raise ValueError("enrollment_store_unavailable") from None

    def _write(self, key: str, value: dict) -> None:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True)
        if len(raw) > 16384:
            raise ValueError("invalid_enrollment_state")
        try:
            self.backend.set_password(SERVICE, key, raw)
            if self.backend.get_password(SERVICE, key) != raw:
                raise ValueError()
        except Exception:
            raise ValueError("enrollment_store_unavailable") from None

    def _validate(self, row: PendingEnrollment, *, allow_expired: bool = False) -> None:
        for value in (row.installation_id, row.owner, row.profile):
            _identity(value)
        self._connection_key(row.connection_id)
        self._state_key(row.state)
        if not re.fullmatch(r"[-A-Za-z0-9_.~]{43,128}", row.verifier):
            raise ValueError("invalid_enrollment_state")
        DeviceSigner(row.private_key)
        uri = urlsplit(row.redirect_uri)
        if (
            uri.scheme != "http"
            or uri.hostname != "127.0.0.1"
            or uri.path != "/api/v3/connections/callback"
            or uri.query
            or uri.fragment
            or uri.username
            or uri.password
            or not uri.port
        ):
            raise ValueError("invalid_callback_uri")
        if (
            type(row.expires_at_ms) is not int
            or (not allow_expired and row.expires_at_ms <= self._clock() * 1000)
            or type(row.completed) is not bool
            or not isinstance(row.intent_json, str)
            or len(row.intent_json) > 4096
            or len(row.client_id) > 2048
        ):
            raise ValueError("invalid_enrollment_state")
        for secret in (row.access_token, row.refresh_token, row.origin_key):
            if not isinstance(secret, str) or len(secret) > 8192:
                raise ValueError("invalid_enrollment_state")

    def _lock(self):
        from superlocalmemory.core.file_lock import exclusive_lock

        path = self.root / "native-oauth.lock"
        if path.is_symlink():
            raise ValueError("unsafe_enrollment_path")
        return exclusive_lock(path)

    @staticmethod
    def _profile_key(installation: str, owner: str, profile: str) -> str:
        for identity in (installation, owner, profile):
            _identity(identity)
        return (
            "desktop:"
            + hashlib.sha256(json.dumps([installation, owner, profile]).encode()).hexdigest()
        )

    def profile_client(self, installation: str, owner: str, profile: str) -> dict | None:
        value = self._read(self._profile_key(installation, owner, profile))
        if value is None:
            return None
        if (
            set(value) != {"client_id", "private_key", "redirect_uri"}
            or not isinstance(value["client_id"], str)
            or not value["client_id"]
        ):
            raise ValueError("invalid_desktop_binding")
        DeviceSigner(value["private_key"])
        return value

    def bind_profile_client(self, row: PendingEnrollment) -> None:
        key = self._profile_key(row.installation_id, row.owner, row.profile)
        value = {
            "client_id": row.client_id,
            "private_key": row.private_key,
            "redirect_uri": row.redirect_uri,
        }
        with self._lock():
            previous = self._read(key)
            if previous and previous != value:
                raise ValueError("desktop_binding_conflict")
            self._write(key, value)

    def save(self, row: PendingEnrollment) -> None:
        self._validate(row, allow_expired=row.completed)
        key = self._connection_key(row.connection_id)
        with self._lock():
            previous = self._read(key)
            if previous and previous.get("cancelled"):
                raise ValueError("enrollment_cancelled")
            if previous:
                old = PendingEnrollment(**previous["value"])
                if any(
                    getattr(old, name) != getattr(row, name)
                    for name in (
                        "installation_id",
                        "owner",
                        "profile",
                        "state",
                        "verifier",
                        "private_key",
                        "intent_json",
                        "redirect_uri",
                    )
                ):
                    raise ValueError("enrollment_binding_conflict")
                if old.completed and not row.completed:
                    raise ValueError("enrollment_completed")
            self._write(key, {"value": asdict(row)})
            self._write(self._state_key(row.state), {"connection": row.connection_id})

    def by_connection(
        self, connection: str, *, for_cleanup: bool = False
    ) -> PendingEnrollment | None:
        value = self._read(self._connection_key(connection))
        if not value or value.get("cancelled"):
            return None
        try:
            row = PendingEnrollment(**value["value"])
            self._validate(row, allow_expired=row.completed or for_cleanup)
            return row
        except (ValueError, TypeError, KeyError):
            return None

    def by_state(self, state: str) -> PendingEnrollment | None:
        pointer = self._read(self._state_key(state))
        if not pointer or not isinstance(pointer.get("connection"), str):
            return None
        row = self.by_connection(pointer["connection"])
        return row if row and row.state == state and not row.completed else None

    def complete(self, row: PendingEnrollment) -> None:
        self.save(replace(row, completed=True))

    def cancel(self, connection: str) -> None:
        with self._lock():
            self._write(self._connection_key(connection), {"cancelled": True})
