"""Opt-in connector credentials in an OS secure store, never browser state.

No keyring is opened on import. Secure-store failure disables only remote
connectivity; it never falls back to plaintext or changes local configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from superlocalmemory.remote_connections.journal import _identity
from superlocalmemory.remote_connections.private_state import protect_directory

SERVICE = "SuperLocalMemory.RemoteConnections.v1"
_FIELDS = {
    "installation_id",
    "owner",
    "profile",
    "connection_id",
    "generation",
    "expires_at_ms",
    "device_token",
    "origin_key",
    "device_private_key",
}


class CredentialError(RuntimeError):
    """Bounded non-secret error code for remote-only handling."""


class SecureBackend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...


@dataclass(frozen=True)
class ConnectorCredential:
    installation_id: str
    owner: str
    profile: str
    connection_id: str
    generation: int
    expires_at_ms: int
    device_token: str = field(repr=False)
    origin_key: str = field(repr=False)
    device_private_key: str = field(default="", repr=False)


def _validate(value: ConnectorCredential) -> None:
    try:
        if not isinstance(value.device_private_key, str) or len(value.device_private_key) > 2048:
            raise ValueError("invalid")
        if value.device_private_key:
            from superlocalmemory.remote_connections.proof import DeviceSigner

            DeviceSigner(value.device_private_key)
        for identity in (value.installation_id, value.owner, value.profile):
            _identity(identity)
        if (
            not isinstance(value.connection_id, str)
            or not re.fullmatch(r"[a-f0-9]{32}", value.connection_id)
            or type(value.generation) is not int
            or not 1 <= value.generation < 2**53 - 1
            or type(value.expires_at_ms) is not int
            or not 0 <= value.expires_at_ms < 2**53
            or not isinstance(value.device_token, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value.device_token)
            or not isinstance(value.origin_key, str)
            or not re.fullmatch(r"slmr_[A-Za-z0-9_-]{43}", value.origin_key)
        ):
            raise ValueError("invalid")
    except (ValueError, AttributeError):
        raise CredentialError("invalid_connector_credential") from None


def _native_backend() -> SecureBackend:
    try:
        import keyring

        backend = keyring.get_keyring()
        identity = (type(backend).__module__, type(backend).__name__)
        if identity not in {
            ("keyring.backends.macOS", "Keyring"),
            ("keyring.backends.Windows", "WinVaultKeyring"),
            ("keyring.backends.SecretService", "Keyring"),
        }:
            raise ValueError("unsupported_backend")
        return backend
    except Exception:
        raise CredentialError("secure_keyring_unavailable") from None


class CredentialVault:
    """Cross-process fenced keychain writes. Injected backends are test adapters.

    The journal authenticates enrollment before delivery. This vault independently
    enforces installation/owner/profile identity, monotonic generation, expiry
    and terminal revocation. Its private lock contains no credential material.
    """

    def __init__(
        self,
        root: Path,
        *,
        backend: SecureBackend | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.root = Path(root)
        if self.root.is_symlink():
            raise CredentialError("unsafe_credential_path")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        protect_directory(self.root)
        self._backend = backend if backend is not None else _native_backend()
        self._clock = clock

    @staticmethod
    def _key(installation: str, owner: str, profile: str, connection: str) -> str:
        for value in (installation, owner, profile):
            try:
                _identity(value)
            except ValueError:
                raise CredentialError("invalid_credential_binding") from None
        if not isinstance(connection, str) or not re.fullmatch(r"[a-f0-9]{32}", connection):
            raise CredentialError("invalid_credential_binding")
        return hashlib.sha256(
            json.dumps([installation, owner, profile, connection]).encode()
        ).hexdigest()

    @contextmanager
    def _locked(self):
        from superlocalmemory.core.file_lock import exclusive_lock

        lock = self.root / "credential.lock"
        if lock.is_symlink():
            raise CredentialError("unsafe_credential_path")
        with exclusive_lock(lock):
            yield

    def _get(self, key: str) -> dict | None:
        try:
            raw = self._backend.get_password(SERVICE, key)
        except Exception:
            raise CredentialError("credential_store_unavailable") from None
        if raw is None:
            return None
        try:
            if not isinstance(raw, str) or len(raw) > 8192:
                raise ValueError("invalid")
            record = json.loads(raw)
            if not isinstance(record, dict) or record.get("version") != 1:
                raise ValueError("invalid")
            if record.get("revoked") is True and set(record) == {"version", "revoked"}:
                return record
            if set(record) == (_FIELDS - {"device_private_key"}) | {"version"}:
                record["device_private_key"] = ""
            if set(record) != _FIELDS | {"version"}:
                raise ValueError("invalid")
            credential = ConnectorCredential(**{field: record[field] for field in _FIELDS})
            _validate(credential)
            if (
                self._key(
                    credential.installation_id,
                    credential.owner,
                    credential.profile,
                    credential.connection_id,
                )
                != key
            ):
                raise ValueError("binding")
            return record
        except (ValueError, TypeError, CredentialError):
            raise CredentialError("invalid_stored_credential") from None

    def _put(self, key: str, record: dict) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        try:
            self._backend.set_password(SERVICE, key, encoded)
            if self._backend.get_password(SERVICE, key) != encoded:
                raise ValueError("verification_failed")
        except Exception:
            raise CredentialError("credential_store_unavailable") from None

    def save(self, credential: ConnectorCredential) -> None:
        _validate(credential)
        if credential.expires_at_ms <= self._clock() * 1000:
            raise CredentialError("credential_expired")
        key = self._key(
            credential.installation_id,
            credential.owner,
            credential.profile,
            credential.connection_id,
        )
        record = {"version": 1, **asdict(credential)}
        with self._locked():
            old = self._get(key)
            if old and old.get("revoked") is True:
                raise CredentialError("credential_revoked")
            if old and old["generation"] > credential.generation:
                raise CredentialError("credential_stale")
            if old and old["generation"] == credential.generation and old != record:
                raise CredentialError("credential_conflict")
            self._put(key, record)

    def load(
        self, installation: str, owner: str, profile: str, connection: str
    ) -> ConnectorCredential | None:
        key = self._key(installation, owner, profile, connection)
        with self._locked():
            record = self._get(key)
        if (
            record is None
            or record.get("revoked") is True
            or record["expires_at_ms"] <= self._clock() * 1000
        ):
            return None
        return ConnectorCredential(**{field: record[field] for field in _FIELDS})

    def revoke(self, installation: str, owner: str, profile: str, connection: str) -> None:
        key = self._key(installation, owner, profile, connection)
        with self._locked():
            # Retain only a terminal tombstone so a stale delivery cannot revive
            # this connection. Actual credential values are overwritten.
            self._put(key, {"version": 1, "revoked": True})
