# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The signing key of D-16: one random key, kept only in the operating system's keychain.

What this covers, and what it does not (D-16, stated once and repeated in the docs):

* It detects a change made by something that has the *file* but not the user's *login
  session*: a synced or restored copy, another operating-system user, a disk image. Such
  a party cannot read the keychain entry, so it cannot make a signature that checks.
* It does **not** stop a process running as the user: that process can ask the keychain
  for the key like we do.
* A lost key loses no data. Signatures made under it can no longer be checked and are
  reported as "signed under another key"; nothing is deleted and nothing is re-signed
  unless the person says so (``slm signing resign``).

The store is the one the connector credentials already use
(``remote_connections/credentials._native_backend``): macOS Keychain, the Windows
Credential Manager (DPAPI-protected) and the Secret Service (libsecret). Where none of
them is available, signing is **unavailable**. There is no fallback key file, ever.

The key is read **once**, at daemon start, with a timeout (:func:`start_for_daemon`), and
held in memory. A hook, the recall path and a CLI read command never ask the keychain;
the two commands a person runs on purpose (``slm doctor --truth`` to verify and
``slm signing resign``) read it themselves, once.

Nothing here logs or returns a key. ``key_id`` is a digest of the key, safe to store.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

SERVICE = "SuperLocalMemory.FactSigning.v1"
ACCOUNT = "signing-key"
KEY_BYTES = 32
#: Bound on one keychain read. A locked or hung keychain must never hold the daemon's start.
READ_TIMEOUT_S = 5.0

READY = "ready"
UNAVAILABLE_STORE = "no_secure_store"
UNAVAILABLE_TIMEOUT = "timeout"
UNAVAILABLE_UNREADABLE = "unreadable"
NO_KEY = "no_key"
OFF = "off"

_WORDS = {
    READY: "Signing is ready: the key is in your keychain.",
    UNAVAILABLE_STORE: ("Signing is unavailable on this computer: no supported keychain "
                        "(macOS Keychain, Windows Credential Manager, or libsecret). "
                        "Nothing is signed and no key file is written."),
    UNAVAILABLE_TIMEOUT: "Signing is unavailable right now: the keychain did not answer in time.",
    UNAVAILABLE_UNREADABLE: ("Signing is unavailable: the keychain entry could not be read "
                             "or is not a valid key."),
    NO_KEY: "There is no signing key yet (the daemon creates one at its next start).",
    OFF: "Signing is off (feature flag trust.signing).",
}


def words(status: str) -> str:
    """The status in a sentence for a person."""
    return _WORDS.get(status, status)


def key_id_of(key: bytes) -> str:
    """A digest of the key that names it without revealing it (16 hex digits)."""
    return hashlib.sha256(b"slm-signing-key-id-v1" + key).hexdigest()[:16]


@dataclass(frozen=True)
class Signer:
    """A key in memory. ``repr`` never shows it."""

    key: bytes = field(repr=False)
    key_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, bytes | bytearray) or len(self.key) != KEY_BYTES:
            raise ValueError("a signing key is exactly 32 bytes")
        if not self.key_id:
            object.__setattr__(self, "key_id", key_id_of(bytes(self.key)))

    def mac(self, data: bytes) -> str:
        return hmac.new(bytes(self.key), data, hashlib.sha256).hexdigest()

    def matches(self, data: bytes, mac: str) -> bool:
        return hmac.compare_digest(self.mac(data), str(mac))


class SecureBackend(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...


class StoreUnavailable(RuntimeError):
    """There is no keychain to hold a key. The message is a status code, never a secret."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class KeychainKeyStore:
    """The key in the OS keychain. ``backend`` is injected by tests; the real one is opened lazily."""

    def __init__(self, backend: SecureBackend | None = None) -> None:
        self._backend = backend

    def _store(self) -> SecureBackend:
        if self._backend is None:
            from superlocalmemory.remote_connections.credentials import (
                CredentialError,
                _native_backend,
            )

            try:
                self._backend = _native_backend()
            except CredentialError:
                raise StoreUnavailable(UNAVAILABLE_STORE) from None
        return self._backend

    def read(self) -> bytes | None:
        """The stored key, or ``None`` when there is none. Raises on an unreadable entry."""
        try:
            raw = self._store().get_password(SERVICE, ACCOUNT)
        except StoreUnavailable:
            raise
        except Exception:  # noqa: BLE001 - the keychain's own error text is not ours to log
            raise StoreUnavailable(UNAVAILABLE_UNREADABLE) from None
        if raw is None:
            return None
        try:
            key = base64.urlsafe_b64decode(str(raw) + "=" * (-len(str(raw)) % 4))
        except (binascii.Error, ValueError):
            raise StoreUnavailable(UNAVAILABLE_UNREADABLE) from None
        if len(key) != KEY_BYTES:
            raise StoreUnavailable(UNAVAILABLE_UNREADABLE)
        return key

    def create(self) -> bytes:
        """Make a new random key, store it, read it back. Never overwrites an existing key."""
        if self.read() is not None:
            raise StoreUnavailable(UNAVAILABLE_UNREADABLE)
        key = secrets.token_bytes(KEY_BYTES)
        text = base64.urlsafe_b64encode(key).decode().rstrip("=")
        try:
            backend = self._store()
            backend.set_password(SERVICE, ACCOUNT, text)
            if self.read() != key:
                raise ValueError("read-back differs")
        except StoreUnavailable:
            raise
        except Exception:  # noqa: BLE001
            raise StoreUnavailable(UNAVAILABLE_UNREADABLE) from None
        return key


@dataclass(frozen=True)
class Loaded:
    signer: Signer | None
    status: str

    @property
    def ready(self) -> bool:
        return self.signer is not None


def load(store: Any = None, *, create: bool = False, timeout: float = READ_TIMEOUT_S) -> Loaded:
    """Read the key once, bounded by ``timeout``; with ``create`` make one if there is none.

    Never raises. A keychain that hangs (a locked login keychain waiting for a prompt)
    answers ``timeout`` and the caller carries on without signing.
    """
    key_store = store if store is not None else KeychainKeyStore()
    out: dict[str, Loaded] = {}

    def work() -> None:
        try:
            key = key_store.read()
            if key is None and create:
                key = key_store.create()
            out["r"] = (Loaded(Signer(key), READY) if key is not None
                        else Loaded(None, NO_KEY))
        except StoreUnavailable as exc:
            out["r"] = Loaded(None, exc.status)
        except Exception:  # noqa: BLE001 - signing is evidence, never a reason to fail a start
            out["r"] = Loaded(None, UNAVAILABLE_UNREADABLE)

    thread = threading.Thread(target=work, name="slm-signing-key", daemon=True)
    thread.start()
    thread.join(timeout)
    return out.get("r", Loaded(None, UNAVAILABLE_TIMEOUT))


# -- what the daemon holds ------------------------------------------------------------------------

_lock = threading.Lock()
_held: Signer | None = None
_status = OFF


def install(signer: Signer | None, status: str | None = None) -> None:
    """Hold ``signer`` for this process (the daemon, at start). ``None`` clears it."""
    global _held, _status
    with _lock:
        _held = signer
        _status = status or (READY if signer is not None else OFF)


def current() -> Signer | None:
    """The key this process holds, or ``None``. Memory only: never touches the keychain."""
    return _held


def status() -> str:
    return _status


def start_for_daemon(slm_home: Path | None = None, *, store: Any = None,
                     timeout: float = READ_TIMEOUT_S,
                     flags: Callable[[Path | None], Any] | None = None) -> str:
    """The daemon's one read of the key. Returns the status. Never raises.

    Only while ``trust.signing`` is on or in shadow and the kill switch is not thrown;
    otherwise the keychain is not asked at all.
    """
    try:
        from superlocalmemory.core.feature_flags import FeatureFlags
        from superlocalmemory.infra.data_root import canonical_data_root

        home = slm_home if slm_home is not None else canonical_data_root()
        snapshot = flags(home) if flags is not None else FeatureFlags(home)
        if snapshot.killed or not snapshot.observing("trust.signing"):
            install(None, OFF)
            return OFF
        result = load(store, create=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        install(None, UNAVAILABLE_UNREADABLE)
        return UNAVAILABLE_UNREADABLE
    install(result.signer, result.status)
    if not result.ready:
        logger.warning("fact signing unavailable (%s)", result.status)
    return result.status


__all__ = ["ACCOUNT", "KEY_BYTES", "KeychainKeyStore", "Loaded", "NO_KEY", "OFF", "READY",
           "READ_TIMEOUT_S", "SERVICE", "Signer", "StoreUnavailable", "UNAVAILABLE_STORE",
           "UNAVAILABLE_TIMEOUT", "UNAVAILABLE_UNREADABLE", "current", "install", "key_id_of",
           "load", "start_for_daemon", "status", "words"]
