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
``slm signing resign``) read it themselves, once. A keychain answer that comes after the
timeout (a person who clicks "Allow" a minute late) is collected by the signing maintenance
step through :func:`retry_late_key`, with a growing wait, and by nothing else.

The operating-system keychain is never opened where it cannot work without a dialog:
in a test run (``SLM_TEST_ISOLATION=1``) and, on macOS, under a HOME with no login keychain
(:func:`keychain_blocked`). Signing is then "unavailable", in plain words, with no OS call.

Nothing here logs or returns a key. ``key_id`` is a digest of the key, safe to store.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import secrets
import sys
import threading
import time
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
UNAVAILABLE_TEST_RUN = "test_run"
UNAVAILABLE_NO_LOGIN_KEYCHAIN = "no_login_keychain"
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
    UNAVAILABLE_TEST_RUN: ("Signing is unavailable: this is a test run, and a test run never asks "
                           "the operating system's keychain."),
    UNAVAILABLE_NO_LOGIN_KEYCHAIN: ("Signing is unavailable: this account has no login keychain, so "
                                    "the operating system could only answer with a dialog. Nothing "
                                    "is signed and no key file is written."),
    NO_KEY: "There is no signing key yet (the daemon creates one at its next start).",
    OFF: "Signing is off (feature flag trust.signing).",
}


def keychain_blocked() -> str | None:
    """Why the operating system's keychain must not be asked here, or ``None`` when it may be.

    F-02 (19 Nov 2026): a process with a made-up ``HOME`` has no login keychain, and on macOS
    asking anyway shows the person at the computer a "keychain not found" dialog. So the real
    keychain is never opened (a) in a test run (``SLM_TEST_ISOLATION=1``) and (b) on macOS when the
    effective HOME holds neither ``login.keychain-db`` nor ``login.keychain``. No OS call is made
    to decide this: it looks at two file names.
    """
    if os.environ.get("SLM_TEST_ISOLATION") == "1":
        return UNAVAILABLE_TEST_RUN
    if sys.platform == "darwin":
        try:
            folder = Path.home() / "Library" / "Keychains"
            has_login = (folder / "login.keychain-db").exists() or (folder / "login.keychain").exists()
        except (OSError, RuntimeError):
            has_login = False
        if not has_login:
            return UNAVAILABLE_NO_LOGIN_KEYCHAIN
    return None


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
            blocked = keychain_blocked()
            if blocked is not None:
                raise StoreUnavailable(blocked)
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


class _Read:
    """One keychain read in its own thread. It may outlive the wait for it."""

    def __init__(self, key_store: Any, create: bool) -> None:
        self.out: dict[str, Loaded] = {}
        self._key_store = key_store
        self._create = create
        self.thread = threading.Thread(target=self._work, name="slm-signing-key", daemon=True)

    def _work(self) -> None:
        try:
            key = self._key_store.read()
            if key is None and self._create:
                key = self._key_store.create()
            self.out["r"] = (Loaded(Signer(key), READY) if key is not None
                             else Loaded(None, NO_KEY))
        except StoreUnavailable as exc:
            self.out["r"] = Loaded(None, exc.status)
        except Exception:  # noqa: BLE001 - signing is evidence, never a reason to fail a start
            self.out["r"] = Loaded(None, UNAVAILABLE_UNREADABLE)

    def wait(self, timeout: float) -> Loaded:
        self.thread.start()
        self.thread.join(timeout)
        return self.out.get("r", Loaded(None, UNAVAILABLE_TIMEOUT))


def load(store: Any = None, *, create: bool = False, timeout: float = READ_TIMEOUT_S) -> Loaded:
    """Read the key once, bounded by ``timeout``; with ``create`` make one if there is none.

    Never raises. A keychain that hangs (a locked login keychain waiting for a prompt)
    answers ``timeout`` and the caller carries on without signing.
    """
    key_store = store if store is not None else KeychainKeyStore()
    return _Read(key_store, create).wait(timeout)


# -- what the daemon holds ------------------------------------------------------------------------

_lock = threading.Lock()
_held: Signer | None = None
_status = OFF

#: F-07: the daemon's start read that did not answer in time, kept so a late answer is collected.
_pending: _Read | None = None
_next_try = 0.0
_attempts = 0
#: The wait before the maintenance step looks again at a read that has not answered (seconds):
#: it grows, so a person who is away is not nagged and a hung keychain is not called twice.
RETRY_WAITS_S = (30.0, 120.0, 600.0, 1800.0)
RETRY_MAX_WAIT_S = RETRY_WAITS_S[-1]
_clock: Callable[[], float] = time.monotonic


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


def _arm_retry(read: _Read | None, clock: Callable[[], float]) -> None:
    global _pending, _next_try, _attempts
    with _lock:
        _pending, _attempts = read, 0
        _next_try = clock() + RETRY_WAITS_S[0] if read is not None else 0.0


def retry_late_key(*, store: Any = None, timeout: float = READ_TIMEOUT_S,
                   clock: Callable[[], float] | None = None) -> Signer | None:
    """Pick up a keychain answer that came after the start's bound (F-07). Never raises.

    Called by the signing maintenance step and by nothing else: a hook, a recall and a CLI read
    command never ask the keychain. It acts only while the status is ``timeout``, and only once
    the wait since the last look (30 s, 2 min, 10 min, then every 30 min) has passed. The read
    that timed out is still in its own thread, so its late answer is collected without a second
    keychain call; a new, bounded read is made only if there is no such thread. An answer that is
    a refusal or an unreadable entry ends the retries for this run.
    """
    global _pending, _next_try, _attempts
    now = clock if clock is not None else _clock
    with _lock:
        if _held is not None or _status != UNAVAILABLE_TIMEOUT or now() < _next_try:
            return _held
        pending, attempts = _pending, _attempts
    try:
        if pending is not None and pending.thread.is_alive():
            result = Loaded(None, UNAVAILABLE_TIMEOUT)  # still waiting for the person
        elif pending is not None and "r" in pending.out:
            result = pending.out["r"]
        else:
            key_store = store if store is not None else KeychainKeyStore()
            pending = _Read(key_store, True)
            result = pending.wait(timeout)
    except Exception:  # noqa: BLE001 - a retry never breaks the maintenance cycle
        result = Loaded(None, UNAVAILABLE_UNREADABLE)
    if result.status == UNAVAILABLE_TIMEOUT:
        with _lock:
            _pending = pending
            _attempts = attempts + 1
            _next_try = now() + RETRY_WAITS_S[min(_attempts, len(RETRY_WAITS_S) - 1)]
        return None
    install(result.signer, result.status)
    _arm_retry(None, now)
    if not result.ready:
        logger.warning("fact signing unavailable (%s)", result.status)
    return result.signer


def start_for_daemon(slm_home: Path | None = None, *, store: Any = None,
                     timeout: float = READ_TIMEOUT_S,
                     flags: Callable[[Path | None], Any] | None = None,
                     clock: Callable[[], float] | None = None) -> str:
    """The daemon's one read of the key. Returns the status. Never raises.

    Only while ``trust.signing`` is on or in shadow and the kill switch is not thrown;
    otherwise the keychain is not asked at all. An answer later than ``timeout`` is
    collected afterwards by :func:`retry_late_key`.
    """
    try:
        from superlocalmemory.core.feature_flags import FeatureFlags
        from superlocalmemory.infra.data_root import canonical_data_root

        home = slm_home if slm_home is not None else canonical_data_root()
        snapshot = flags(home) if flags is not None else FeatureFlags(home)
        if snapshot.killed or not snapshot.observing("trust.signing"):
            install(None, OFF)
            return OFF
        read = _Read(store if store is not None else KeychainKeyStore(), True)
        result = read.wait(timeout)
    except Exception:  # noqa: BLE001
        install(None, UNAVAILABLE_UNREADABLE)
        return UNAVAILABLE_UNREADABLE
    install(result.signer, result.status)
    _arm_retry(read if result.status == UNAVAILABLE_TIMEOUT else None,
               clock if clock is not None else _clock)
    if not result.ready:
        logger.warning("fact signing unavailable (%s)", result.status)
    return result.status


__all__ = ["ACCOUNT", "KEY_BYTES", "KeychainKeyStore", "Loaded", "NO_KEY", "OFF", "READY",
           "READ_TIMEOUT_S", "SERVICE", "Signer", "StoreUnavailable", "UNAVAILABLE_STORE",
           "UNAVAILABLE_TIMEOUT", "UNAVAILABLE_UNREADABLE", "current", "install", "key_id_of",
           "keychain_blocked", "load", "retry_late_key", "start_for_daemon", "status", "words"]
