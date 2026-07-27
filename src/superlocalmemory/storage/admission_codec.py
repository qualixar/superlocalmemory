# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Production encryption codec for durable remember admission commands."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from superlocalmemory.infra.data_root import state_path

_KEY_BYTES = 32
_NONCE_BYTES = 12
_FORMAT_VERSION = b"\x01"
_ASSOCIATED_DATA = b"superlocalmemory:remember-admission:v1"
_KEY_READ_ATTEMPTS = 100
_KEY_READ_RETRY_SECONDS = 0.005


class AdmissionKeyError(RuntimeError):
    """The machine-local admission key cannot be created or loaded safely."""


class MachineKeyCommandCodec:
    """AES-256-GCM codec backed by a machine-local, permission-bound key.

    The key is intentionally separate from cache encryption keys.  Exclusive
    creation makes concurrent daemon startup converge on one key, while the
    format byte and AEAD associated data prevent cross-protocol ciphertext use.
    """

    def __init__(self, key_path: str | Path | None = None) -> None:
        self.key_path = Path(key_path) if key_path is not None else state_path(
            "admission-key.bin"
        )
        self._key = _load_or_create_key(self.key_path)

    def encrypt(self, plaintext: bytes) -> bytes:
        if not isinstance(plaintext, bytes):
            raise TypeError("plaintext must be bytes")
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._key).encrypt(nonce, plaintext, _ASSOCIATED_DATA)
        return _FORMAT_VERSION + nonce + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not isinstance(ciphertext, bytes):
            raise TypeError("ciphertext must be bytes")
        minimum = len(_FORMAT_VERSION) + _NONCE_BYTES + 16
        if len(ciphertext) < minimum or ciphertext[:1] != _FORMAT_VERSION:
            raise ValueError("unsupported admission ciphertext format")
        nonce = ciphertext[1 : 1 + _NONCE_BYTES]
        payload = ciphertext[1 + _NONCE_BYTES :]
        return AESGCM(self._key).decrypt(nonce, payload, _ASSOCIATED_DATA)


def _load_or_create_key(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        os.chmod(path.parent, 0o700)

    try:
        fd = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError:
        fd = -1
    else:
        try:
            key = os.urandom(_KEY_BYTES)
            view = memoryview(key)
            written = 0
            while written < _KEY_BYTES:
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise AdmissionKeyError("admission key write was incomplete")
                written += count
            os.fsync(fd)
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        finally:
            os.close(fd)

    for attempt in range(_KEY_READ_ATTEMPTS):
        try:
            info = path.lstat()
        except OSError as exc:
            raise AdmissionKeyError("admission key is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise AdmissionKeyError("admission key path must be a regular file")
        if os.name != "nt":
            if info.st_uid != os.getuid():
                raise AdmissionKeyError("admission key is not owned by the current user")
            os.chmod(path, 0o600)
        try:
            key = path.read_bytes()
        except OSError as exc:
            raise AdmissionKeyError("admission key cannot be read") from exc
        if len(key) == _KEY_BYTES:
            return key
        # A process that lost O_EXCL may observe the winning creator between
        # file creation and its final fsync/close. Wait only for that bounded
        # window; a persistently truncated or oversized key still fails closed.
        if attempt + 1 < _KEY_READ_ATTEMPTS:
            time.sleep(_KEY_READ_RETRY_SECONDS)
    raise AdmissionKeyError("admission key has an invalid length")
