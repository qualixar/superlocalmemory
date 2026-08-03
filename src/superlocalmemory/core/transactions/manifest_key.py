# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Installation-bound HMAC signing key for completion manifests and erasure receipts.

The signing key is a 32-byte secret persisted at
``~/.superlocalmemory/.manifest_signing_key`` (mode 0600).  It is generated
once on first use and read on every subsequent call.  Purpose-specific
sub-keys are derived via HMAC so a key compromise in one context cannot be
used to forge a receipt in another.

The key is NEVER logged or included in any error message.

Key file location: resolved via ``state_path(".manifest_signing_key")``.
In tests, monkeypatch ``_signing_key_path`` and/or ``_ensure_signing_key``
to return a deterministic path or byte sequence.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import os
import secrets
import sys
from pathlib import Path

_KEY_CONTEXT_MANIFEST: bytes = b"slm-manifest-v2"
_KEY_CONTEXT_RECEIPT: bytes = b"slm-receipt-v2"
_MANIFEST_VERSION: int = 2


def _signing_key_path() -> Path:
    from superlocalmemory.infra.data_root import state_path

    return state_path(".manifest_signing_key")


def _ensure_signing_key() -> bytes:
    """Read the 32-byte hex signing key, creating it on first call.

    Returns the raw bytes of the key (NOT the hex string).
    """
    key_path = _signing_key_path()
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        raw = key_path.read_text(encoding="utf-8").strip()
        if len(raw) == 64:
            return bytes.fromhex(raw)

    raw = secrets.token_hex(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(key_path), flags, 0o600)
        try:
            os.write(fd, raw.encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        # Another process won the race — read its key.
        try:
            existing = key_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if len(existing) == 64:
            return bytes.fromhex(existing)
        # Truncated write by the winner — fall through and use our in-memory key.
    except OSError:
        # Permission / symlink error — do NOT overwrite an existing key file.
        # Return an ephemeral key for this call; next call retries the file path.
        return bytes.fromhex(raw)

    if sys.platform != "win32":
        try:
            os.chmod(str(key_path), 0o600)
        except OSError:
            pass

    return bytes.fromhex(raw)


def derive_manifest_hmac_key() -> bytes:
    """Return the HMAC sub-key for completion manifests."""
    master = _ensure_signing_key()
    return _hmac.new(master, _KEY_CONTEXT_MANIFEST, hashlib.sha256).digest()


def derive_receipt_hmac_key() -> bytes:
    """Return the HMAC sub-key for erasure receipts."""
    master = _ensure_signing_key()
    return _hmac.new(master, _KEY_CONTEXT_RECEIPT, hashlib.sha256).digest()


def compute_hmac(key: bytes, payload: bytes) -> str:
    """Return HMAC-SHA256 of ``payload`` under ``key`` as a hex string."""
    return _hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_hmac(stored_mac: str, key: bytes, payload: bytes) -> bool:
    """Constant-time verify that ``stored_mac`` matches HMAC of ``payload`` under ``key``."""
    try:
        expected = _hmac.new(key, payload, hashlib.sha256).digest()
        stored_bytes = bytes.fromhex(stored_mac)
    except (ValueError, TypeError):
        return False
    return _hmac.compare_digest(expected, stored_bytes)


__all__ = [
    "MANIFEST_VERSION",
    "compute_hmac",
    "derive_manifest_hmac_key",
    "derive_receipt_hmac_key",
    "verify_hmac",
]

MANIFEST_VERSION = _MANIFEST_VERSION
