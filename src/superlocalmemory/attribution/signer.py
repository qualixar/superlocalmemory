# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cryptographic signing for content attribution and tamper detection.

Provides HMAC-SHA256 signing so that every piece of content produced
by the system carries a verifiable proof of origin.  Verification
detects any modification to the content after signing.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone
from typing import Dict

from superlocalmemory.infra.data_root import state_path


def _get_or_create_key() -> str:
    """Load key from env or generate a persistent random one."""
    env_key = os.environ.get("SLM_SIGNER_KEY")
    if env_key:
        return env_key

    key_path = state_path(".signer_key")
    try:
        key = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key = ""
    if key:
        return key

    import secrets

    key = secrets.token_hex(32)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(key_path), flags, 0o600)
        try:
            os.write(fd, key.encode("utf-8"))
        finally:
            os.close(fd)
    except FileExistsError:
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"signer key is empty: {key_path}")
    if os.name != "nt":
        os.chmod(key_path, 0o600)
    return key


class QualixarSigner:
    """Signs content with HMAC-SHA256 for tamper-proof attribution.

    Typical usage::

        signer = QualixarSigner()
        attribution = signer.sign("some content")
        assert signer.verify("some content", attribution) is True

    Args:
        secret_key: Shared secret used for HMAC computation.
    """

    # ---- Branding constants (single source of truth) ----
    _PLATFORM: str = "Qualixar"
    _AUTHOR: str = "Varun Pratap Bhardwaj"
    _AUTHOR_URL: str = "https://varunpratap.com"
    _LICENSE: str = "AGPL-3.0-or-later"

    def __init__(self, secret_key: str | None = None) -> None:
        if secret_key is None:
            secret_key = _get_or_create_key()
        if not secret_key:
            raise ValueError("secret_key must be a non-empty string")
        self._secret_key: bytes = secret_key.encode("utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sign(self, content: str) -> Dict[str, str]:
        """Sign *content* and return attribution metadata.

        Args:
            content: The text to sign.

        Returns:
            A dict containing:

            - ``platform``     – always ``"Qualixar"``
            - ``author``       – always ``"Varun Pratap Bhardwaj"``
            - ``license``      – always ``"AGPL-3.0-or-later"``
            - ``content_hash`` – SHA-256 hex digest of *content*
            - ``signature``    – HMAC-SHA256 hex digest (key-dependent)
            - ``timestamp``    – ISO 8601 UTC timestamp
        """
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        signature = hmac.new(
            self._secret_key, content_bytes, hashlib.sha256
        ).hexdigest()
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        return {
            "platform": self._PLATFORM,
            "author": self._AUTHOR,
            "license": self._LICENSE,
            "content_hash": content_hash,
            "signature": signature,
            "timestamp": timestamp,
        }

    def verify(self, content: str, attribution: Dict[str, str]) -> bool:
        """Verify that *content* matches its attribution signature.

        Checks both the SHA-256 content hash and the HMAC-SHA256
        signature.  Returns ``False`` if either has been tampered with.

        Args:
            content: The text to verify.
            attribution: The dict returned by :meth:`sign`.

        Returns:
            ``True`` if valid, ``False`` if tampered or mismatched.
        """
        content_bytes = content.encode("utf-8")

        # 1. Verify content hash
        expected_hash = hashlib.sha256(content_bytes).hexdigest()
        if not hmac.compare_digest(
            expected_hash, attribution.get("content_hash", "")
        ):
            return False

        # 2. Verify HMAC signature
        expected_sig = hmac.new(
            self._secret_key, content_bytes, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(
            expected_sig, attribution.get("signature", "")
        ):
            return False

        return True

    @staticmethod
    def get_attribution() -> Dict[str, str]:
        """Return basic (unsigned) attribution metadata.

        This is a convenience method for embedding attribution in
        contexts where full signing is not needed (e.g., file headers).

        Returns:
            A dict with ``platform``, ``author``, ``author_url``, and
            ``license`` keys.
        """
        return {
            "platform": QualixarSigner._PLATFORM,
            "author": QualixarSigner._AUTHOR,
            "author_url": QualixarSigner._AUTHOR_URL,
            "license": QualixarSigner._LICENSE,
        }
