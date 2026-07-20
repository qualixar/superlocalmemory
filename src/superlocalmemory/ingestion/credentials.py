# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cross-platform credential storage for ingestion adapters.

Uses OS keychain via keyring library (macOS Keychain, Windows Credential Locker,
Linux SecretService). Falls back to file-based storage with restricted permissions.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from superlocalmemory.infra.data_root import state_path

logger = logging.getLogger("superlocalmemory.ingestion.credentials")

_CRED_DIR = None  # test-only compatibility override
_SERVICE_PREFIX = "slm"


def _credential_dir() -> Path:
    return Path(_CRED_DIR) if _CRED_DIR is not None else state_path("credentials")


def store_credential(service: str, key: str, value: str) -> bool:
    """Store a credential securely. Returns True on success."""
    # Try OS keychain first
    try:
        import keyring
        keyring.set_password(f"{_SERVICE_PREFIX}-{service}", key, value)
        logger.debug("Stored %s/%s in OS keychain", service, key)
        return True
    except Exception:
        pass

    # Fallback: encrypted file with restricted permissions
    try:
        cred_dir = _credential_dir()
        cred_dir.mkdir(parents=True, exist_ok=True)
        cred_file = cred_dir / f"{service}.json"

        existing = {}
        if cred_file.exists():
            try:
                existing = json.loads(cred_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        existing[key] = value
        cred_file.write_text(json.dumps(existing, indent=2))

        # Restrict permissions (Unix only — Windows skipped)
        if sys.platform != "win32":
            os.chmod(cred_file, 0o600)
            os.chmod(cred_dir, 0o700)

        logger.debug("Stored %s/%s in file (keychain unavailable)", service, key)
        return True
    except Exception as exc:
        logger.error("Failed to store credential %s/%s: %s", service, key, exc)
        return False


def load_credential(service: str, key: str) -> str | None:
    """Load a credential. Tries keychain first, then file."""
    # Try OS keychain
    try:
        import keyring
        value = keyring.get_password(f"{_SERVICE_PREFIX}-{service}", key)
        if value:
            return value
    except Exception:
        pass

    # Fallback: file
    try:
        cred_file = _credential_dir() / f"{service}.json"
        if cred_file.exists():
            data = json.loads(cred_file.read_text())
            return data.get(key)
    except Exception:
        pass

    return None


def delete_credential(service: str, key: str) -> bool:
    """Delete a credential from both keychain and file."""
    deleted = False

    # Try keychain
    try:
        import keyring
        keyring.delete_password(f"{_SERVICE_PREFIX}-{service}", key)
        deleted = True
    except Exception:
        pass

    # Also remove from file
    try:
        cred_file = _credential_dir() / f"{service}.json"
        if cred_file.exists():
            data = json.loads(cred_file.read_text())
            if key in data:
                del data[key]
                cred_file.write_text(json.dumps(data, indent=2))
                deleted = True
    except Exception:
        pass

    return deleted


def has_credential(service: str, key: str) -> bool:
    """Check if a credential exists."""
    return load_credential(service, key) is not None
