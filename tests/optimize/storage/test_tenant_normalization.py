# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tenant-id normalization regression (v3.6.10).

The proxy CacheManager stores entries under SHA-256(tenant) when the tenant
isn't already a 64-hex digest. The public CacheDB helpers (entry_count,
clear_tenant, entry_exists) MUST hash the same way — otherwise the dashboard
reports "0 entries" while the cache is populated (found in live testing).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from superlocalmemory.optimize.storage.db import CacheDB, _normalize_tenant_id


def test_normalize_plain_tenant_is_hashed() -> None:
    assert _normalize_tenant_id("default") == hashlib.sha256(b"default").hexdigest()


def test_normalize_already_hex_is_unchanged() -> None:
    h = hashlib.sha256(b"default").hexdigest()
    assert _normalize_tenant_id(h) == h


def test_normalize_empty_falls_back_to_default() -> None:
    assert _normalize_tenant_id("") == hashlib.sha256(b"default").hexdigest()


def test_entry_count_sees_hashed_tenant_entry(tmp_path: Path) -> None:
    """An entry stored under the hashed tenant is counted by entry_count('default')."""
    db = CacheDB(tmp_path / "llmcache.db")
    hashed = hashlib.sha256(b"default").hexdigest()
    # Store via the canonical set() path using the hashed tenant (as the proxy does).
    db.set(
        "k" * 64,
        hashed,
        b'{"ok":true}',
        model="claude-x",
        ttl_expires=None,
        tags=[],
    )
    # Dashboard-style query with the PLAIN tenant must find it.
    assert db.entry_count("default") == 1
    assert db.entry_exists("k" * 64, "default") is True
    # And clearing by the plain tenant removes it.
    assert db.clear_tenant("default") == 1
    assert db.entry_count("default") == 0
    db.close()
