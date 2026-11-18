# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Shared pieces: a migrated store, an in-memory keychain (never the real one), the flag."""

from __future__ import annotations

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, env_name
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import signing_key as sk
from tests.test_core.test_coverage_backfill import _db

SIGNING = env_name("trust.signing")


class FakeKeychain:
    """The SecureBackend protocol over a dict. No file, no real keychain."""

    def __init__(self, entries=None):
        self.entries = dict(entries or {})
        self.reads = 0

    def get_password(self, service, username):
        self.reads += 1
        return self.entries.get((service, username))

    def set_password(self, service, username, password):
        self.entries[(service, username)] = password


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    monkeypatch.delenv(KILL_ENV, raising=False)
    monkeypatch.delenv(SIGNING, raising=False)
    sk.install(None)
    yield
    sk.install(None)


@pytest.fixture
def db(tmp_path):
    return _db(tmp_path / "memory.db")


@pytest.fixture
def keychain():
    return FakeKeychain()


@pytest.fixture
def signer():
    return sk.Signer(bytes(range(32)))


def save(db, text, *, memory_id=None):
    if memory_id is None:
        record = MemoryRecord(content=text, session_date="2026-11-15")
        db.store_memory(record)
        memory_id = record.memory_id
    fact = AtomicFact(memory_id=memory_id, content=text)
    db.store_fact(fact)
    return fact
