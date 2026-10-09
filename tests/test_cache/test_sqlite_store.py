# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The on-disk derivation cache."""

import os
import sqlite3
import stat
import sys
import threading

import pytest

from superlocalmemory.cache import SqliteDeriveCache

from .conftest import make_key


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "derive_cache.db"


def test_a_miss_on_a_missing_file_creates_nothing(path):
    cache = SqliteDeriveCache(path)
    assert not path.exists()  # the constructor does no I/O
    assert cache.get(make_key(1)) is None
    assert cache.stats() == {"backend": "sqlite", "entries": 0, "bytes": 0,
                             "path": str(path), "exists": False}
    assert cache.invalidate(deriver_id="pdf.render") == 0
    assert not path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_the_first_put_creates_the_file_private(path):
    SqliteDeriveCache(path).put(make_key(1), b"x", kind="text")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


@pytest.mark.parametrize("kind,payload", [("text", b"hello"), ("vector_f32", b"\x00\x00\x80?"),
                                          ("json", b'{"a":1}'), ("png", b"\x89PNG")])
def test_every_kind_round_trips(path, kind, payload):
    cache = SqliteDeriveCache(path)
    cache.put(make_key(1), payload, kind=kind)
    assert cache.get(make_key(1)) == payload


def test_an_unknown_kind_or_non_bytes_is_refused(path):
    cache = SqliteDeriveCache(path)
    with pytest.raises(ValueError):
        cache.put(make_key(1), b"x", kind="pickle")
    with pytest.raises(TypeError):
        cache.put(make_key(1), "text", kind="text")  # type: ignore[arg-type]
    assert not path.exists()


def test_putting_the_same_key_replaces_it(path):
    cache = SqliteDeriveCache(path)
    cache.put(make_key(1), b"one", kind="text")
    cache.put(make_key(1), b"two", kind="text")
    assert cache.get(make_key(1)) == b"two"
    assert cache.stats()["entries"] == 1


def test_invalidate_by_deriver_and_by_model(path):
    cache = SqliteDeriveCache(path)
    cache.put(make_key(1, deriver="a.b"), b"1", kind="text")
    cache.put(make_key(2, deriver="a.b", model="m1"), b"2", kind="text")
    cache.put(make_key(3, deriver="c.d", model="m1"), b"3", kind="text")
    cache.put(make_key(4, deriver="c.d", model="m2"), b"4", kind="text")
    assert cache.invalidate(model_id="m1") == 2
    assert cache.get(make_key(4, deriver="c.d", model="m2")) == b"4"
    assert cache.invalidate(deriver_id="a.b") == 1
    assert cache.stats()["entries"] == 1


def test_invalidate_needs_a_filter(path):
    with pytest.raises(ValueError):
        SqliteDeriveCache(path).invalidate()


def test_clear_removes_the_file_and_its_sidecars(path):
    cache = SqliteDeriveCache(path)
    cache.put(make_key(1), b"x", kind="text")
    assert path.with_name(path.name + "-wal").exists()  # WAL mode keeps sidecars
    cache.clear()
    assert not path.exists()
    assert not list(path.parent.glob("derive_cache.db*"))
    assert cache.get(make_key(1)) is None


def test_the_size_cap_trims_the_least_recently_used(path, monkeypatch):
    monkeypatch.setenv("SLM_DERIVE_CACHE_MAX_MB", "1")
    cache = SqliteDeriveCache(path)
    blob = b"z" * 100_000
    for n in range(8):
        cache.put(make_key(n), blob, kind="text")
    assert cache.get(make_key(0)) == blob  # touched: now the newest by use
    for n in range(8, 14):
        cache.put(make_key(n), blob, kind="text")
    stats = cache.stats()
    assert stats["bytes"] <= 1024 * 1024
    assert cache.get(make_key(13)) == blob
    assert cache.get(make_key(1)) is None  # the oldest-used went first


def test_a_corrupt_file_is_set_aside_and_the_cache_works(path):
    path.write_bytes(b"this is not a database" * 100)
    cache = SqliteDeriveCache(path)
    assert cache.get(make_key(1)) is None
    cache.put(make_key(1), b"ok", kind="text")
    assert cache.get(make_key(1)) == b"ok"
    assert list(path.parent.glob("derive_cache.db.corrupt-*"))


def test_schema_matches_the_documented_tables(path):
    SqliteDeriveCache(path).put(make_key(1), b"x", kind="text")
    conn = sqlite3.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"derivations", "ix_deriv_lru", "cache_meta"} <= names
    assert conn.execute("SELECT value FROM cache_meta WHERE key='schema_version'"
                        ).fetchone() == ("1",)
    conn.close()


def test_four_threads_can_put_and_get(path):
    cache = SqliteDeriveCache(path)
    errors: list[BaseException] = []

    def work(t):
        try:
            for n in range(200):
                k = make_key(t * 1000 + n)
                cache.put(k, f"{t}-{n}".encode(), kind="text")
                assert cache.get(k) == f"{t}-{n}".encode()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert all(cache.get(make_key(t * 1000 + n)) for t in range(4) for n in range(200))
