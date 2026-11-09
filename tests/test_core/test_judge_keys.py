# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Owner-only key storage for the hosted answer check (Jev).

Every test uses an explicit ``slm_home=tmp_path`` so no test can ever touch a
real ``~/.superlocalmemory``. The global ``_block_live_slm_home_writes``
fixture (tests/conftest.py) already redirects ``SLM_DATA_DIR`` to a per-test
tmp_path too, so the no-argument constructor is exercised once, separately,
to prove that path (the real resolver wiring) also works.
"""

from __future__ import annotations

import os
import stat

import pytest

from superlocalmemory.core.judge_keys import (
    PROVIDERS,
    JudgeKeyStore,
    JudgeKeyStoreError,
)
from tests._portable import assert_owner_only

VALID_KEY = "sk-live-" + "a1B2c3D4" * 4  # 40 printable ASCII chars, no spaces


@pytest.fixture
def store(tmp_path) -> JudgeKeyStore:
    return JudgeKeyStore(slm_home=tmp_path)


# ---------------------------------------------------------------------------
# provider validation
# ---------------------------------------------------------------------------

def test_providers_are_typesafe_and_openrouter():
    assert PROVIDERS == ("typesafe", "openrouter")


@pytest.mark.parametrize("method", ["has_key", "masked", "clear", "load"])
def test_unknown_provider_is_rejected_by_every_read_method(store, method):
    with pytest.raises(ValueError):
        getattr(store, method)("not-a-real-provider")


def test_unknown_provider_is_rejected_by_set_key(store):
    with pytest.raises(ValueError):
        store.set_key("not-a-real-provider", VALID_KEY)


# ---------------------------------------------------------------------------
# key validation (set_key) — never echoes the key in the error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_key", ["", "short", "a" * 7, "a" * 4097])
def test_set_key_rejects_out_of_range_length(store, bad_key):
    with pytest.raises(ValueError) as excinfo:
        store.set_key("typesafe", bad_key)
    if bad_key:
        assert bad_key not in str(excinfo.value)


@pytest.mark.parametrize(
    "bad_key",
    [
        "has a space" + "x" * 20,
        "has\ttab" + "x" * 20,
        "has\nnewline" + "x" * 20,
        "unicode-é-char" + "x" * 20,
    ],
)
def test_set_key_rejects_non_printable_ascii_or_whitespace(store, bad_key):
    with pytest.raises(ValueError) as excinfo:
        store.set_key("typesafe", bad_key)
    assert bad_key not in str(excinfo.value)


def test_set_key_accepts_a_valid_key(store):
    store.set_key("typesafe", VALID_KEY)
    assert store.has_key("typesafe")
    assert store.load("typesafe") == VALID_KEY


def test_exactly_eight_chars_is_the_minimum_valid_length(store):
    store.set_key("typesafe", "a" * 8)
    assert store.load("typesafe") == "a" * 8


# ---------------------------------------------------------------------------
# masked()
# ---------------------------------------------------------------------------

def test_masked_is_empty_when_no_key_stored(store):
    assert store.masked("typesafe") == ""


def test_masked_shows_last_four_when_key_is_longer_than_eight(store):
    store.set_key("typesafe", VALID_KEY)
    assert store.masked("typesafe") == "****" + VALID_KEY[-4:]


def test_masked_never_reveals_more_than_four_chars_of_a_short_key(store):
    """An exactly-8-char key: last4 would reveal HALF the key. Must not happen."""
    short_key = "abcd1234"
    store.set_key("typesafe", short_key)
    masked = store.masked("typesafe")
    assert masked == "****"
    assert short_key not in masked
    assert short_key[-4:] not in masked


def test_masked_never_contains_the_full_key(store):
    store.set_key("typesafe", VALID_KEY)
    assert VALID_KEY not in store.masked("typesafe")


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_removes_only_the_named_provider(store):
    store.set_key("typesafe", VALID_KEY)
    store.set_key("openrouter", "or-" + VALID_KEY)
    store.clear("typesafe")
    assert not store.has_key("typesafe")
    assert store.has_key("openrouter")
    assert store.load("openrouter") == "or-" + VALID_KEY


def test_clear_on_a_missing_key_does_not_raise(store):
    store.clear("typesafe")  # never set; must be a no-op, not an error


# ---------------------------------------------------------------------------
# filesystem permissions
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="folder mode bits do nothing on Windows")
def test_secrets_dir_is_mode_0700(store, tmp_path):
    store.set_key("typesafe", VALID_KEY)
    mode = stat.S_IMODE((tmp_path / "secrets").stat().st_mode)
    assert mode == 0o700


def test_key_file_is_mode_0600(store, tmp_path):
    store.set_key("typesafe", VALID_KEY)
    key_path = tmp_path / "secrets" / "jev-typesafe.key"
    assert_owner_only(key_path)  # on Windows: the owner-only access list


def test_set_key_is_atomic_no_tmp_files_left_behind(store, tmp_path):
    store.set_key("typesafe", VALID_KEY)
    store.set_key("typesafe", VALID_KEY + "x")  # overwrite
    leftovers = [p for p in (tmp_path / "secrets").iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    assert store.load("typesafe") == VALID_KEY + "x"


# ---------------------------------------------------------------------------
# symlink / tampering refusal
# ---------------------------------------------------------------------------

def test_load_refuses_a_symlinked_key_file(store, tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    os.chmod(secrets_dir, 0o700)
    target = tmp_path / "elsewhere.txt"
    target.write_text("not a key", encoding="utf-8")
    (secrets_dir / "jev-typesafe.key").symlink_to(target)

    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")
    with pytest.raises(JudgeKeyStoreError):
        store.has_key("typesafe")
    with pytest.raises(JudgeKeyStoreError):
        store.masked("typesafe")


def test_set_key_refuses_to_overwrite_a_symlink(store, tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    os.chmod(secrets_dir, 0o700)
    target = tmp_path / "elsewhere.txt"
    target.write_text("not a key", encoding="utf-8")
    (secrets_dir / "jev-typesafe.key").symlink_to(target)

    with pytest.raises(JudgeKeyStoreError):
        store.set_key("typesafe", VALID_KEY)
    # The symlink target must be untouched.
    assert target.read_text(encoding="utf-8") == "not a key"


def test_clear_removes_a_symlink_without_following_it(store, tmp_path):
    """unlink() on a symlink removes the link, never the target."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    os.chmod(secrets_dir, 0o700)
    target = tmp_path / "elsewhere.txt"
    target.write_text("must survive", encoding="utf-8")
    (secrets_dir / "jev-typesafe.key").symlink_to(target)

    store.clear("typesafe")

    assert not (secrets_dir / "jev-typesafe.key").exists()
    assert target.read_text(encoding="utf-8") == "must survive"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no named pipes in the file system on Windows")
def test_load_refuses_a_non_regular_file(store, tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    os.chmod(secrets_dir, 0o700)
    fifo_path = secrets_dir / "jev-typesafe.key"
    os.mkfifo(fifo_path)

    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


@pytest.mark.skipif(not hasattr(os, "getuid"),
                    reason="Windows has no POSIX file owner uid; the store skips this check there")
def test_load_refuses_a_file_not_owned_by_current_user(store, tmp_path, monkeypatch):
    store.set_key("typesafe", VALID_KEY)
    # Simulate a different owner without needing root to chown for real.
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)
    with pytest.raises(JudgeKeyStoreError):
        store.load("typesafe")


# ---------------------------------------------------------------------------
# home resolution — honours SLM_DATA_DIR via the shared resolver
# ---------------------------------------------------------------------------

def test_default_home_uses_canonical_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    store_default = JudgeKeyStore()
    store_default.set_key("typesafe", VALID_KEY)
    assert (tmp_path / "secrets" / "jev-typesafe.key").exists()


# ---------------------------------------------------------------------------
# the key is never in an exception message
# ---------------------------------------------------------------------------

def test_no_exception_message_anywhere_contains_the_key(store, tmp_path, caplog):
    store.set_key("typesafe", VALID_KEY)
    errors = []
    for bad in ("", "short", "a" * 4097, "has space" + "x" * 20):
        try:
            store.set_key("typesafe", bad)
        except ValueError as exc:
            errors.append(str(exc))
    for message in errors:
        assert VALID_KEY not in message
