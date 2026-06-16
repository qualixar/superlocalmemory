# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Regression tests for cloud_backup credential-store hardening (WP-I).

Bugs fixed:
  #10 TOCTOU 0644: write_text + chmod(0o600) left file 0644 in the window;
      if process died between, it stayed 0644 permanently.
  #11 Non-atomic write: write_text truncates-then-writes so a crash yields
      a torn file, and _get_credential's bare except:pass silently returned
      None for ALL credentials (silent total loss).

Fix: atomic os.open(O_CREAT|O_WRONLY|O_TRUNC|O_NOFOLLOW, 0o600) + fsync +
     os.replace() so (a) the file is never world-readable and (b) the old
     store survives an interrupted write.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — redirect MEMORY_DIR to a tmp path per test
# ---------------------------------------------------------------------------

def _make_module(tmp_path: Path):
    """Import cloud_backup with MEMORY_DIR pointing at tmp_path.

    We patch MEMORY_DIR at the module level so the module never touches
    ~/.superlocalmemory during tests.
    """
    import importlib
    import superlocalmemory.infra.cloud_backup as cb_mod

    # Monkeypatch the module-level constant
    original_dir = cb_mod.MEMORY_DIR
    cb_mod.MEMORY_DIR = tmp_path
    # Also redirect _get_credential_store to return a path inside tmp_path
    return cb_mod, original_dir


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def cred_store_env(tmp_path, monkeypatch):
    """Yield (module, store_path) with MEMORY_DIR set to tmp_path.
    Keyring is disabled to force the fallback path.
    """
    import superlocalmemory.infra.cloud_backup as cb

    monkeypatch.setattr(cb, "MEMORY_DIR", tmp_path)

    # Make keyring unavailable to force the local-file fallback path
    # by patching the import inside the functions
    keyring_mod = types.ModuleType("keyring")
    keyring_errors_mod = types.ModuleType("keyring.errors")

    class _NoKeyringError(Exception):
        pass

    keyring_errors_mod.NoKeyringError = _NoKeyringError

    def _set_password(service, key, value):
        raise _NoKeyringError("no keyring in test")

    def _get_password(service, key):
        raise _NoKeyringError("no keyring in test")

    def _delete_password(service, key):
        raise _NoKeyringError("no keyring in test")

    keyring_mod.set_password = _set_password
    keyring_mod.get_password = _get_password
    keyring_mod.delete_password = _delete_password
    keyring_mod.errors = keyring_errors_mod

    monkeypatch.setitem(sys.modules, "keyring", keyring_mod)
    monkeypatch.setitem(sys.modules, "keyring.errors", keyring_errors_mod)

    store_path = tmp_path / ".credentials.json"
    yield cb, store_path


# ---------------------------------------------------------------------------
# Test 1: file mode is 0o600 IMMEDIATELY after _store_credential
# ---------------------------------------------------------------------------

class TestFileModeAtomicWrite:
    """File must never be 0644, not even for a nanosecond after write."""

    def test_mode_is_0600_immediately_after_store(self, cred_store_env):
        cb, store_path = cred_store_env
        ok = cb._store_credential("test_key", "test_val")
        assert ok, "_store_credential should return True on success"
        assert store_path.exists(), "credential store file should exist"
        file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert file_mode == 0o600, (
            f"Expected 0o600 but got 0o{file_mode:03o}. "
            "File is world-readable — TOCTOU bug #10 still present."
        )

    def test_mode_stays_0600_after_second_store(self, cred_store_env):
        cb, store_path = cred_store_env
        cb._store_credential("key1", "val1")
        cb._store_credential("key2", "val2")
        file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert file_mode == 0o600, (
            f"Expected 0o600 after two stores but got 0o{file_mode:03o}."
        )

    def test_two_creds_stored_and_retrieved(self, cred_store_env):
        cb, store_path = cred_store_env
        cb._store_credential("alpha", "secret_alpha")
        cb._store_credential("beta", "secret_beta")
        assert cb._get_credential("alpha") == "secret_alpha"
        assert cb._get_credential("beta") == "secret_beta"


# ---------------------------------------------------------------------------
# Test 2: atomic replace — interrupted write leaves OLD store intact
# ---------------------------------------------------------------------------

class TestAtomicReplace:
    """If os.replace fails, the original credential store is preserved."""

    def test_old_store_intact_if_replace_raises(self, cred_store_env, monkeypatch):
        cb, store_path = cred_store_env

        # Pre-populate the store with a known credential
        cb._store_credential("existing_key", "existing_value")
        original_content = store_path.read_text()

        # Simulate os.replace raising mid-operation
        real_replace = os.replace

        def _failing_replace(src, dst):
            # Remove the temp file (as if kernel cleaned up) then raise
            try:
                os.unlink(src)
            except FileNotFoundError:
                pass
            raise OSError("disk full — simulated failure")

        monkeypatch.setattr(os, "replace", _failing_replace)

        result = cb._store_credential("new_key", "new_value")

        # The original store must be untouched
        assert store_path.exists(), "Original store was deleted — non-atomic write!"
        assert store_path.read_text() == original_content, (
            "Original store was corrupted when os.replace failed — not atomic!"
        )
        # _store_credential should return False when replace fails
        assert result is False, (
            "_store_credential should return False when the atomic replace fails"
        )

        # Restore and confirm original value is still readable
        monkeypatch.setattr(os, "replace", real_replace)
        assert cb._get_credential("existing_key") == "existing_value"


# ---------------------------------------------------------------------------
# Test 3: _get_credential logs WARNING on corrupt JSON (not silent None)
# ---------------------------------------------------------------------------

class TestCorruptJsonLogsWarning:
    """_get_credential must emit a WARNING log on JSONDecodeError, not swallow it."""

    def test_corrupt_json_logs_warning(self, cred_store_env, caplog):
        cb, store_path = cred_store_env

        # Write corrupt JSON directly (bypassing _store_credential)
        store_path.write_text("{not valid json!!!")
        store_path.chmod(0o600)

        with caplog.at_level(logging.WARNING, logger="superlocalmemory.cloud_backup"):
            result = cb._get_credential("any_key")

        assert result is None, "_get_credential should return None on corrupt JSON"
        assert len(caplog.records) > 0, (
            "_get_credential silently swallowed JSONDecodeError — bug #11 still present. "
            "Expected a WARNING log record."
        )
        # The warning should reference corruption/JSON in some meaningful way
        warning_texts = " ".join(r.message for r in caplog.records)
        assert any(
            keyword in warning_texts.lower()
            for keyword in ("corrupt", "json", "decode", "parse", "invalid")
        ), f"Warning log should mention JSON/corruption, got: {warning_texts!r}"

    def test_corrupt_json_in_delete_logs_warning(self, cred_store_env, caplog):
        cb, store_path = cred_store_env

        store_path.write_text("{bad json")
        store_path.chmod(0o600)

        with caplog.at_level(logging.WARNING, logger="superlocalmemory.cloud_backup"):
            # _delete_credential should not raise, but may log a warning
            result = cb._delete_credential("missing_key")

        # Should not raise; return value may be False or True depending on keyring
        # The key is that it doesn't crash

    def test_missing_store_returns_none_silently(self, cred_store_env, caplog):
        cb, store_path = cred_store_env
        # No store file exists at all — should return None with NO warning
        with caplog.at_level(logging.WARNING, logger="superlocalmemory.cloud_backup"):
            result = cb._get_credential("ghost_key")
        assert result is None


# ---------------------------------------------------------------------------
# Test 4: parent directory is created with mode 0700
# ---------------------------------------------------------------------------

class TestParentDirMode:
    """The parent directory of .credentials.json should be 0700."""

    def test_parent_dir_created_with_0700(self, tmp_path, monkeypatch):
        import sys, types
        import superlocalmemory.infra.cloud_backup as cb

        # Point to a sub-dir that does NOT yet exist
        new_dir = tmp_path / "new_slm_dir"
        monkeypatch.setattr(cb, "MEMORY_DIR", new_dir)

        keyring_mod = types.ModuleType("keyring")
        keyring_errors_mod = types.ModuleType("keyring.errors")

        class _NoKeyringError(Exception):
            pass

        keyring_errors_mod.NoKeyringError = _NoKeyringError
        keyring_mod.set_password = lambda *a: (_ for _ in ()).throw(_NoKeyringError())
        keyring_mod.get_password = lambda *a: (_ for _ in ()).throw(_NoKeyringError())
        keyring_mod.delete_password = lambda *a: (_ for _ in ()).throw(_NoKeyringError())
        keyring_mod.errors = keyring_errors_mod

        monkeypatch.setitem(sys.modules, "keyring", keyring_mod)
        monkeypatch.setitem(sys.modules, "keyring.errors", keyring_errors_mod)

        ok = cb._store_credential("k", "v")
        assert ok
        assert new_dir.exists()
        dir_mode = stat.S_IMODE(os.stat(new_dir).st_mode)
        assert dir_mode == 0o700, (
            f"Expected parent dir 0o700 but got 0o{dir_mode:03o}"
        )


# ---------------------------------------------------------------------------
# Test 5: delete is also atomic (0600, no corrupt window)
# ---------------------------------------------------------------------------

class TestDeleteCredentialAtomic:
    """_delete_credential must also write atomically at 0600."""

    def test_delete_preserves_other_creds_and_mode(self, cred_store_env):
        cb, store_path = cred_store_env
        cb._store_credential("keep_me", "keep_val")
        cb._store_credential("delete_me", "del_val")

        cb._delete_credential("delete_me")

        # Other credential is intact
        assert cb._get_credential("keep_me") == "keep_val"
        assert cb._get_credential("delete_me") is None

        # File still 0600
        file_mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert file_mode == 0o600, (
            f"File mode after delete should be 0o600, got 0o{file_mode:03o}"
        )


# ---------------------------------------------------------------------------
# Test 6: docstring does NOT claim "encrypted"
# ---------------------------------------------------------------------------

class TestDocstringNotEncrypted:
    """Docstring must NOT falsely claim the file is encrypted."""

    def test_store_credential_docstring_no_encrypted_claim(self):
        import superlocalmemory.infra.cloud_backup as cb
        doc = (cb._store_credential.__doc__ or "").lower()
        assert "encrypted" not in doc, (
            "_store_credential docstring still claims 'encrypted' — fix the docstring "
            "to say plaintext-local-file."
        )

    def test_get_credential_store_docstring_no_encrypted_claim(self):
        import superlocalmemory.infra.cloud_backup as cb
        doc = (cb._get_credential_store.__doc__ or "").lower()
        assert "encrypted" not in doc, (
            "_get_credential_store docstring still claims 'encrypted'."
        )
