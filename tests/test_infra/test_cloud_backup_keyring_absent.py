"""Stage-9 regression: cloud_backup credential funcs must NOT raise
UnboundLocalError when the `keyring` package is absent (headless Linux /
minimal Docker). Previously `from keyring.errors import NoKeyringError` lived
inside the same try whose `import keyring` failed, leaving the name unbound
when the `except (ImportError, NoKeyringError)` tuple was evaluated.
"""
import importlib
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def cb_no_keyring(monkeypatch, tmp_path):
    # Simulate keyring not installed: importing it raises ImportError.
    monkeypatch.setitem(sys.modules, "keyring", None)
    import superlocalmemory.infra.cloud_backup as cb
    cb = importlib.reload(cb)
    monkeypatch.setattr(cb, "MEMORY_DIR", Path(tmp_path))
    yield cb
    # reload again clean for other tests
    monkeypatch.undo()
    importlib.reload(cb)


def test_store_credential_no_keyring_does_not_raise(cb_no_keyring):
    assert cb_no_keyring._store_credential("k", "v") is True


def test_get_credential_no_keyring_does_not_raise(cb_no_keyring):
    cb_no_keyring._store_credential("k", "v")
    assert cb_no_keyring._get_credential("k") == "v"


def test_delete_credential_no_keyring_does_not_raise(cb_no_keyring):
    cb_no_keyring._store_credential("k", "v")
    assert cb_no_keyring._delete_credential("k") is True
    assert cb_no_keyring._get_credential("k") is None


def test_full_roundtrip_via_fallback(cb_no_keyring):
    assert cb_no_keyring._store_credential("a", "1") is True
    assert cb_no_keyring._store_credential("b", "2") is True
    assert cb_no_keyring._get_credential("a") == "1"
    assert cb_no_keyring._get_credential("b") == "2"
