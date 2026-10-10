"""The yes/no that keeps folder sources off while remote access is set up."""

from __future__ import annotations

import pytest

from superlocalmemory.server import remote_access_state as state


class _Key:
    def __init__(self, active):
        self.active = active


class _Store:
    def __init__(self, keys):
        self._keys = keys

    def list(self):
        return tuple(self._keys)


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.remote_listener.try_load_config",
                        lambda main_port=None: (None, None))
    monkeypatch.setattr("superlocalmemory.server.remote_keys.default_store", lambda: _Store([]))
    monkeypatch.setattr("superlocalmemory.core.remote_mode.is_remote_mode", lambda: False)


def test_nothing_set_up(quiet):
    assert state.remote_access_configured() is False


def test_listener_config_counts(quiet, monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.remote_listener.try_load_config",
                        lambda main_port=None: (object(), None))
    assert state.remote_access_configured() is True


def test_broken_listener_config_counts(quiet, monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.remote_listener.try_load_config",
                        lambda main_port=None: (None, ValueError("bad")))
    assert state.remote_access_configured() is True


def test_active_key_counts_revoked_does_not(quiet, monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.remote_keys.default_store",
                        lambda: _Store([_Key(False)]))
    assert state.remote_access_configured() is False
    monkeypatch.setattr("superlocalmemory.server.remote_keys.default_store",
                        lambda: _Store([_Key(False), _Key(True)]))
    assert state.remote_access_configured() is True


def test_lan_mode_counts(quiet, monkeypatch):
    monkeypatch.setattr("superlocalmemory.core.remote_mode.is_remote_mode", lambda: True)
    assert state.remote_access_configured() is True


def test_any_failure_counts_as_on(quiet, monkeypatch):
    def boom():
        raise OSError("unreadable")

    monkeypatch.setattr("superlocalmemory.server.remote_keys.default_store", boom)
    assert state.remote_access_configured() is True
