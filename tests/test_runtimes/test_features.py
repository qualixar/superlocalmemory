"""The feature switch: off by default, written only when the person turns it on."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading

import pytest

from superlocalmemory.media import media_db_path
from superlocalmemory.runtimes import features
from superlocalmemory.runtimes.managed_env import EnvStatus


class FakeEnv:
    def __init__(self, state="not_installed"):
        self.state = state
        self.installs = 0
        self.removed = []
        self.done = threading.Event()

    def status(self):
        return EnvStatus(state=self.state, progress=0.0, step="x")

    def precheck(self):
        return {"disk_ok": True}

    def install(self, **_):
        self.installs += 1
        self.state = "ready"
        self.done.set()
        return self.status()

    def remove(self, *, keep_weights):
        self.removed.append(keep_weights)
        self.state = "not_installed"


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    r.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    return r


def test_defaults_and_reads_create_nothing(root):
    (root / "memory.db").write_bytes(b"")
    before = sorted(os.listdir(root))
    assert features.read_features(root)["media"]["enabled"] is False
    assert features.media_enabled(root) is False
    status = features.media_feature_status(root, env=FakeEnv())
    assert status["enabled"] is False and status["media_db"] is False
    assert sorted(os.listdir(root)) == before


def test_enable_writes_file_creates_db_and_installs_once(root):
    env = FakeEnv()
    out = features.enable_media(source="cli", env=env, data_root=root)
    assert out["enabled"] is True
    assert env.done.wait(10)
    path = features.features_path(root)
    data = json.loads(path.read_text())
    assert data["media"]["enabled"] is True and data["media"]["choice_source"] == "cli"
    assert data["media"]["enabled_at"]
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert media_db_path(root).exists()
    features.enable_media(source="cli", env=env, data_root=root)
    assert env.installs == 1


def test_enable_without_install(root):
    env = FakeEnv()
    features.enable_media(source="api", start_install=False, env=env, data_root=root)
    assert env.installs == 0 and features.media_enabled(root)


def test_bad_source_raises_and_writes_nothing(root):
    with pytest.raises(ValueError):
        features.enable_media(source="telepathy", env=FakeEnv(), data_root=root)
    assert not features.features_path(root).exists()


def test_disable_keeps_media_db_and_calls_stop_hook(root):
    env = FakeEnv("ready")
    features.enable_media(source="dashboard", start_install=False, env=env, data_root=root)
    calls = []
    features.register_media_stop_hook(lambda: calls.append(1))
    try:
        out = features.disable_media(env=env, data_root=root)
    finally:
        features.register_media_stop_hook(None)
    assert out["enabled"] is False and calls == [1]
    assert media_db_path(root).exists() and env.removed == []


def test_disable_with_remove_files_removes_env(root):
    env = FakeEnv("ready")
    features.enable_media(source="npm", start_install=False, env=env, data_root=root)
    features.disable_media(remove_files=True, env=env, data_root=root)
    assert env.removed == [False]


def test_corrupt_file_reads_as_defaults(root):
    features.features_path(root).write_text("{not json")
    assert features.media_enabled(root) is False


def test_read_only_root_returns_failed_status_without_raising(root, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(features, "_write_features", boom)
    out = features.enable_media(source="cli", start_install=False, env=FakeEnv(), data_root=root)
    assert out["enabled"] is False and out.get("error")
