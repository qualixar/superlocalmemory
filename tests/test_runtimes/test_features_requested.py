# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A request recorded by the installer is honoured once, by the daemon, and the
restart flag says when a running daemon has not yet picked the feature up."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.runtimes import features
from tests.test_runtimes.test_features import FakeEnv


def _request(root, **extra):
    (root / "features.json").write_text(json.dumps(
        {"schema": 1, "media": {"requested": True, "requested_at": "t", "choice_source": "npm", **extra}}))


def test_a_request_is_enabled_once_with_the_installer_as_source(tmp_path):
    _request(tmp_path)
    env = FakeEnv()
    first = features.apply_requested(source="npm", env=env, data_root=tmp_path)
    assert first is not None and first["enabled"] is True
    saved = features.read_features(tmp_path)["media"]
    assert saved["enabled"] is True and saved["choice_source"] == "npm"
    assert not saved.get("requested")
    assert features.apply_requested(source="npm", env=env, data_root=tmp_path) is None
    env.done.wait(5)
    assert env.installs == 1


def test_no_request_means_nothing_happens(tmp_path):
    assert features.apply_requested(source="npm", env=FakeEnv(), data_root=tmp_path) is None
    assert not (tmp_path / "features.json").exists()


def test_an_already_enabled_feature_is_left_alone(tmp_path):
    (tmp_path / "features.json").write_text(json.dumps(
        {"schema": 1, "media": {"enabled": True, "requested": True, "choice_source": "cli"}}))
    assert features.apply_requested(source="npm", env=FakeEnv(), data_root=tmp_path) is None
    assert features.read_features(tmp_path)["media"]["choice_source"] == "cli"


def test_an_unreadable_file_is_a_no_op(tmp_path):
    (tmp_path / "features.json").write_text("{not json")
    assert features.apply_requested(source="npm", env=FakeEnv(), data_root=tmp_path) is None
    assert (tmp_path / "features.json").read_text() == "{not json"


def test_it_never_raises(tmp_path, monkeypatch):
    _request(tmp_path)

    def boom(**_):
        raise RuntimeError("no")

    monkeypatch.setattr(features, "enable_media", boom)
    assert features.apply_requested(source="npm", env=FakeEnv(), data_root=tmp_path) is None


def test_a_request_is_reported_as_requested(tmp_path):
    _request(tmp_path)
    assert features.media_requested(tmp_path) is True
    assert features.media_enabled(tmp_path) is False


@pytest.fixture()
def fresh_flag():
    features._reset_media_loaded()
    yield
    features._reset_media_loaded()


def test_restart_is_required_only_for_a_ready_env_the_daemon_has_not_loaded(tmp_path, fresh_flag):
    on = {"enabled": True, "env": {"state": "ready"}}
    assert features.restart_required(on) is True
    features.mark_media_loaded()
    assert features.restart_required(on) is False


def test_restart_is_not_required_when_off_or_not_ready(fresh_flag):
    assert features.restart_required({"enabled": False, "env": {"state": "ready"}}) is False
    assert features.restart_required({"enabled": True, "env": {"state": "installing"}}) is False


def test_a_daemon_that_starts_with_a_ready_env_needs_no_restart(tmp_path, fresh_flag):
    (tmp_path / "features.json").write_text(json.dumps({"media": {"enabled": True}}))
    env = FakeEnv("ready")
    features.note_started(env=env, data_root=tmp_path)
    assert features.restart_required({"enabled": True, "env": {"state": "ready"}}) is False


def test_an_install_finishing_after_start_needs_a_restart(tmp_path, fresh_flag):
    (tmp_path / "features.json").write_text(json.dumps({"media": {"enabled": True}}))
    features.note_started(env=FakeEnv("installing"), data_root=tmp_path)
    assert features.restart_required({"enabled": True, "env": {"state": "ready"}}) is True


def test_a_ready_env_with_the_feature_off_marks_nothing(tmp_path, fresh_flag):
    features.note_started(env=FakeEnv("ready"), data_root=tmp_path)
    assert features.restart_required({"enabled": True, "env": {"state": "ready"}}) is True
