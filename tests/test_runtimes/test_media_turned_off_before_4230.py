# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-05 (19 Nov 2026): a person who turned pictures off on 4.2.19 stays off.

``media.opted_out`` is a key 4.2.30 added. 4.2.19's ``slm media disable`` wrote only
``{"enabled": false, "enabled_at": ..., "choice_source": "cli"}``, so the automatic set-up read
"nobody chose" and turned pictures back on three minutes after the first idle start.
The earlier choice is in the file and is read now.
"""

from __future__ import annotations

import json

import pytest

from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_auto as ma
from superlocalmemory.runtimes import media_packs as mp
from tests.test_runtimes.test_media_auto import BIG, Calls, FakeEnv, _tick

# Exactly what 4.2.19 left in features.json after enable, then disable (v4.2.19 disable_media).
FEATURES_4219_DISABLED = {
    "schema": 1,
    "media": {"enabled": False, "enabled_at": "2026-11-02T09:15:00+00:00", "choice_source": "cli"},
    "sources": {"enabled": False},
}


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "slm"
    r.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    for name in ("SLM_FEATURE_MEDIA_CAPTIONS", "SLM_FEATURE_MEDIA_DOCUMENT_PACK", "SLM_4230_OFF",
                 "SLM_MEDIA_ALLOW_LOW_RAM", "SLM_NO_AUTO_DOWNLOAD"):
        monkeypatch.delenv(name, raising=False)
    mp._JOBS.clear()
    mp._THREADS.clear()
    monkeypatch.setattr(ma, "_pressed", lambda: False)
    monkeypatch.setattr(ma, "_STATE", ma.AutoState())
    return r


def _write(root, data):
    (root / "features.json").write_text(json.dumps(data), encoding="utf-8")


def test_enabled_then_disabled_on_4219_reads_as_opted_out(root):
    _write(root, FEATURES_4219_DISABLED)
    assert feat.read_features(root)["media"].get("opted_out") is None  # the key is not in the file
    assert feat.media_opted_out(root, env=FakeEnv()) is True


def test_the_automatic_set_up_never_turns_a_4219_off_back_on(root):
    _write(root, FEATURES_4219_DISABLED)
    env = FakeEnv()
    calls = Calls(env)
    outcome = _tick(root, env, calls, ram=BIG)
    assert outcome.label == "turned_off"
    assert calls.media == [] and calls.packs == []  # no download was asked for
    assert feat.media_enabled(root) is False


def test_installed_and_then_disabled_also_counts(root):
    # The environment is on the disk, the switch is off and nothing records why: someone turned it off.
    _write(root, {"schema": 1, "media": {"enabled": False}, "sources": {"enabled": False}})
    env = FakeEnv(state="ready")
    calls = Calls(env)
    assert feat.media_opted_out(root, env=env) is True
    assert _tick(root, env, calls, ram=BIG).label == "turned_off"
    assert calls.media == []


def test_a_fresh_install_is_not_opted_out(root):
    # No file at all, and an installer request that nobody has acted on: neither is a choice to be off.
    env = FakeEnv()
    assert feat.media_opted_out(root, env=env) is False
    feat.record_media_request(source="npm", data_root=root)
    assert feat.media_opted_out(root, env=env) is False
    calls = Calls(env)
    assert _tick(root, env, calls, ram=BIG).label == "starting_media"
    assert len(calls.media) == 1


def test_a_person_who_turned_it_on_is_not_opted_out(root):
    _write(root, {"schema": 1, "media": {"enabled": True, "enabled_at": "2026-11-02T09:15:00+00:00",
                                         "choice_source": "cli"}, "sources": {"enabled": False}})
    assert feat.media_opted_out(root, env=FakeEnv()) is False


def test_the_new_key_still_wins_and_enable_clears_it(root, monkeypatch):
    _write(root, {"schema": 1, "media": {"enabled": False, "opted_out": True}})
    assert feat.media_opted_out(root, env=FakeEnv()) is True
    monkeypatch.setattr(feat, "_start_install", lambda managed: None)
    feat.enable_media(source="cli", data_root=root, env=FakeEnv())
    assert feat.media_opted_out(root, env=FakeEnv()) is False


def test_a_failed_enable_puts_the_earlier_state_back_and_is_not_read_as_a_choice(root, monkeypatch):
    # No SQLite extensions: enable fails and rolls back. That is not "someone turned it off".
    from superlocalmemory import media as media_pkg

    def refuse(**_kw):
        raise media_pkg.MediaVectorsUnavailable("no extensions")

    monkeypatch.setattr(media_pkg, "open_media_store", refuse)
    feat.enable_media(source="auto", data_root=root, env=FakeEnv())
    assert feat.media_enabled(root) is False
    assert feat.media_opted_out(root, env=FakeEnv()) is False
    assert feat.read_features(root)["media"].get("enabled_at") is None


def test_a_failed_enable_does_not_erase_an_earlier_opt_out(root, monkeypatch):
    from superlocalmemory import media as media_pkg

    _write(root, {"schema": 1, "media": {"enabled": False, "opted_out": True}})
    monkeypatch.setattr(media_pkg, "open_media_store",
                        lambda **_kw: (_ for _ in ()).throw(media_pkg.MediaVectorsUnavailable("x")))
    feat.enable_media(source="cli", data_root=root, env=FakeEnv())
    assert feat.media_opted_out(root, env=FakeEnv()) is True


def test_an_installer_request_beside_a_ready_environment_is_not_an_off(root):
    _write(root, {"schema": 1, "media": {"enabled": False, "requested": True, "choice_source": "npm"}})
    assert feat.media_opted_out(root, env=FakeEnv(state="ready")) is False
