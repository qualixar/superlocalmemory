# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Turning the optional picture models on and off (WS-8, D-19): sizes first, download only on enable."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.feature_flags import FeatureFlags, set_kill_switch
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_packs as mp
from superlocalmemory.runtimes.managed_env import EnvStatus
from superlocalmemory.runtimes.model_sources import LocalDirSource


class _Env:
    def __init__(self, root, state="ready"):
        self.root, self._state = root, state

    def status(self):
        return EnvStatus(state=self._state, progress=1.0, step="")

    def python(self):
        import sys
        from pathlib import Path

        return Path(sys.executable)


class _BoomSource:
    model_id, revision = "x", "y"

    def fetch(self, *a, **k):
        raise AssertionError("nothing may be downloaded here")


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "data"
    r.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(r))
    monkeypatch.setenv("SLM_MEDIA_ALLOW_LOW_RAM", "1")
    for name in ("SLM_FEATURE_MEDIA_CAPTIONS", "SLM_FEATURE_MEDIA_DOCUMENT_PACK", "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    mp._JOBS.clear()
    mp._THREADS.clear()
    return r


@pytest.fixture()
def media_on(root):
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    return _Env(root / "runtimes" / "media")


@pytest.fixture()
def fake_model(tmp_path):
    src = tmp_path / "model-files"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"0" * 64)
    return LocalDirSource(src)


# -- what is pinned and what a person is told -----------------------------------------

def test_the_pins_and_the_words_for_the_size():
    assert mp.CAPTIONS.repo == "florence-community/Florence-2-base"
    assert mp.CAPTIONS.revision == "00921df66db728a9ceb750f5eca43e5c203a2051"
    assert mp.DOCUMENTS.repo == "PaddlePaddle/PaddleOCR-VL-1.6"
    assert len(mp.DOCUMENTS.revision) == 40 and mp.DOCUMENTS.revision != mp.CAPTIONS.revision
    assert mp.download_words(mp.CAPTIONS) == "Describe pictures: a 0.5 GB download"
    assert mp.download_words(mp.DOCUMENTS) == "Read scanned documents better: a 1.9 GB download"
    assert (mp.CAPTIONS.licence, mp.DOCUMENTS.licence) == ("MIT", "Apache-2.0")
    assert mp.CAPTIONS.native_class == "Florence2ForConditionalGeneration"


def test_every_part_is_off_by_default_and_creates_nothing(root):
    shown = mp.status_all(root, env=_Env(root / "runtimes" / "media"))
    assert [p["pack"] for p in shown] == ["captions", "documents", "regions"]
    assert all(p["state"] == "off" and p["enabled"] is False for p in shown)
    assert not any(root.iterdir())  # reading the state wrote and created nothing


def test_an_unknown_part_is_refused_in_words():
    with pytest.raises(ValueError, match="captions"):
        mp.get("nonsense")


# -- turning on -----------------------------------------------------------------------

def test_enable_downloads_only_on_enable_and_then_switches_on(root, media_on, fake_model):
    assert mp.is_active("captions", root) is False
    got = mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    assert got["state"] == "on" and got["installed"] is True and got["enabled"] is True
    assert (mp.model_dir(mp.CAPTIONS, root) / "config.json").is_file()
    assert FeatureFlags(root).enabled("media.captions") and mp.is_active("captions", root)
    info = json.loads((mp.pack_dir(mp.CAPTIONS, root) / mp.READY_FILE).read_text())
    assert info["revision"] == mp.CAPTIONS.revision and info["repo"] == mp.CAPTIONS.repo


def test_a_flag_alone_never_downloads_and_never_counts_as_on(root, media_on, monkeypatch):
    monkeypatch.setenv("SLM_FEATURE_MEDIA_CAPTIONS", "on")
    assert FeatureFlags(root).enabled("media.captions")
    assert mp.is_active("captions", root) is False      # the files are not here
    assert mp.pack_client("captions", root, env=media_on) is None   # and no worker is made for them


def test_status_and_pack_client_never_touch_the_network(root, media_on, monkeypatch):
    from superlocalmemory.runtimes import model_sources

    monkeypatch.setattr(model_sources.HuggingFaceSource, "fetch",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    mp.status_all(root, env=media_on)
    assert mp.pack_client("documents", root, env=media_on) is None


def test_a_failed_download_leaves_the_flag_off_and_says_so_in_words(root, media_on):
    class Failing:
        model_id, revision = "x", "y"

        def fetch(self, python, dest, *, progress, cancel=None):
            return False, "network", "Connection reset by peer 10.0.0.9"

    got = mp.enable("captions", data_root=root, env=media_on, model_source=Failing(), background=False)
    assert got["state"] == "failed" and "connection" in got["error"].lower()
    assert "10.0.0.9" not in json.dumps(got)  # the detail goes to the log, not to the person
    assert not FeatureFlags(root).enabled("media.captions") and not mp.installed(mp.CAPTIONS, root)


def test_a_download_that_raises_is_a_plain_failure(root, media_on):
    class Raising:
        model_id, revision = "x", "y"

        def fetch(self, *a, **k):
            raise RuntimeError("secret detail")

    got = mp.enable("captions", data_root=root, env=media_on, model_source=Raising(), background=False)
    assert got["state"] == "failed" and "secret detail" not in json.dumps(got)


def test_enable_is_refused_without_images_on(root):
    got = mp.enable("captions", data_root=root, env=_Env(root / "x"), model_source=_BoomSource(), background=False)
    assert got["refused"] == mp.NEEDS_MEDIA_WORDS and got["state"] == "off"
    assert not (root / "runtimes").exists()


def test_enable_is_refused_while_the_environment_is_not_ready(root, media_on):
    got = mp.enable("captions", data_root=root, env=_Env(media_on.root, state="installing"),
                    model_source=_BoomSource(), background=False)
    assert got["refused"] == mp.NEEDS_MEDIA_WORDS


def test_the_kill_switch_refuses_and_turns_a_running_part_off(root, media_on, fake_model):
    mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    assert mp.is_active("captions", root)
    set_kill_switch(root, True)
    assert mp.is_active("captions", root) is False and mp.pack_client("captions", root, env=media_on) is None
    got = mp.enable("documents", data_root=root, env=media_on, model_source=_BoomSource(), background=False)
    assert got["refused"] == mp.KILLED_WORDS


def test_enable_is_refused_without_room_on_the_disk(root, media_on, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 9, 100))
    got = mp.enable("documents", data_root=root, env=media_on, model_source=_BoomSource(), background=False)
    assert "free disk space" in got["refused"] and "GB" in got["refused"]


def test_a_small_computer_is_refused_with_the_standard_words(root, media_on, monkeypatch):
    monkeypatch.delenv("SLM_MEDIA_ALLOW_LOW_RAM")
    from superlocalmemory.runtimes import managed_env

    monkeypatch.setattr(managed_env, "_ram_bytes", lambda: 8 * 1024 ** 3)
    got = mp.enable("captions", data_root=root, env=media_on, model_source=_BoomSource(), background=False)
    assert "16 GB" in got["refused"]


# -- turning off ------------------------------------------------------------------------

def test_disable_switches_off_and_keeps_the_files_unless_asked(root, media_on, fake_model):
    mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    got = mp.disable("captions", data_root=root, env=media_on)
    assert got["state"] == "ready" and got["installed"] and not got["enabled"]
    again = mp.enable("captions", data_root=root, env=media_on, model_source=_BoomSource(), background=False)
    assert again["state"] == "on"          # the files were kept: no second download
    gone = mp.disable("captions", remove_files=True, data_root=root, env=media_on)
    assert gone["state"] == "off" and not mp.pack_dir(mp.CAPTIONS, root).exists()


def test_disable_stops_a_loaded_worker(root, media_on, fake_model, monkeypatch):
    from superlocalmemory.runtimes import worker_client

    mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    stopped = []

    class _C:
        model_id, root = mp.CAPTIONS.repo, media_on.root

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(worker_client, "live_pack_clients", lambda: [_C()])
    mp.disable("captions", data_root=root, env=media_on)
    assert stopped == [True]


def test_a_cancelled_download_is_reported_not_installed(root, media_on):
    import threading

    started, gate = threading.Event(), threading.Event()

    class Slow:
        model_id, revision = "x", "y"

        def fetch(self, python, dest, *, progress, cancel=None):
            started.set()
            gate.wait(5)
            return (False, "cancelled", "stopped") if cancel is not None and cancel.is_set() else (True, "", "")

    got = mp.enable("captions", data_root=root, env=media_on, model_source=Slow(), background=True)
    assert got["state"] == "downloading"
    started.wait(5)
    again = mp.enable("captions", data_root=root, env=media_on, model_source=_BoomSource(), background=True)
    assert again["state"] == "downloading"     # one download at a time: the second call starts nothing
    threading_cancel = mp._CANCELS["captions"]
    threading_cancel.set()
    gate.set()
    mp._THREADS["captions"].join(5)
    assert mp.status("captions", root, env=media_on)["state"] == "failed"
    assert not mp.installed(mp.CAPTIONS, root)


# -- the third part: no model, only a switch -----------------------------------------------

def test_regions_is_a_switch_with_nothing_to_download(root, media_on):
    got = mp.enable("regions", data_root=root, env=media_on)
    assert got["state"] == "on" and got["download_words"] == "No download"
    assert FeatureFlags(root).enabled("media.regions")
    assert mp.flag_enabled("media.regions", root)
    assert mp.disable("regions", data_root=root, env=media_on)["state"] == "off"


def test_regions_needs_images_on_and_respects_the_kill_switch(root):
    assert mp.enable("regions", data_root=root, env=_Env(root / "x"))["refused"] == mp.NEEDS_MEDIA_WORDS
    set_kill_switch(root, True)
    assert mp.enable("regions", data_root=root, env=_Env(root / "x"))["refused"] == mp.KILLED_WORDS


# -- the flags file is shared with images and documents -------------------------------------

def test_turning_images_on_keeps_the_4230_flags_and_the_kill_switch(root):
    """features.json holds both; rewriting it from the media keys alone erased the flags (found in WS-8)."""
    from superlocalmemory.core.feature_flags import FlagState, write_flag

    write_flag(root, "portability.import", FlagState.ON)
    set_kill_switch(root, True)
    feat.record_media_request(source="api", data_root=root)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    raw = json.loads((root / "features.json").read_text())
    assert raw["features"] == {"portability.import": "on"} and raw["kill_switch"] is True
    assert raw["media"]["enabled"] is True
    feat.clear_engine_upgrade_request(root)           # popping an owned key still works
    assert "engine_upgrade" not in json.loads((root / "features.json").read_text())


# -- what counts as installed ---------------------------------------------------------------

def test_files_of_another_revision_or_without_the_marker_are_not_installed(root, media_on, fake_model):
    mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    marker = mp.pack_dir(mp.CAPTIONS, root) / mp.READY_FILE
    info = json.loads(marker.read_text())
    marker.write_text(json.dumps({**info, "revision": "0" * 40}))
    assert mp.installed(mp.CAPTIONS, root) is False and mp.is_active("captions", root) is False
    marker.write_text(json.dumps({**info, "repo": "someone/else"}))
    assert mp.installed(mp.CAPTIONS, root) is False
    marker.write_text("not json")
    assert mp.installed(mp.CAPTIONS, root) is False
    marker.write_text(json.dumps(info))
    assert mp.installed(mp.CAPTIONS, root) is True
    marker.unlink()
    assert mp.installed(mp.CAPTIONS, root) is False                       # no marker: a half download never counts
    assert mp.status("captions", root, env=media_on)["state"] == "off"


def test_pack_client_is_the_one_worker_per_part_with_its_folder_and_short_idle(root, media_on, fake_model):
    mp.enable("documents", data_root=root, env=media_on, model_source=fake_model, background=False)
    client = mp.pack_client("documents", root, env=media_on)
    assert client is mp.pack_client("documents", root, env=media_on)
    assert client.role == "document" and client.model_id == mp.DOCUMENTS.repo and client.idle_s == mp.PACK_IDLE_S
    assert client._model_dir == mp.model_dir(mp.DOCUMENTS, root)
    assert mp.pack_client("captions", root, env=media_on) is None         # the other part is not on


def test_the_state_words_follow_the_files_and_the_flag(root, media_on, fake_model):
    assert mp.status("captions", root, env=media_on)["state"] == "off"
    mp.enable("captions", data_root=root, env=media_on, model_source=fake_model, background=False)
    assert mp.status("captions", root, env=media_on)["state"] == "on"
    mp.disable("captions", data_root=root, env=media_on)
    assert mp.status("captions", root, env=media_on)["state"] == "ready"
    shown = mp.status("captions", root, env=media_on)
    assert shown["model"] == mp.CAPTIONS.repo and shown["revision"] == mp.CAPTIONS.revision
    assert shown["size_words"] == "0.5 GB" and shown["media_ready"] is True and shown["killed"] is False


def test_turning_a_part_off_stops_a_download_in_progress(root, media_on):
    import threading
    import time

    started = threading.Event()

    class Waiting:
        model_id, revision = "x", "y"

        def fetch(self, python, dest, *, progress, cancel=None):
            started.set()
            end = time.time() + 3
            while time.time() < end:
                if cancel is not None and cancel.is_set():
                    return False, "cancelled", "stopped by the person"
                time.sleep(0.01)
            return True, "", ""                      # nobody stopped it: it would install

    mp.enable("captions", data_root=root, env=media_on, model_source=Waiting(), background=True)
    started.wait(5)
    began = time.time()
    got = mp.disable("captions", data_root=root, env=media_on)
    assert time.time() - began < 2.5 and got["installed"] is False and got["state"] == "off"
    assert not FeatureFlags(root).enabled("media.captions")
