"""The automatic model set-up on a computer with 16 GB or more, and what never happens (4.2.30)."""

from __future__ import annotations

import asyncio
import json

import pytest

from superlocalmemory.core.feature_flags import FlagState, set_kill_switch, write_flag
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_auto as ma
from superlocalmemory.runtimes import media_packs as mp
from superlocalmemory.runtimes.managed_env import EnvStatus

GIB = 1024 ** 3
BIG, SMALL = 32 * GIB, 8 * GIB
NO_NOT = {}  # an empty environment: SLM_NO_AUTO_DOWNLOAD unset


class FakeEnv:
    def __init__(self, state="not_installed", error_kind="", step="", disk_ok=True):
        self.state, self.error_kind, self.step, self.disk_ok = state, error_kind, step, disk_ok

    def status(self):
        return EnvStatus(state=self.state, progress=0.4, step=self.step, error_kind=self.error_kind)

    def precheck(self):
        return {"disk_ok": self.disk_ok}


class Calls:
    """Records every download the loop asks for; nothing real is ever started."""

    def __init__(self, env):
        self.env, self.media, self.packs = env, [], []

    def enable_media(self, **kw):
        self.media.append(kw)
        feat._write_features(kw["data_root"], {**feat.read_features(kw["data_root"]),
                                               "media": {"enabled": True, "choice_source": kw["source"]}})
        return {"enabled": True}

    def enable_pack(self, pack_id, **kw):
        self.packs.append((pack_id, kw["source"]))
        return {"state": "downloading"}


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


def _tick(root, env, calls, *, ram=BIG, now=1000.0, environ=NO_NOT, state=None):
    return ma.tick(root, ram_bytes=ram, env=env, now=now, environ=environ, state=state or ma._STATE,
                   enable_media=calls.enable_media, enable_pack=calls.enable_pack)


def _installed(root, pack_id):
    spec = mp.get(pack_id)
    mp.pack_dir(spec, root).mkdir(parents=True, exist_ok=True)
    mp.model_dir(spec, root).mkdir(parents=True, exist_ok=True)
    (mp.pack_dir(spec, root) / mp.READY_FILE).write_text(json.dumps({"repo": spec.repo, "revision": spec.revision}))


# -- decision table: 16 GB+ ---------------------------------------------------------------------------

def test_on_a_big_machine_the_first_look_starts_images_and_documents_as_auto(root):
    env = FakeEnv()
    calls = Calls(env)
    out = _tick(root, env, calls)
    assert out == ma.Outcome("starting_media", ma.POLL_S)
    assert calls.media == [{"source": "auto", "env": env, "data_root": root}] and calls.packs == []


def test_while_the_environment_installs_nothing_else_starts(root):
    env, calls = FakeEnv("installing"), None
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    assert _tick(root, env, calls).label == "downloading_media"
    assert calls.media == [] and calls.packs == []


def test_once_ready_the_packs_download_one_at_a_time_captions_first(root):
    env = FakeEnv("ready")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    assert _tick(root, env, calls).label == "starting_pack"
    assert calls.packs == [("captions", "auto")] and calls.media == []
    _installed(root, "captions")
    assert _tick(root, env, calls).label == "starting_pack"
    assert calls.packs[-1] == ("documents", "auto")


def test_a_running_pack_download_blocks_the_next_one(root):
    env = FakeEnv("ready")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    mp._set_job("captions", state="downloading", progress=0.5, step="x")
    out = _tick(root, env, calls)
    assert out.label == "downloading_pack" and calls.packs == []


def test_when_everything_is_here_the_loop_ends(root):
    env = FakeEnv("ready")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    _installed(root, "captions")
    _installed(root, "documents")
    assert _tick(root, env, calls) == ma.Outcome("done", None) and calls.packs == []


# -- decision table: below 16 GB and unknown ------------------------------------------------------------

@pytest.mark.parametrize("ram", [SMALL, 0, 15 * GIB - 1])
def test_below_16_gb_or_unknown_nothing_starts_by_itself(root, ram):
    env = FakeEnv()
    calls = Calls(env)
    assert _tick(root, env, calls, ram=ram) == ma.Outcome("one_click", None)
    assert calls.media == [] and calls.packs == []


def test_the_developer_override_makes_a_small_machine_automatic(root):
    env = FakeEnv()
    calls = Calls(env)
    out = _tick(root, env, calls, ram=SMALL, environ={"SLM_MEDIA_ALLOW_LOW_RAM": "1"})
    assert out.label == "starting_media"


# -- decision table: turned off ------------------------------------------------------------------------

def test_a_person_who_turned_images_off_is_never_overruled(root):
    env = FakeEnv()
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": False, "opted_out": True}})
    assert _tick(root, env, calls).label == "turned_off"
    assert calls.media == [] and calls.packs == []


@pytest.mark.parametrize("how", ["file", "env"])
def test_a_pack_turned_off_stays_off_while_the_other_downloads(root, monkeypatch, how):
    env = FakeEnv("ready")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    if how == "file":
        write_flag(root, "media.captions", FlagState.OFF)  # what `slm media pack disable` writes
        environ = NO_NOT
    else:
        environ = {"SLM_FEATURE_MEDIA_CAPTIONS": "off"}
        monkeypatch.setenv("SLM_FEATURE_MEDIA_CAPTIONS", "off")
    _tick(root, env, calls, environ=environ)
    assert calls.packs == [("documents", "auto")]


def test_disable_pack_writes_the_off_the_loop_respects(root, monkeypatch):
    env = FakeEnv("ready")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    mp.disable("captions", data_root=root, env=env)
    mp.disable("documents", data_root=root, env=env)
    assert _tick(root, env, calls) == ma.Outcome("done", None) and calls.packs == []


def test_the_kill_switch_stops_everything_and_nothing_is_undone(root):
    env = FakeEnv()
    calls = Calls(env)
    set_kill_switch(root, True)
    assert _tick(root, env, calls).label == "killed"
    assert calls.media == [] and calls.packs == []
    assert json.loads((root / "features.json").read_text())["kill_switch"] is True


def test_the_env_guard_turns_the_loop_off(root):
    env = FakeEnv()
    calls = Calls(env)
    assert _tick(root, env, calls, environ={"SLM_NO_AUTO_DOWNLOAD": "1"}) == ma.Outcome("disabled", None)
    assert calls.media == [] and calls.packs == []


def test_an_install_the_person_cancelled_is_not_retried(root):
    env = FakeEnv("failed", error_kind="cancelled", step="Setup was cancelled.")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    assert _tick(root, env, calls).label == "stopped_by_you" and calls.media == []


# -- never in a tight loop, never under pressure, never without disk ----------------------------------

def test_a_failed_download_is_retried_later_with_a_growing_wait_and_then_given_up(root):
    env = FakeEnv("failed", step="Can't reach the download server.")
    calls = Calls(env)
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    state = ma.AutoState()
    state.part("media").open_attempt = True
    now = 1000.0
    waits = []
    for _ in range(ma.MAX_FAILURES - 1):
        out = _tick(root, env, calls, now=now, state=state)
        assert out.label == "retry_later" and out.wait_s <= ma.WAIT_S
        waits.append(state.part("media").next_try - now)
        assert _tick(root, env, calls, now=now + 1, state=state).label == "retry_later"  # no tight loop
        assert calls.media == []
        now = state.part("media").next_try
        state.part("media").open_attempt = True  # the retry is attempted again and fails again
    assert waits == [600.0, 1200.0, 2400.0, 4800.0, 9600.0][:len(waits)]
    assert _tick(root, env, calls, now=now, state=state) == ma.Outcome("gave_up", None)
    assert ma.backoff_s(99) == ma.RETRY_MAX_S


def test_a_part_that_gave_up_stays_given_up_for_this_run_whatever_the_clock_says(root):
    env = FakeEnv()  # not installed, nothing running: only the give-up memory stops a new start
    calls = Calls(env)
    state = ma.AutoState()
    state.part("media").failures = ma.MAX_FAILURES
    state.part("media").next_try = 0.0  # the back-off long over
    assert _tick(root, env, calls, now=10 ** 9, state=state) == ma.Outcome("gave_up", None)
    assert calls.media == [] and calls.packs == []


def test_the_reason_stays_readable_in_plain_words(root):
    env = FakeEnv("failed", step="Can't reach the download server.")
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    state = ma.AutoState()
    state.part("media").open_attempt = True
    ma.tick(root, ram_bytes=BIG, env=env, now=5.0, environ=NO_NOT, state=ma._STATE,
            enable_media=Calls(env).enable_media, enable_pack=Calls(env).enable_pack)
    ma._STATE.part("media").open_attempt = True
    ma.tick(root, ram_bytes=BIG, env=env, now=6.0, environ=NO_NOT, state=ma._STATE,
            enable_media=Calls(env).enable_media, enable_pack=Calls(env).enable_pack)
    assert ma.last_reason("media") == "Can't reach the download server."


def test_nothing_starts_while_the_machine_is_short_of_memory(root, monkeypatch):
    monkeypatch.setattr(ma, "_pressed", lambda: True)
    env = FakeEnv()
    calls = Calls(env)
    assert _tick(root, env, calls).label == "waiting_memory"
    assert calls.media == [] and calls.packs == []


def test_nothing_starts_without_the_disk_for_it(root):
    env = FakeEnv(disk_ok=False)
    calls = Calls(env)
    out = _tick(root, env, calls)
    assert out.label == "retry_later" and calls.media == []
    assert "disk" in ma.last_reason("media")


def test_a_refused_pack_start_is_a_failure_with_its_reason(root):
    env = FakeEnv("ready")
    calls = Calls(env)
    calls.enable_pack = lambda pack_id, **kw: {"refused": "Not enough free disk space: this needs about 1.0 GB."}
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    assert _tick(root, env, calls).label == "retry_later"
    assert "free disk space" in ma.last_reason("captions")


def test_the_look_never_raises(root, monkeypatch):
    env = FakeEnv()
    calls = Calls(env)

    def boom(**kw):
        raise RuntimeError("no")

    calls.enable_media = boom
    assert _tick(root, env, calls) == ma.Outcome("error", ma.WAIT_S)


# -- the daemon loop ------------------------------------------------------------------------------------

def test_the_loop_waits_for_start_up_and_for_idle_then_looks_until_done(root):
    slept, looks = [], []
    answers = iter([ma.Outcome("starting_media", 20.0), ma.Outcome("done", None)])

    async def sleep(s):
        slept.append(s)

    def tick_fn(data_root):
        looks.append(data_root)
        return next(answers)

    idle = iter([False, True, True])
    out = asyncio.run(ma.run(root, first_delay_s=180.0, idle=lambda: next(idle), tick_fn=tick_fn, sleep=sleep))
    assert out.label == "done" and slept[0] == 180.0 and ma.POLL_S in slept and 20.0 in slept
    assert looks == [root, root]


def test_the_loop_does_nothing_at_all_when_the_guard_is_set(root, monkeypatch):
    monkeypatch.setenv("SLM_NO_AUTO_DOWNLOAD", "1")
    out = asyncio.run(ma.run(root, tick_fn=lambda *_: pytest.fail("must not look"),
                             sleep=lambda s: pytest.fail("must not wait")))
    assert out == ma.Outcome("disabled", None)


# -- plain words ------------------------------------------------------------------------------------------

def test_the_words_a_person_reads():
    base = {"pack": "captions", "enabled": True, "installed": False, "killed": False, "size_words": "0.5 GB"}
    assert ma.pack_words({**base, "state": "downloading", "progress": 0.4}, "auto") == \
        "Downloading the picture description model, 40%"
    assert ma.pack_words({**base, "state": "on", "installed": True}, "auto") == "Ready"
    assert ma.pack_words({**base, "state": "off", "enabled": False}, "auto") == "Turned off by you"
    assert "One click to download the picture description model (0.5 GB)" in ma.pack_words(
        {**base, "state": "off"}, "one_click")
    assert "by itself" in ma.pack_words({**base, "state": "off"}, "auto")
    failed = ma.pack_words({**base, "state": "failed", "error": "The disk is full."}, "auto")
    assert failed == "The disk is full. It will try again later."
    assert ma.pack_words({**base, "killed": True, "state": "off"}, "auto").startswith("Switched off")
    assert ma.media_words({"enabled": True, "env_state": "installing", "progress": 0.25}, "auto") == \
        "Downloading the picture model, 25%"
    assert ma.media_words({"enabled": True, "env_state": "ready"}, "auto") == "Ready"
    assert ma.media_words({"enabled": False, "env_state": "not_installed"}, "auto", opted_out=True) == \
        "Turned off by you"


def test_describe_and_pack_status_carry_the_same_words_and_the_small_machine_warning(root, monkeypatch):
    from superlocalmemory.runtimes import managed_env

    monkeypatch.setattr(managed_env, "_ram_bytes", lambda: SMALL)
    env = FakeEnv("ready")
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    got = ma.describe(root, env=env)
    assert got["mode"] == "one_click" and got["warning"].startswith("This computer has 8.0 GB of memory")
    status = mp.status("captions", root, env=env)
    assert status["words"] == next(p["words"] for p in got["parts"] if p["pack"] == "captions")
    assert status["mode"] == "one_click" and status["warning"] == got["warning"]
