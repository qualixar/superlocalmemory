# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# ruff: noqa: F811 - pytest fixtures imported by name
"""F-11 and F-16 (19 Nov 2026): the laptop-hang guards of the automatic download, and its memory.

F-11 (b): a pack download never starts while the machine is short of memory.
F-11 (c): the loop never looks while the daemon is busy (the idle gate).
F-16: failures and the back-off are kept in the data folder, so a daemon that restarts does not
try the download again at every start; "gave up" holds across restarts for that version.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_auto as ma
from superlocalmemory.runtimes import media_packs as mp
from tests.test_runtimes.test_media_auto import BIG, NO_NOT, Calls, FakeEnv, _installed, _tick, root  # noqa: F401


def _media_on(root):
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})


# -- F-11 (b): memory pressure, with the environment ready and a pack to fetch ----------------------

def test_no_pack_download_starts_while_the_machine_is_short_of_memory(root, monkeypatch):
    env = FakeEnv("ready")
    calls = Calls(env)
    _media_on(root)
    monkeypatch.setattr(ma, "_pressed", lambda: True)
    assert _tick(root, env, calls).label == "waiting_memory"
    assert calls.packs == [] and calls.media == []
    monkeypatch.setattr(ma, "_pressed", lambda: False)  # pressure over: the same look now starts it
    assert _tick(root, env, calls).label == "starting_pack"
    assert calls.packs == [("captions", "auto")]


# -- F-11 (c): the idle gate --------------------------------------------------------------------------

def test_the_loop_never_looks_while_the_daemon_is_busy(root):
    slept: list[float] = []

    class Stop(Exception):
        pass

    async def sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 6:
            raise Stop

    def tick_fn(_root):
        pytest.fail("a look was made while the daemon was busy")

    with pytest.raises(Stop):
        asyncio.run(ma.run(root, first_delay_s=1.0, idle=lambda: False, tick_fn=tick_fn, sleep=sleep))
    assert slept == [1.0] + [ma.POLL_S] * 5  # waited for start-up, then only waited for idle


def test_the_loop_looks_once_the_daemon_is_idle(root):
    answers = iter([ma.Outcome("done", None)])
    idle = iter([False, False, True])
    looks: list[object] = []

    async def sleep(_s):
        return None

    def tick_fn(data_root):
        looks.append(data_root)
        return next(answers)

    out = asyncio.run(ma.run(root, first_delay_s=0.0, idle=lambda: next(idle), tick_fn=tick_fn, sleep=sleep))
    assert out.label == "done" and looks == [root]


# -- F-16: the back-off and the give-up survive a restart ---------------------------------------------

class WallClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture()
def wall(monkeypatch):
    clock = WallClock()
    monkeypatch.setattr(ma, "_wall", clock)
    return clock


def _failing_media_tick(root, env, calls, state, now):
    state.part("media").open_attempt = True
    return _tick(root, env, calls, now=now, state=state)


def test_a_failure_is_kept_in_the_data_folder_and_a_restart_waits_it_out(root, wall):
    env = FakeEnv("failed", step="Can't reach the download server.")
    calls = Calls(env)
    _media_on(root)
    first = ma.AutoState()
    assert _failing_media_tick(root, env, calls, first, 1000.0).label == "retry_later"
    saved = json.loads((root / ma.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["parts"]["media"]["failures"] == 1 and saved["version"] == ma._version()
    assert "Can't reach" in saved["parts"]["media"]["reason"]

    # the daemon restarts: a new process has a new, empty state and a new monotonic clock
    restarted = ma.AutoState()
    wall.t += 120.0  # two minutes later
    out = _tick(root, env, calls, now=5.0, state=restarted)
    assert out.label == "retry_later" and calls.media == []
    assert restarted.part("media").failures == 1
    assert 400 < restarted.part("media").next_try - 5.0 <= 600  # what is left of the 10 minutes

    wall.t += 600.0  # the wait is over
    later = ma.AutoState()
    assert _tick(root, env, calls, now=9.0, state=later).label != "gave_up"
    assert later.part("media").failures == 1  # carried, and still not past the limit


def test_gave_up_holds_across_restarts_for_that_version(root, wall):
    env = FakeEnv()
    calls = Calls(env)
    state = ma.AutoState()
    for _ in range(ma.MAX_FAILURES):
        ma._fail(state, "media", 100.0, "no network")
    ma._save_state(root, state, 100.0)

    for _ in range(3):  # three restarts, however much time passes
        wall.t += 10 ** 7
        restarted = ma.AutoState()
        assert _tick(root, env, calls, now=1.0, state=restarted) == ma.Outcome("gave_up", None)
    assert calls.media == [] and calls.packs == []


def test_a_new_version_gets_a_fresh_chance(root, wall, monkeypatch):
    env = FakeEnv()
    calls = Calls(env)
    state = ma.AutoState()
    for _ in range(ma.MAX_FAILURES):
        ma._fail(state, "media", 100.0, "no network")
    ma._save_state(root, state, 100.0)
    monkeypatch.setattr(ma, "_version", lambda: "9.9.9")
    out = _tick(root, env, calls, now=1.0, state=ma.AutoState())
    assert out.label == "starting_media" and len(calls.media) == 1


def test_a_success_clears_the_record(root, wall):
    env = FakeEnv("failed", step="x")
    calls = Calls(env)
    _media_on(root)
    state = ma.AutoState()
    _failing_media_tick(root, env, calls, state, 1000.0)
    assert (root / ma.STATE_FILE).exists()
    env.state = "ready"
    _installed(root, "captions")
    _installed(root, "documents")
    wall.t += 10 ** 6
    assert _tick(root, env, calls, now=10 ** 6, state=state) == ma.Outcome("done", None)
    saved = json.loads((root / ma.STATE_FILE).read_text(encoding="utf-8"))
    assert saved["parts"] == {}


def test_a_pack_failure_is_kept_too(root, wall):
    env = FakeEnv("ready")
    calls = Calls(env)
    calls.enable_pack = lambda pack_id, **kw: {"refused": "Not enough free disk space."}
    _media_on(root)
    assert _tick(root, env, calls, state=ma.AutoState()).label == "retry_later"
    restarted = ma.AutoState()
    assert _tick(root, env, calls, now=3.0, state=restarted).label == "retry_later"
    assert restarted.part("captions").failures == 1


@pytest.mark.parametrize("junk", ["", "not json", "[]", '{"version": 3}', '{"version": "%s", "parts": 7}',
                                  '{"version": "%s", "parts": {"media": {"failures": "many"}}}'])
def test_an_unreadable_record_is_ignored_never_a_crash(root, wall, junk):
    junk = junk.replace("%s", ma._version())
    (root / ma.STATE_FILE).write_text(junk, encoding="utf-8")
    env = FakeEnv()
    calls = Calls(env)
    assert _tick(root, env, calls, state=ma.AutoState()).label == "starting_media"


def test_the_record_is_written_only_when_something_changed(root, wall):
    env = FakeEnv()
    calls = Calls(env)
    _tick(root, env, calls, state=ma.AutoState())
    assert not (root / ma.STATE_FILE).exists()  # nothing failed: nothing to remember


def test_a_record_with_a_part_nobody_knows_or_no_failures_adds_nothing(root, wall):
    record = {"version": ma._version(), "parts": {
        "evil": {"failures": 3, "reason": "x", "next_try_at": 1.0},
        "media": {"failures": 0, "reason": "", "next_try_at": 1.0}}}
    (root / ma.STATE_FILE).write_text(json.dumps(record), encoding="utf-8")
    state = ma.AutoState()
    _tick(root, FakeEnv(), Calls(FakeEnv()), state=state)
    assert "evil" not in state.parts and state.part("media").failures == 0


# -- F-16 (b): a download finishing late never switches back on what a person just switched off -------

class _LateCancelSource:
    """The last file lands, and only then is the cancel (the person's "turn off") seen."""

    model_id, revision = "x", "y"

    def __init__(self, set_cancel):
        self.set_cancel = set_cancel

    def fetch(self, python, dest, *, progress, cancel=None):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        if self.set_cancel:
            cancel.set()
        return True, "", ""


def _download(root, *, cancelled):
    import threading

    from tests.test_runtimes.test_media_packs import _Env

    mp._set_job("captions", state="downloading", progress=0.0, step="")
    mp._download(mp.CAPTIONS, root, _Env(root / "runtimes" / "media"),
                 _LateCancelSource(cancelled), threading.Event())


def test_a_pack_switched_off_while_its_last_file_lands_stays_off(root):
    from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag

    write_flag(root, mp.CAPTIONS.flag_id, FlagState.OFF)  # what `slm media pack disable` writes
    _download(root, cancelled=True)
    assert FeatureFlags(root).enabled(mp.CAPTIONS.flag_id) is False


def test_a_download_nobody_cancelled_still_switches_its_pack_on(root):
    from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag

    write_flag(root, mp.CAPTIONS.flag_id, FlagState.OFF)
    _download(root, cancelled=False)
    assert FeatureFlags(root).enabled(mp.CAPTIONS.flag_id) is True
