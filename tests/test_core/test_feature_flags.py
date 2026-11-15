# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""4.2.30 feature flags: environment over file over default, never raising."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.feature_flags import (
    FLAGS,
    FeatureFlags,
    FlagState,
    env_name,
    read_features_file,
    resolve,
    write_flag,
)


def test_every_flag_defaults_off_so_4230_ships_dark() -> None:
    assert FLAGS and all(f.default is FlagState.OFF for f in FLAGS.values())
    assert all(f.workstream.startswith("WS-") for f in FLAGS.values())


def test_environment_beats_file_beats_default() -> None:
    fid = "truth.slots"
    assert resolve(fid, file_values={}, environ={}) is FlagState.OFF
    assert resolve(fid, file_values={fid: "shadow"}, environ={}) is FlagState.SHADOW
    assert resolve(fid, file_values={fid: "shadow"},
                   environ={env_name(fid): "on"}) is FlagState.ON
    assert env_name(fid) == "SLM_FEATURE_TRUTH_SLOTS"


@pytest.mark.parametrize("raw,state", [
    (True, FlagState.ON), (False, FlagState.OFF), ("1", FlagState.ON), ("yes", FlagState.ON),
    ("OFF", FlagState.OFF), (" Shadow ", FlagState.SHADOW), ("bogus", FlagState.OFF),
    (3, FlagState.OFF), (None, FlagState.OFF)])
def test_values_parse_or_fall_back(raw, state) -> None:
    assert resolve("truth.leases", file_values={"truth.leases": raw}, environ={}) is state


def test_shadow_on_an_unshadowable_flag_reads_off() -> None:
    fid = "truth.slots.auto_apply"
    assert not FLAGS[fid].shadowable
    assert resolve(fid, file_values={fid: "shadow"}, environ={}) is FlagState.OFF


def test_unknown_flag_is_off_not_an_error() -> None:
    assert resolve("no.such.flag", environ={}) is FlagState.OFF
    flags = FeatureFlags(file_values={}, environ={})
    assert flags.state("no.such.flag") is FlagState.OFF and not flags.observing("no.such.flag")


def test_enabled_and_observing() -> None:
    flags = FeatureFlags(file_values={"truth.slots": "shadow", "truth.leases": "on"}, environ={})
    assert flags.observing("truth.slots") and not flags.enabled("truth.slots")
    assert flags.observing("truth.leases") and flags.enabled("truth.leases")
    assert flags.as_dict()["truth.slots"] == "shadow"


def test_file_round_trip_and_corrupt_file(tmp_path) -> None:
    assert read_features_file(tmp_path) == {}
    write_flag(tmp_path, "truth.slots", FlagState.SHADOW)
    write_flag(tmp_path, "truth.leases", FlagState.ON)
    data = json.loads((tmp_path / "features.json").read_text())
    assert data == {"features": {"truth.leases": "on", "truth.slots": "shadow"}}
    assert FeatureFlags(tmp_path, environ={}).enabled("truth.leases")
    (tmp_path / "features.json").write_text("{not json")
    assert read_features_file(tmp_path) == {}
    (tmp_path / "features.json").write_text('["a list"]')
    assert read_features_file(tmp_path) == {}
    with pytest.raises(KeyError):
        write_flag(tmp_path, "no.such.flag", FlagState.ON)


# -- D-23: the kill switch --------------------------------------------------------------------

def test_the_kill_switch_turns_every_flag_off(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import FLAGS, FeatureFlags, FlagState, set_kill_switch, write_flag

    for fid in FLAGS:
        write_flag(tmp_path, fid, FlagState.ON)
    on = FeatureFlags(tmp_path, environ={})
    assert not on.killed and on.enabled("judge.ladder")
    set_kill_switch(tmp_path, True)
    off = FeatureFlags(tmp_path, environ={"SLM_FEATURE_JUDGE_LADDER": "on"})
    assert off.killed and not any(off.observing(fid) for fid in FLAGS)
    set_kill_switch(tmp_path, False)
    assert FeatureFlags(tmp_path, environ={}).enabled("judge.ladder")  # flags kept


def test_the_environment_can_pull_the_kill_switch(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag

    write_flag(tmp_path, "truth.slots", FlagState.ON)
    assert FeatureFlags(tmp_path, environ={"SLM_4230_OFF": "1"}).killed
    assert not FeatureFlags(tmp_path, environ={"SLM_4230_OFF": "1"}).observing("truth.slots")


def test_writing_a_flag_keeps_the_kill_switch(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, set_kill_switch, write_flag

    set_kill_switch(tmp_path, True)
    write_flag(tmp_path, "truth.slots", FlagState.ON)
    assert FeatureFlags(tmp_path, environ={}).killed
