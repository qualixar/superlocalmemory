# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-28: the one off switch every surface calls."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.feature_flags import KILL_ENV, FeatureFlags, FlagState, set_kill_switch
from superlocalmemory.outcomes import disclosure, switch

INTERVENTIONS = ("outcomes.holdout", "outcomes.swaps")


@pytest.fixture(autouse=True)
def _shipped_defaults(monkeypatch):
    """The suite forces these two off; here we want what a user gets."""
    for name in ("SLM_FEATURE_OUTCOMES_HOLDOUT", "SLM_FEATURE_OUTCOMES_SWAPS",
                 "SLM_FEATURE_OUTCOMES_RECEIPTS", "SLM_FEATURE_OUTCOMES_USAGE",
                 "SLM_FEATURE_DELIVERY_NEED_GATE", KILL_ENV):
        monkeypatch.delenv(name, raising=False)


def _states(home):
    flags = FeatureFlags(home)
    return {f: flags.state(f).value for f in (*INTERVENTIONS, "outcomes.receipts",
                                              "outcomes.usage", "delivery.need_gate")}


def test_a_fresh_install_measures_by_default(tmp_path) -> None:
    assert _states(tmp_path) == {"outcomes.holdout": "on", "outcomes.swaps": "on",
                                 "outcomes.receipts": "on", "outcomes.usage": "on",
                                 "delivery.need_gate": "off"}
    assert switch.status(tmp_path)["measuring"] is True


def test_off_turns_the_interventions_and_usage_detection_off_and_nothing_else(tmp_path) -> None:
    switch.set_measurement(tmp_path, False, channel="cli")
    assert _states(tmp_path) == {"outcomes.holdout": "off", "outcomes.swaps": "off",
                                 "outcomes.receipts": "on", "outcomes.usage": "off",
                                 "delivery.need_gate": "off"}
    assert switch.status(tmp_path)["measuring"] is False


def test_off_leaves_the_need_gate_as_it_was_set(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import write_flag

    write_flag(tmp_path, "delivery.need_gate", FlagState.ON)
    switch.set_measurement(tmp_path, False, channel="dashboard")
    after = _states(tmp_path)
    assert after["outcomes.usage"] == "off" and after["delivery.need_gate"] == "on"


def test_the_notice_gates_nothing_every_default_measure_runs_while_it_is_pending(tmp_path) -> None:
    flags = FeatureFlags(tmp_path, environ={})
    assert disclosure.pending(tmp_path) is True
    for flag_id in ("outcomes.receipts", "outcomes.holdout", "outcomes.swaps", "outcomes.usage"):
        assert flags.enabled(flag_id), flag_id


def test_on_turns_the_interventions_back_on(tmp_path) -> None:
    switch.set_measurement(tmp_path, False, channel="cli")
    switch.set_measurement(tmp_path, True, channel="cli")
    assert _states(tmp_path)["outcomes.holdout"] == "on" == _states(tmp_path)["outcomes.swaps"]


def test_receipts_go_only_when_asked(tmp_path) -> None:
    switch.set_measurement(tmp_path, False, channel="cli", include_receipts=True)
    assert _states(tmp_path)["outcomes.receipts"] == "off"
    switch.set_measurement(tmp_path, True, channel="cli", include_receipts=True)
    assert _states(tmp_path)["outcomes.receipts"] == "on"


def test_an_unrelated_flag_and_the_kill_switch_survive_a_change(tmp_path) -> None:
    from superlocalmemory.core.feature_flags import write_flag

    write_flag(tmp_path, "truth.slots", FlagState.ON)
    set_kill_switch(tmp_path, True)
    switch.set_measurement(tmp_path, False, channel="cli")
    data = json.loads((tmp_path / "features.json").read_text())
    assert data["kill_switch"] is True and data["features"]["truth.slots"] == "on"


def test_the_kill_switch_still_turns_everything_off_and_status_says_so(tmp_path) -> None:
    set_kill_switch(tmp_path, True)
    out = switch.status(tmp_path)
    assert out["kill_switch"] is True and out["measuring"] is False
    assert {f["state"] for f in out["flags"]} == {"off"}


def test_status_names_every_flag_in_plain_words(tmp_path) -> None:
    out = switch.status(tmp_path)
    ids = [f["id"] for f in out["flags"]]
    assert ids == ["outcomes.receipts", "outcomes.holdout", "outcomes.swaps",
                   "outcomes.usage", "delivery.need_gate"]
    for f in out["flags"]:
        assert f["label"] and f["what"] and f["state"] in {"on", "off"}
    assert out["notice_version"] == disclosure.NOTICE_VERSION and out["pending"] is True
    assert "off unless" not in {f["id"]: f["what"] for f in out["flags"]}["outcomes.usage"].lower()


def test_an_environment_override_shows_as_the_effective_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLM_FEATURE_OUTCOMES_HOLDOUT", "off")
    out = switch.status(tmp_path)
    assert {f["id"]: f["state"] for f in out["flags"]}["outcomes.holdout"] == "off"
    assert out["measuring"] is True  # swaps still run: either intervention on counts


def test_each_change_is_recorded_with_time_channel_and_what_changed(tmp_path) -> None:
    switch.set_measurement(tmp_path, False, channel="dashboard")
    switch.set_measurement(tmp_path, False, channel="dashboard")  # nothing to change
    changes = json.loads((tmp_path / "disclosures.json").read_text())["changes"]
    assert len(changes) == 1
    assert changes[0]["action"] == "off" and changes[0]["channel"] == "dashboard"
    assert changes[0]["at"] and set(changes[0]["flags"]) == {*INTERVENTIONS, "outcomes.usage"}


def test_a_failed_record_never_blocks_turning_it_off(tmp_path, monkeypatch) -> None:
    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(disclosure, "record_change", broken)
    switch.set_measurement(tmp_path, False, channel="cli")
    assert _states(tmp_path)["outcomes.holdout"] == "off"


def test_an_unknown_channel_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        switch.set_measurement(tmp_path, False, channel="smoke-signal")
    assert _states(tmp_path)["outcomes.holdout"] == "on"


def test_the_switch_really_stops_the_draws(tmp_path) -> None:
    from superlocalmemory.outcomes import swaps

    assert any(swaps.decide(tmp_path, f"d{i}", 10) is not None for i in range(400))
    switch.set_measurement(tmp_path, False, channel="cli")
    assert all(swaps.decide(tmp_path, f"d{i}", 10) is None for i in range(400))
