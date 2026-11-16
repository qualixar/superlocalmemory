# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""R-41 / D-27: the scheduler builds a shadow ladder without a hosted tier.

The ladder itself is tested in ``test_judge_ladder.py``; this pins the wiring,
so a later edit to the maintenance cycle cannot bring hosted calls back into
shadow while the consent to send is on file.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from superlocalmemory.core import consent_store
from superlocalmemory.core.feature_flags import FLAGS, FlagState, env_name, write_flag
from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler


@pytest.fixture
def built(tmp_path, monkeypatch):
    from superlocalmemory.infra import data_root
    from superlocalmemory.truth import dispute_pass, slot_pass

    for fid in FLAGS:
        monkeypatch.delenv(env_name(fid), raising=False)
    monkeypatch.setattr(data_root, "canonical_data_root", lambda *a, **k: tmp_path)
    monkeypatch.setattr(data_root, "state_path", lambda *a, **k: tmp_path / "state")

    class NoSlotPass:
        def __init__(self, *a, **k):
            pass

        def run(self):
            return None

    ladders = []

    class CapturingDisputePass:
        def __init__(self, db, flags, ladder, *, hash_key):
            ladders.append(ladder)

        def run(self):
            return None

    monkeypatch.setattr(slot_pass, "SlotPass", NoSlotPass)
    monkeypatch.setattr(dispute_pass, "DisputePass", CapturingDisputePass)
    consent_store.record(tmp_path, "judge_ladder_hosted", True, via="test")
    write_flag(tmp_path, "truth.slots", FlagState.SHADOW)

    config = MagicMock()
    config.forgetting.scheduler_interval_minutes = 360
    live = {"judge": object()}
    scheduler = MaintenanceScheduler(MagicMock(), config, judge_supplier=lambda: live["judge"])
    scheduler._running = True

    def run(ladder_state: FlagState, judge=None):
        if judge is not None:
            live["judge"] = judge
        write_flag(tmp_path, "judge.ladder", ladder_state)
        scheduler._slot_pass()
        return ladders[-1]

    return run


def test_a_shadow_ladder_has_no_hosted_tier_even_with_consent(built) -> None:
    assert built(FlagState.SHADOW).tier_names == ("rules", "laya")


def test_an_acting_ladder_has_the_hosted_tier(built) -> None:
    ladder = built(FlagState.ON)
    assert ladder.tier_names == ("rules", "laya", "jev")
    jev = ladder._tiers[-1]
    assert jev._consent_ref().startswith("judge_ladder_hosted@")  # the record, for I-15


def test_a_consent_for_another_provider_is_not_used(built, tmp_path) -> None:
    from types import SimpleNamespace

    consent_store.change(tmp_path, "judge_ladder_hosted", True, via="test", actor="t",
                         provider="typesafe", audit_path=tmp_path / "audit.db")
    jev = built(FlagState.ON, judge=SimpleNamespace(_provider="openrouter"))._tiers[-1]
    assert jev._consent_ref() is None  # the answer check moved: nothing is sent
    jev = built(FlagState.ON, judge=SimpleNamespace(_provider="typesafe"))._tiers[-1]
    assert jev._consent_ref().startswith("judge_ladder_hosted@")
