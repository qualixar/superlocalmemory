# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Shipping a dependency is not the same as using it.

Cozo and LanceDB have been required dependencies since 3.7 and the projections
that make them serve were built only by three manual commands. Almost nobody
ran them, so almost every store kept answering graph and similarity questions
out of SQLite while carrying two unused libraries.

What matters in the automatic version is what it refuses to do: it must never
leave a store worse than it found it, and it must never stop one from working.
A user whose native extension does not match their interpreter would otherwise
have no product at all — and that was the state of the machine this was written
on.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import scale_autopromote as auto


class _Config:
    def __init__(self, state="local_core"):
        self.scale_engine_state = state


class _Manager:
    """A scale engine that records what it was asked to do."""

    def __init__(self, *, verified=True, repair=False, raise_on=""):
        self.calls: list[str] = []
        self._verified = verified
        self._repair = repair
        self._raise_on = raise_on

    def status(self):
        self.calls.append("status")
        return {"migration_repair_required": self._repair}

    def prepare(self):
        self.calls.append("prepare")
        if self._raise_on == "prepare":
            raise RuntimeError("disk full")
        return {"stage_id": "stage-1"}

    def verify(self, stage_id):
        self.calls.append(f"verify:{stage_id}")
        return {"state": "verified" if self._verified else "prepared"}

    def promote(self, stage_id):
        self.calls.append(f"promote:{stage_id}")
        return {"stage_id": stage_id, "restart_required": True}


@pytest.fixture()
def manager(monkeypatch):
    made: list[_Manager] = []

    def install(m: _Manager):
        monkeypatch.setattr(auto, "_missing_libraries", lambda: [])
        import superlocalmemory.core.scale_engine as engine
        monkeypatch.setattr(engine, "ScaleEngineManager", lambda config: m)
        made.append(m)
        return m

    return install


def test_a_store_that_has_not_got_them_gets_them(manager) -> None:
    m = manager(_Manager())
    result = auto.auto_promote_scale_backends(_Config())

    assert result.promoted is True
    assert result.restart_required is True
    assert m.calls == ["status", "prepare", "verify:stage-1", "promote:stage-1"], (
        "the sequence must be prepare, then verify, then promote"
    )


def test_a_store_that_already_has_them_is_left_alone(manager) -> None:
    m = manager(_Manager())
    result = auto.auto_promote_scale_backends(_Config("promoted"))

    assert result.attempted is False
    assert result.promoted is True
    assert m.calls == [], "it re-ran a promotion that had already happened"


def test_a_projection_that_does_not_match_is_not_promoted(manager) -> None:
    """This is the whole point of verifying before promoting."""
    m = manager(_Manager(verified=False))
    result = auto.auto_promote_scale_backends(_Config())

    assert result.promoted is False
    assert "did not match" in result.reason
    assert not any(c.startswith("promote") for c in m.calls), (
        "a projection that failed verification was promoted anyway"
    )


def test_an_interrupted_promotion_is_left_for_repair(manager) -> None:
    m = manager(_Manager(repair=True))
    result = auto.auto_promote_scale_backends(_Config())

    assert result.attempted is False
    assert "repair" in result.reason
    assert m.calls == ["status"]


def test_a_failure_never_stops_the_store_working(manager) -> None:
    manager(_Manager(raise_on="prepare"))
    result = auto.auto_promote_scale_backends(_Config())   # must not raise

    assert result.promoted is False
    assert "disk full" in result.reason


def test_missing_libraries_are_reported_not_raised(monkeypatch) -> None:
    """The state of the machine this was written on."""
    monkeypatch.setattr(auto, "_missing_libraries", lambda: ["pycozo", "lancedb"])
    result = auto.auto_promote_scale_backends(_Config())

    assert result.attempted is False
    assert result.promoted is False
    assert "pycozo" in result.reason and "lancedb" in result.reason


def test_the_library_check_is_a_real_import(monkeypatch) -> None:
    """find_spec would call a broken native extension usable.

    No real native library is imported here: ``import lancedb`` starts a
    background event loop that lives for the rest of the process, and in the
    normal lane that loop outlived this test and was still running when an
    unrelated numpy test later crashed. Each library is answered by a stub or
    a failure; what is under test is that the check really imports.
    """
    import importlib
    import types

    asked: list[str] = []

    def broken(name, *args, **kwargs):
        asked.append(name)
        if name == "pycozo":
            raise ImportError("dlopen failed: incompatible architecture")
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", broken)
    missing = auto._missing_libraries()
    assert missing == ["pycozo"]
    assert sorted(asked) == sorted(auto.REQUIRED_LIBRARIES)


def test_a_switch_that_says_no_is_obeyed(manager) -> None:
    """It exists, it is honoured elsewhere, and this ignored it.

    Moving somebody's store on a setting that says not to would be worse than
    never having automated it at all.
    """
    m = manager(_Manager())
    config = _Config()
    config.scale_auto_promote_enabled = False

    result = auto.auto_promote_scale_backends(config)

    assert result.attempted is False
    assert "switched off" in result.reason
    assert m.calls == [], "it projected a store whose configuration said not to"


def test_a_stage_already_built_is_resumed_not_rebuilt(manager, monkeypatch) -> None:
    """Only "promoted" used to stop it, so every start built another stage.

    A start that prepares and then fails to verify leaves the state at
    "prepared". The next start built a second stage, the one after a third, and
    the staging directory grew with each.
    """
    class _WithStage(_Manager):
        def status(self):
            self.calls.append("status")
            return {
                "migration_repair_required": False,
                "stages": [
                    {"stage_id": "earlier", "state": "prepared", "created_at": "1"},
                    {"stage_id": "latest", "state": "verified", "created_at": "2"},
                ],
            }

    m = manager(_WithStage())
    result = auto.auto_promote_scale_backends(_Config("prepared"))

    assert "prepare" not in m.calls, "it built another stage over one already built"
    assert m.calls == ["status", "verify:latest", "promote:latest"]
    assert result.promoted is True


def test_the_newest_usable_stage_wins() -> None:
    picked = auto._resumable_stage({
        "stages": [
            {"stage_id": "a", "state": "prepared", "created_at": "1"},
            {"stage_id": "b", "state": "verified", "created_at": "2"},
            {"stage_id": "gone", "state": "promoted", "created_at": "3"},
            {"stage_id": "broken", "state": "corrupt", "created_at": "4"},
        ],
    })
    assert picked == "b"
    assert auto._resumable_stage({"stages": []}) == ""
