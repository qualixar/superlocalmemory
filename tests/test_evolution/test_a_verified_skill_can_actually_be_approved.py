# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A mutation that passes verification used to have nowhere to go.

Automatic approval is off by default and stays off — this rewrites the
instructions an AI follows, and doing that without a person agreeing is not a
default worth shipping. But there was no way for a person to agree either, so
every verified improvement stopped in a quarantine directory and stayed there.

These cover the path that was missing, and the refusals that keep it safe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.evolution.skill_activator import (
    SkillActivator,
)


@pytest.fixture
def roots(tmp_path):
    live = tmp_path / "live"
    backup = tmp_path / "backup"
    quarantine = tmp_path / "quarantine"
    for d in (live, backup, quarantine):
        d.mkdir()
    return live, backup, quarantine


@pytest.fixture
def activator(roots):
    live, backup, quarantine = roots
    return SkillActivator(
        live_root=live, backup_root=backup, quarantine_root=quarantine,
    )


def _stage(quarantine: Path, dir_name: str, body: str) -> Path:
    d = quarantine / dir_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body, encoding="utf-8")
    return p


def _live(live: Path, skill: str, body: str) -> Path:
    d = live / skill
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_approving_replaces_the_live_instructions(activator, roots):
    live, _backup, quarantine = roots
    _live(live, "brainstorming", "old instructions\n")
    _stage(quarantine, "brainstorming-vabc12", "improved instructions\n")

    result = activator.activate(
        "brainstorming", "brainstorming-vabc12", actor_id="dashboard",
    )

    assert (live / "brainstorming" / "SKILL.md").read_text(encoding="utf-8") == "improved instructions\n"
    assert result["content_hash"]
    assert result["actor_id"] == "dashboard"


def test_the_previous_version_is_kept_so_approval_is_reversible(activator, roots):
    live, _backup, quarantine = roots
    _live(live, "brainstorming", "old instructions\n")
    _stage(quarantine, "brainstorming-vabc12", "improved instructions\n")

    activator.activate("brainstorming", "brainstorming-vabc12")
    rolled = activator.rollback("brainstorming")

    assert rolled["rolled_back"] is True
    assert (live / "brainstorming" / "SKILL.md").read_text(encoding="utf-8") == "old instructions\n"


def test_rolling_back_a_brand_new_skill_removes_it(activator, roots):
    """There is no previous version to restore, so the correct undo is removal."""
    live, _backup, quarantine = roots
    _stage(quarantine, "newskill-v1", "brand new\n")

    activator.activate("newskill", "newskill-v1")
    assert (live / "newskill" / "SKILL.md").exists()

    activator.rollback("newskill")
    assert not (live / "newskill" / "SKILL.md").exists()


def test_approving_something_that_is_not_there_fails_loudly(activator):
    with pytest.raises(FileNotFoundError):
        activator.activate("brainstorming", "does-not-exist")


def test_a_skill_name_cannot_escape_its_directory(activator, roots):
    """The name arrives over HTTP, so it is not trusted as a path."""
    _live_root, _backup, quarantine = roots
    _stage(quarantine, "evil", "payload\n")
    with pytest.raises(ValueError):
        activator.activate("../../../../etc/malicious", "evil")


# ---------------------------------------------------------------------------
# The route's own guard
# ---------------------------------------------------------------------------

def test_only_a_verified_candidate_may_be_approved():
    """Approving twice would destroy the thing rollback restores.

    The second activation writes the mutation over the backup taken by the
    first, so the rollback target becomes the mutation itself and the previous
    instructions are gone. The status guard is what prevents that, so the set of
    approvable statuses is asserted directly.
    """
    from superlocalmemory.evolution.types import EvolutionStatus
    from superlocalmemory.server.routes.evolution import _APPROVABLE

    assert EvolutionStatus.VERIFIED_QUARANTINED.value in _APPROVABLE
    # The legacy alias for the same state, still present on older rows.
    assert EvolutionStatus.PROMOTED.value in _APPROVABLE

    for blocked in (
        EvolutionStatus.ACTIVE,
        EvolutionStatus.APPROVED,
        EvolutionStatus.REJECTED,
        EvolutionStatus.FAILED,
        EvolutionStatus.CANDIDATE,
        EvolutionStatus.ROLLED_BACK,
    ):
        assert blocked.value not in _APPROVABLE, (
            f"{blocked.value} is approvable; activating from it would overwrite "
            "the backup that rollback depends on"
        )


def test_automatic_approval_is_still_off_by_default():
    """This task added a path for a person to approve, not an automatic one."""
    from superlocalmemory.core.config import EvolutionConfig

    assert getattr(EvolutionConfig(), "auto_approve", False) is False
