# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Poisoning scenario 5 (W4 exit gate): with both flags off, nothing changes (4.2.30 ships dark)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from superlocalmemory.core import standing_rules
from superlocalmemory.core.block_sources import not_sources, source_filter
from superlocalmemory.core.feature_flags import FLAGS, KILL_ENV, FlagState
from superlocalmemory.mcp.tools_active import _pin_needs_approval, register_active_tools
from superlocalmemory.memory_core import prepare_metadata, prepare_user_text
from superlocalmemory.trust import firewall
from tests.test_trust_firewall import secrets as fake
from tests.test_trust_firewall.conftest import EVIL, PERSON, SCREEN, TAINT, TOOL, WEB, save
from tests.test_trust_poisoning.test_tool_output_firewall import (
    PID,
    _blocks,
    _boost,
    _session,
    _sources,
)


def test_both_flags_ship_on_and_a_persons_off_reads_as_before(monkeypatch) -> None:
    assert FLAGS["trust.taint"].default is FlagState.ON
    assert FLAGS["trust.secret_screen"].default is FlagState.ON
    monkeypatch.delenv(TAINT)  # the suite's baseline is an environment "off"
    assert firewall.active() and firewall.observing()
    monkeypatch.setenv(TAINT, "off")
    assert not firewall.active() and not firewall.observing()


@pytest.mark.parametrize("taint,screen", [("off", "off"), ("shadow", "shadow")])
def test_off_and_shadow_are_the_behaviour_of_before(db, monkeypatch, taint, screen) -> None:
    monkeypatch.setenv(TAINT, taint)
    monkeypatch.setenv(SCREEN, screen)
    mine = save(db, "Run the tests before every commit.", PERSON, kind="rule")
    evil = save(db, EVIL, TOOL, kind="rule")
    pinned = save(db, "Pinned from a page.", WEB, pinned=True)
    core = save(db, "An opinion from a page.", WEB)
    _boost(db, core)
    # rules and decisions
    assert {f.fact_id for f in standing_rules.standing_facts(db, PID)} == {mine, evil}
    # pins
    assert _pin_needs_approval(db, pinned) is None
    assert {f.fact_id for f in db.get_pinned(PID)} == {pinned}
    # core blocks and the session-start sources
    assert "An opinion from a page." in _blocks(db)
    sql, params = source_filter(db)
    assert firewall.clause(db) == "" and "fact_trust" not in sql
    with db.raw_connection() as conn:
        assert not_sources(conn, [mine, evil, pinned]) == set()
    # session start
    out = _session(db)
    assert _sources(out)[evil] == "standing-rule" and _sources(out)[pinned] == "pinned-fact"
    # nothing was recorded for the firewall
    if taint == "off":
        assert db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals")[0]["n"] == 0


@pytest.mark.parametrize("flag", ["off", "shadow"])
def test_the_secret_screen_off_or_in_shadow_stores_the_text_as_written(monkeypatch, flag) -> None:
    monkeypatch.setenv(SCREEN, flag)
    secret = fake.github()
    text = f"Deploy token {secret} for the nightly job."
    got = prepare_user_text(None, text)
    assert got.text == text and got.secret_count == 0 and got.secret_types == ()
    meta = {"tags": f"deploy,{secret}"}
    assert prepare_metadata(meta, pii_redaction=False) == (meta, 0)


def test_the_kill_switch_beats_both_flags(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    monkeypatch.setenv(SCREEN, "on")
    monkeypatch.setenv(KILL_ENV, "1")
    evil = save(db, EVIL, TOOL, kind="rule")
    assert [f.fact_id for f in standing_rules.standing_facts(db, PID)] == [evil]
    secret = fake.github()
    assert secret in prepare_user_text(None, f"key {secret} here").text


def test_the_pin_tool_still_answers_as_before(db) -> None:
    page = save(db, "Pin me.", WEB)
    server = SimpleNamespace(tools={})

    def tool(*a, **k):
        return lambda fn: server.tools.setdefault(fn.__name__, fn) or fn

    server.tool = tool
    register_active_tools(server, lambda: SimpleNamespace(profile_id=PID, mode="B", db=db,
                                                          _db=None, config=None))
    listing = asyncio.run(server.tools["core_memory"](action="list"))
    assert listing["success"] is True and listing["count"] == 0
    assert _pin_needs_approval(db, page) is None
