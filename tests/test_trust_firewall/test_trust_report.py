# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Trust section of Memory Health: counts per type, never a value, fail-open."""

from __future__ import annotations

import json

from superlocalmemory.outcomes import memory_report
from superlocalmemory.trust import report
from tests.test_trust_firewall import secrets as fake
from tests.test_trust_firewall.conftest import (
    EVIL,
    PERSON,
    SCREEN,
    TAINT,
    TOOL,
    WEB,
    connect,
    save,
)


def _store(db, monkeypatch):
    monkeypatch.setenv(TAINT, "on")
    monkeypatch.setenv(SCREEN, "on")
    from superlocalmemory.memory_core import prepare_user_text

    for secret in (fake.github(), fake.github()[:-1] + "Z", fake.aws()):
        text = prepare_user_text(None, f"Deploy note with the credential {secret} inside.").text
        save(db, text, PERSON)
    save(db, EVIL, TOOL, kind="rule")
    save(db, "Pinned from a page.", WEB, pinned=True)
    return fake.github(), fake.aws()


def test_the_section_counts_credentials_by_type_and_never_shows_one(db, monkeypatch) -> None:
    secrets = _store(db, monkeypatch)
    with connect(db) as conn:
        section = report.build(conn, "default")
    assert section["secrets_held_back"]["by_type"] == {"AWS": 1, "GITHUB": 2}
    assert section["secrets_held_back"]["memories"] == 3
    assert section["taint"] == "on" and section["secret_screen"] == "on"
    assert section["waiting"] == 2 and section["unclassified"] == 0
    assert {i["origin"] for i in section["waiting_items"]} == {"tool_output", "web"}
    assert all(set(i) == {"fact_id", "origin", "wanted", "preview"} for i in section["waiting_items"])
    assert section["origins"]["user_direct"] == 3 and section["origins"]["tool_output"] == 1
    blob = json.dumps(section)
    assert all(s not in blob for s in secrets)


def test_the_words_for_a_terminal(db, monkeypatch) -> None:
    _store(db, monkeypatch)
    with connect(db) as conn:
        lines = report.describe(report.build(conn, "default"))
    text = "\n".join(lines)
    assert "instruction firewall on" in text and "secret screen on" in text
    assert "Held back until promoted: 2" in text
    assert "slm trust promote" in text and "recorded" in text
    assert "your OK" not in text and "at a terminal" not in text
    assert "Credentials stored as placeholders: AWS 1, GITHUB 2" in text
    assert report.describe({}) == []


def test_the_report_carries_the_section_and_prints_it(db, monkeypatch) -> None:
    _store(db, monkeypatch)
    with connect(db) as conn:
        full = memory_report.build(conn, "default")
    assert full["trust"]["secrets_held_back"]["by_type"]["GITHUB"] == 2
    assert any("Credentials stored as placeholders" in line for line in memory_report.describe(full))


def test_a_store_before_the_tables_reports_empty_not_an_error(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    db.execute("DROP TABLE fact_trust")
    db.execute("DROP TABLE trust_approvals")
    with connect(db) as conn:
        section = report.build(conn, "default")
    assert section["origins"] == {} and section["waiting"] == 0
    assert section["secrets_held_back"] == {"by_type": {}, "memories": 0, "capped": False}


def test_an_unreadable_section_never_breaks_the_report(monkeypatch, db) -> None:
    monkeypatch.setattr(report.firewall, "counts", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    with connect(db) as conn:
        assert "trust" in memory_report.build(conn, "default")


def test_the_scan_is_bounded(db, monkeypatch) -> None:
    monkeypatch.setattr(report, "SCAN_ROWS", 1)
    secrets = _store(db, monkeypatch)
    with connect(db) as conn:
        held = report.placeholders(conn, "default")
    assert held["capped"] is True and sum(held["by_type"].values()) == 1
    assert all(s not in json.dumps(held) for s in secrets)


def test_a_memory_with_several_facts_is_counted_once(db, monkeypatch) -> None:
    from superlocalmemory.storage.models import MemoryRecord

    monkeypatch.setenv(SCREEN, "on")
    from superlocalmemory.memory_core import prepare_user_text

    text = prepare_user_text(None, f"Deploy note with the credential {fake.github()} inside.").text
    record = MemoryRecord(content=text, session_date="2026-11-10")
    db.store_memory(record)
    for i in range(3):  # the extractor writes a fact per sentence; each repeats the placeholder
        save(db, f"{text} Part {i}.", PERSON, memory_id=record.memory_id)
    assert db.execute("SELECT COUNT(*) AS n FROM atomic_facts")[0]["n"] == 3
    with connect(db) as conn:
        held = report.placeholders(conn, "default")
    assert held["by_type"] == {"GITHUB": 1} and held["memories"] == 1


def test_quoting_the_masking_rule_is_not_a_credential() -> None:
    from superlocalmemory.trust import secret_screen

    text = "Mask secrets as [REDACTED:TYPE:LAST4]; this one is [REDACTED:GITHUB_PAT:ab12]."
    assert dict(secret_screen.types_in(text)) == {"GITHUB_PAT": 1}


def test_with_the_firewall_off_nothing_is_said_to_wait() -> None:
    from superlocalmemory.trust import report

    off = "\n".join(report.describe({"taint": "off", "secret_screen": "off", "waiting": 31,
                                     "approved": 0}))
    assert "Would be held back if the firewall were on: 31" in off
    assert "Held back until promoted: 31" not in off
    on = "\n".join(report.describe({"taint": "on", "secret_screen": "off", "waiting": 3,
                                    "approved": 1}))
    assert "Held back until promoted: 3; promoted: 1" in on


def test_memory_health_counts_the_decisions_agents_made(db, monkeypatch) -> None:
    from superlocalmemory.trust import firewall

    monkeypatch.setenv(TAINT, "on")
    fact = save(db, "Pinned from a page.", WEB, pinned=True)
    with connect(db) as conn:
        assert report.build(conn, "default")["decisions"] == {"total": 0, "undone": 0, "open": 0}
        assert not any("Decisions" in x for x in report.describe(report.build(conn, "default")))
        firewall.approve(conn, fact, by="agent:claude-code", channel="mcp")
        section = report.build(conn, "default")
        conn.commit()
    assert section["decisions"] == {"total": 1, "undone": 0, "open": 1}
    line = [x for x in report.describe(section) if "Decisions" in x]
    assert line == ["  Decisions by your agents: 1 recorded, 0 undone (the dashboard lists them with an Undo)"]


def test_the_decisions_line_counts_what_was_undone(db, monkeypatch) -> None:
    from superlocalmemory.trust import firewall

    monkeypatch.setenv(TAINT, "on")
    fact = save(db, "Pinned from a page.", WEB, pinned=True)
    with connect(db) as conn:
        firewall.approve(conn, fact, by="agent:claude-code", channel="mcp")
        firewall.withdraw(conn, fact, by="dashboard", channel="dashboard")
        section = report.build(conn, "default")
        conn.commit()
    assert section["decisions"]["undone"] == 1
    assert any("1 undone" in x for x in report.describe(section))
