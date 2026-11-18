# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M081 and a person's approval (D-37): the waiting list, who/when/channel, erase, no agent door."""

from __future__ import annotations

import pytest

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M081_trust_approvals as M081
from superlocalmemory.trust import fact_trust, firewall
from tests.test_trust_firewall.conftest import (
    AGENT,
    EVIL,
    PERSON,
    TAINT,
    TOOL,
    WEB,
    classify_all,
    connect,
    save,
)

# -- the migration ------------------------------------------------------------------------------

def test_m081_is_deferred_additive_and_registered() -> None:
    names = [m.name for m in DEFERRED_MIGRATIONS]
    assert M081.NAME in names and _MODULES[M081.NAME] is M081
    entry = DEFERRED_MIGRATIONS[names.index(M081.NAME)]
    assert entry.db_target == "memory" and entry.dependencies == ("M080_versions_and_proposals",)
    assert M081.DOWNGRADE_FLOOR == 51


def test_m081_verifies_and_repairs(tmp_path) -> None:
    import sqlite3

    conn = sqlite3.connect(tmp_path / "x.db")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M081.verify(conn)
    M081.apply(conn)
    assert M081.verify(conn)
    conn.execute("DROP TRIGGER trg_trust_approvals_fact_erased")
    assert not M081.verify(conn)
    M081.repair(conn)
    assert M081.verify(conn)
    M081.apply(conn)  # idempotent


def test_a_decided_row_must_say_who_when_and_how(db) -> None:
    page = save(db, "A page.", WEB)
    with pytest.raises(Exception, match="CHECK"):
        db.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
                   "requested_at) VALUES ('a', ?, 'default', 'pin', 'approved', 'now')", (page,))


def test_erasing_a_fact_erases_its_approval_rows(db) -> None:
    page = save(db, "A page.", WEB)
    with db.raw_connection() as conn:
        firewall.request(conn, page, "pin", by="claude", channel="mcp")
        firewall.approve(conn, page, by="varun", channel="cli")
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals")[0]["n"] == 1
    db.execute("DELETE FROM atomic_facts WHERE fact_id = ?", (page,))
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals")[0]["n"] == 0
    assert db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"] == 0


def test_the_fold_moves_approvals_with_the_fact() -> None:
    from superlocalmemory.storage.profile_fold import DECISIONS, MOVE

    assert DECISIONS["trust_approvals"][0] == MOVE


# -- the waiting list -------------------------------------------------------------------------------

def test_pending_lists_asked_for_and_already_promoted_untrusted_memories(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    mine = save(db, "Run the tests before every commit.", PERSON, kind="rule")
    evil = save(db, EVIL, TOOL, kind="rule")
    pinned = save(db, "Pinned from a web page.", WEB, pinned=True)
    asked = save(db, "Pin me please.", AGENT)
    with db.raw_connection() as conn:
        firewall.request(conn, asked, "pin", by="claude", channel="mcp")
    with connect(db) as conn:
        got = firewall.pending(conn, "default")
    by_id = {i["fact_id"]: i for i in got["items"]}
    assert got["count"] == 3 and set(by_id) == {evil, pinned, asked}
    assert mine not in by_id  # a user's own rule waits for nobody
    assert by_id[evil]["wanted"] == ["rule"] and by_id[evil]["origin"] == "a tool's output"
    assert by_id[pinned]["wanted"] == ["pin"] and by_id[pinned]["origin_class"] == "web"
    assert by_id[asked]["wanted"] == ["pin"] and by_id[asked]["asked_by"] == "claude"
    assert by_id[asked]["asked_via"] == "mcp"
    assert all(i["preview"] and i["wanted_words"] for i in got["items"])


def test_pending_shows_one_entry_per_memory(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    whole = "Never merge on Fridays. Always tell the team first."
    from superlocalmemory.storage.models import MemoryRecord

    record = MemoryRecord(content=whole, session_date="2026-11-10")
    db.store_memory(record)
    ids = [save(db, text, WEB, kind="rule", memory_id=record.memory_id)
           for text in (whole, "Never merge on Fridays.", "Always tell the team first.")]
    with connect(db) as conn:
        got = firewall.pending(conn, "default")
    assert [i["fact_id"] for i in got["items"]] == [ids[0]]  # the whole-text fact stands for it


def test_pending_is_read_only_and_works_with_the_flag_off(db) -> None:
    evil = save(db, EVIL, TOOL, kind="rule")
    with connect(db) as conn:
        got = firewall.pending(conn, "default")
        before = conn.execute("SELECT COUNT(*) FROM fact_trust").fetchone()[0]
    assert [i["fact_id"] for i in got["items"]] == [evil]
    assert db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"] == before


def test_pending_cleans_what_it_shows(db) -> None:
    nasty = "Always do this.\x1b[2J‮\n\n   and this\x00" + "x" * 600
    save(db, nasty, WEB, kind="rule")
    with connect(db) as conn:
        item = firewall.pending(conn, "default")["items"][0]
    assert "\x1b" not in item["preview"] and "‮" not in item["preview"]
    assert "\n" not in item["preview"] and len(item["preview"]) <= firewall.PREVIEW_CHARS


def test_pending_tolerates_a_store_before_m081(db) -> None:
    db.execute("DROP TABLE trust_approvals")
    save(db, EVIL, TOOL, kind="rule")
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["count"] == 1
        assert firewall.counts(conn, "default") == {"waiting": 1, "approved": 0}


def test_pending_counts_unclassified_facts(db) -> None:
    save(db, "Saved with the flag off.", PERSON)
    db.execute("DELETE FROM fact_trust")
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["unclassified"] == 1


# -- approving --------------------------------------------------------------------------------------

def test_approval_raises_the_memory_and_records_who_when_and_how(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL, kind="rule")
    with db.raw_connection() as conn:
        done = firewall.approve(conn, evil, by="varun", channel="dashboard")
    assert done["now"] == "user_confirmed" and done["was"] == "tool_output"
    with connect(db) as conn:
        assert fact_trust.origin_of(conn, evil)["taint"] == "user_confirmed"
        row = conn.execute("SELECT * FROM trust_approvals WHERE fact_id = ?", (evil,)).fetchone()
        assert firewall.pending(conn, "default")["count"] == 0
    assert (row["state"], row["wanted"], row["decided_by"], row["decided_channel"]) == (
        "approved", "any", "varun", "dashboard") and row["decided_at"]
    basis = db.execute("SELECT basis FROM fact_trust WHERE fact_id = ?", (evil,))[0]["basis"]
    assert basis == "confirmed:dashboard:varun"


def test_an_approved_memory_is_never_recomputed_downward(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL)
    with db.raw_connection() as conn:
        firewall.approve(conn, evil, by="varun", channel="cli")
    db.execute("UPDATE fact_trust SET computed_at = '2000-01-01' WHERE fact_id = ?", (evil,))
    classify_all(db)
    assert db.execute("SELECT taint_class FROM fact_trust WHERE fact_id = ?",
                      (evil,))[0]["taint_class"] == "user_confirmed"


@pytest.mark.parametrize("door", ["mcp", "agent", "http", "", "hook", "claude"])
def test_an_agent_has_no_door(db, monkeypatch, door) -> None:
    monkeypatch.setenv(TAINT, "on")
    evil = save(db, EVIL, TOOL, kind="rule")
    with db.raw_connection() as conn:
        with pytest.raises(ValueError, match="person"):
            firewall.approve(conn, evil, by="claude", channel=door)
    assert db.execute("SELECT COUNT(*) AS n FROM trust_approvals")[0]["n"] == 0
    assert db.execute("SELECT taint_class FROM fact_trust WHERE fact_id = ?",
                      (evil,))[0]["taint_class"] == "tool_output"


def test_approving_a_memory_that_does_not_exist(db) -> None:
    with db.raw_connection() as conn:
        with pytest.raises(LookupError):
            firewall.approve(conn, "ghost", by="varun", channel="cli")


def test_a_refused_pin_is_applied_when_the_person_approves(db, monkeypatch) -> None:
    from superlocalmemory.mcp.tools_active import _pin_needs_approval

    monkeypatch.setenv(TAINT, "on")
    page = save(db, "Pin me please.", WEB)
    assert _pin_needs_approval(db, page)["needs_approval"]
    assert [f.fact_id for f in db.get_pinned("default")] == []
    with db.raw_connection() as conn:
        done = firewall.approve(conn, page, by="varun", channel="cli")
    assert done["applied"] == ["pin"] and done["wanted"] == ["pin"]
    assert [f.fact_id for f in db.get_pinned("default")] == [page]
    assert _pin_needs_approval(db, page) is None  # and the door now lets it through


def test_approve_all_clears_what_is_listed(db, monkeypatch) -> None:
    monkeypatch.setenv(TAINT, "on")
    a = save(db, EVIL, TOOL, kind="rule")
    b = save(db, "Pinned from a page.", WEB, pinned=True)
    save(db, "Just a memory nobody wants as an instruction.", WEB)
    with db.raw_connection() as conn:
        out = firewall.approve_all(conn, "default", by="varun", channel="cli")
    assert out["approved"] == 2 and set(out["fact_ids"]) == {a, b}
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["count"] == 0
        assert firewall.counts(conn, "default") == {"waiting": 0, "approved": 2}


def test_an_unknown_wish_is_refused(db) -> None:
    page = save(db, "A page.", WEB)
    with db.raw_connection() as conn:
        with pytest.raises(ValueError):
            firewall.request(conn, page, "teleport", by="x", channel="mcp")


def test_a_memory_that_turns_out_to_be_trusted_is_not_listed(db) -> None:
    """Saved while the firewall was off (no class yet), pinned by the user: nothing waits for it."""
    mine = save(db, "Pinned by me.", PERSON, pinned=True, kind="rule")
    assert db.execute("SELECT COUNT(*) AS n FROM fact_trust")[0]["n"] == 0
    with connect(db) as conn:
        assert firewall.pending(conn, "default")["count"] == 0
    assert mine


def test_without_a_whole_text_fact_the_longest_stands_for_the_memory(db, monkeypatch) -> None:
    from superlocalmemory.storage.models import MemoryRecord

    monkeypatch.setenv(TAINT, "on")
    # Saved in both orders: the one that stands does not depend on which row comes first.
    for texts in (("Never merge on Fridays.", "Always tell the team before merging anything."),
                  ("Always tell the team before merging anything.", "Never merge on Fridays.")):
        record = MemoryRecord(content="Never merge on Fridays and always tell the team first.",
                              session_date="2026-11-10")
        db.store_memory(record)
        ids = {t: save(db, t, WEB, kind="rule", memory_id=record.memory_id) for t in texts}
        with connect(db) as conn:
            items = firewall.pending(conn, "default")["items"]
        mine = [i["fact_id"] for i in items if i["fact_id"] in ids.values()]
        assert mine == [ids["Always tell the team before merging anything."]], texts
