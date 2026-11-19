# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""'Shown to your agents': deliveries grouped by conversation, each memory with its reason, read-only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.outcomes import delivered
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.truth.invariants import open_read_only
from tests.test_core.test_coverage_backfill import _db

NOW = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)  # after the facts' real save time, on any day


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    for name in ("SLM_4230_OFF", "SLM_FEATURE_TRUST_TAINT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def db(tmp_path):
    store = _db(tmp_path / "memory.db")
    store.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')")
    return store


def _fact(db, text, *, profile="default", saved="2026-11-15"):
    record = MemoryRecord(content=text, session_date=saved, profile_id=profile)
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=text, profile_id=profile)
    db.store_fact(fact)
    return fact


def _ago(minutes):
    return (NOW - timedelta(minutes=minutes)).isoformat()


def _deliver(db, fact, *, delivery, session="s1", path="session_init", role="recall", position=0,
             delivered=1, included=1, minutes=10, profile="default", client="claude"):
    db.execute(
        "INSERT OR IGNORE INTO injection_deliveries (delivery_id, profile_id, session_id, client, "
        "path, design, holdout_state, epsilon, eligible_count, candidates, delivered_at) "
        "VALUES (?, ?, ?, ?, ?, 'one-out-v2', 'on', 0.05, 1, 1, ?)",
        (delivery, profile, session, client, path, _ago(minutes)))
    db.execute(
        "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position, score, "
        "eligible, exclusion, propensity, included, delivered) VALUES (?, ?, ?, ?, ?, 0.5, ?, ?, ?, ?, ?)",
        (delivery, fact.fact_id, profile, role, position, 1 if role == "recall" else 0,
         None if role == "recall" else "protected", 0.95 if role == "recall" else 1.0, included, delivered))


def _read(db, **kw):
    conn = open_read_only(str(db.db_path))
    try:
        return delivered.recent(conn, "default", now=NOW, **kw)
    finally:
        conn.close()


def test_deliveries_are_grouped_by_conversation_newest_first_with_words_for_each_kind(db) -> None:
    a, b, c = (_fact(db, t) for t in ("The deploy runs on Fridays.", "Tests use pytest.", "Cue memory."))
    _deliver(db, a, delivery="d1", session="s-old", path="session_init", minutes=300)
    _deliver(db, b, delivery="d2", session="s-new", path="hook_recall", minutes=20)
    _deliver(db, c, delivery="d3", session="s-new", path="cue", minutes=5)
    got = _read(db)
    assert [g["session_id"] for g in got["groups"]] == ["s-new", "s-old"]
    new = got["groups"][0]
    assert [d["kind"] for d in new["deliveries"]] == ["cue", "prompt"]
    assert new["deliveries"][1]["kind_words"].startswith("With a prompt")
    assert got["groups"][1]["deliveries"][0]["kind"] == "session"
    assert got["deliveries"] == 3 and got["memories"] == 3


def test_each_memory_says_how_it_was_chosen_how_old_it_is_and_that_it_is_current(db) -> None:
    fact = _fact(db, "Release cadence is weekly.", saved=_ago(60 * 24 * 3))
    _deliver(db, fact, delivery="d1", role="recall", position=2)
    (memory,) = _read(db)["groups"][0]["deliveries"][0]["memories"]
    assert memory["preview"] == "Release cadence is weekly."
    assert memory["why"]["how"] == "Found by search (rank 3)"
    assert memory["why"]["standing"] == "current"
    assert "Still current." in memory["why"]["words"]
    assert memory["why"]["said"].endswith("ago")


def test_a_pinned_memory_has_no_rank_and_a_standing_rule_says_so(db) -> None:
    pin, rule = _fact(db, "Pinned fact."), _fact(db, "Always ask first.")
    _deliver(db, pin, delivery="d1", role="pinned")
    _deliver(db, rule, delivery="d1", role="standing", position=1)
    hows = [m["why"]["how"] for m in _read(db)["groups"][0]["deliveries"][0]["memories"]]
    assert hows == ["A pinned memory", "A standing rule"]


def test_a_left_out_memory_is_counted_and_never_listed(db) -> None:
    shown, also, left = _fact(db, "Shown one."), _fact(db, "Shown too."), _fact(db, "Left out one.")
    _deliver(db, shown, delivery="d1")
    _deliver(db, also, delivery="d1", position=1)
    _deliver(db, left, delivery="d1", included=0, delivered=0, position=2)
    d = _read(db)["groups"][0]["deliveries"][0]
    assert [m["preview"] for m in d["memories"]] == ["Shown one.", "Shown too."] and d["left_out"] == 1


def test_a_memory_left_out_of_one_delivery_is_not_listed_there_though_shown_in_another(db) -> None:
    fact = _fact(db, "Sometimes left out.")
    _deliver(db, fact, delivery="d1", session="s1", minutes=30, included=0, delivered=0)
    _deliver(db, fact, delivery="d2", session="s2", minutes=10)
    by_session = {g["session_id"]: g["deliveries"][0] for g in _read(db)["groups"]}
    assert by_session["s1"]["memories"] == [] and by_session["s1"]["left_out"] == 1
    assert [m["preview"] for m in by_session["s2"]["memories"]] == ["Sometimes left out."]


def test_a_memory_in_another_profile_never_appears_even_if_a_receipt_names_it(db) -> None:
    mine, theirs = _fact(db, "Mine."), _fact(db, "Theirs.", profile="work")
    _deliver(db, mine, delivery="d1")
    _deliver(db, theirs, delivery="d1", position=1)  # a receipt filed under this profile for a work fact
    previews = [m["preview"] for m in _read(db)["groups"][0]["deliveries"][0]["memories"]]
    assert previews == ["Mine."]
    assert _read(db)["memories"] == 1


def test_the_remote_view_filter_hides_a_memory_the_caller_may_not_see(db) -> None:
    ok, secret = _fact(db, "Visible."), _fact(db, "Hidden from this app.")
    _deliver(db, ok, delivery="d1")
    _deliver(db, secret, delivery="d1", position=1)
    got = _read(db, hidden=lambda ids: {i for i in ids if i == secret.fact_id})
    assert [m["preview"] for m in got["groups"][0]["deliveries"][0]["memories"]] == ["Visible."]
    assert "Hidden" not in str(got)


def test_a_disputed_memory_is_labelled_disputed(db, monkeypatch) -> None:
    fact = _fact(db, "The plan costs 10 dollars.")
    _deliver(db, fact, delivery="d1")
    from superlocalmemory.truth import disputes

    monkeypatch.setattr(disputes, "for_facts", lambda conn, profile, ids, hidden=None: {fact.fact_id: [{}]})
    (memory,) = _read(db)["groups"][0]["deliveries"][0]["memories"]
    assert memory["why"]["standing"] == "disputed"
    assert memory["why"]["standing_words"].startswith("Disputed")


def test_a_changed_source_file_is_labelled(db, monkeypatch) -> None:
    fact = _fact(db, "The config says port 80.")
    _deliver(db, fact, delivery="d1")
    from superlocalmemory.truth import leases

    monkeypatch.setattr(leases, "check_for_facts", lambda conn, ids: {fact.fact_id: {"state": "changed"}})
    (memory,) = _read(db)["groups"][0]["deliveries"][0]["memories"]
    assert memory["why"]["standing"] == "source changed"
    assert "source file has changed" in memory["why"]["words"]


def test_the_origin_is_named_only_while_the_origin_work_is_not_off(db, monkeypatch) -> None:
    fact = _fact(db, "From somewhere.")
    _deliver(db, fact, delivery="d1")
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "off")  # on by default; a person's own off
    assert _read(db)["groups"][0]["deliveries"][0]["memories"][0]["why"]["origin"] is None
    monkeypatch.setenv("SLM_FEATURE_TRUST_TAINT", "shadow")
    memory = _read(db)["groups"][0]["deliveries"][0]["memories"][0]
    assert memory["why"]["origin"] and "Origin: " in memory["why"]["words"]


def test_memory_text_is_one_clean_bounded_line(db) -> None:
    fact = _fact(db, "Line one\n\x07line two\u202e\u2069 " + "x" * 500)
    _deliver(db, fact, delivery="d1")
    (memory,) = _read(db)["groups"][0]["deliveries"][0]["memories"]
    assert "\n" not in memory["preview"] and "\x07" not in memory["preview"]
    assert "\u202e" not in memory["preview"] and "\u2069" not in memory["preview"]
    assert len(memory["preview"]) <= delivered.PREVIEW_CHARS


def test_a_long_delivery_lists_a_page_and_counts_the_rest(db) -> None:
    for i in range(delivered.MEMORIES_PER_DELIVERY + 3):
        _deliver(db, _fact(db, f"Memory number {i}."), delivery="d1", position=i)
    d = _read(db)["groups"][0]["deliveries"][0]
    assert len(d["memories"]) == delivered.MEMORIES_PER_DELIVERY and d["more"] == 3


def test_the_limit_bounds_the_deliveries_read(db) -> None:
    for i in range(5):
        _deliver(db, _fact(db, f"M{i}"), delivery=f"d{i}", session=f"s{i}", minutes=i + 1)
    assert _read(db, limit=2)["deliveries"] == 2
    assert _read(db, limit=0)["deliveries"] == 1  # never zero, never unbounded
    assert _read(db, limit=10**6)["deliveries"] == 5


def test_a_delivery_with_no_conversation_is_its_own_group(db) -> None:
    a, b = _fact(db, "A."), _fact(db, "B.")
    _deliver(db, a, delivery="d1", session=None)
    _deliver(db, b, delivery="d2", session=None)
    assert len(_read(db)["groups"]) == 2


def test_an_unknown_path_still_reads_and_a_store_without_receipts_is_empty(db) -> None:
    fact = _fact(db, "Odd path.")
    _deliver(db, fact, delivery="d1", path="brand_new_door")
    assert _read(db)["groups"][0]["deliveries"][0]["kind"] == "other"
    db.execute("DROP TABLE injection_receipts")
    assert _read(db)["groups"][0]["deliveries"][0]["memories"] == []


def test_reading_writes_nothing(db) -> None:
    fact = _fact(db, "Read only.")
    _deliver(db, fact, delivery="d1")
    before = [tuple(r) for r in db.execute("SELECT * FROM injection_receipts")]
    _read(db)
    assert [tuple(r) for r in db.execute("SELECT * FROM injection_receipts")] == before
    conn = open_read_only(str(db.db_path))
    with pytest.raises(Exception):
        conn.execute("DELETE FROM injection_receipts")
    conn.close()


def test_ago_in_plain_units() -> None:
    assert delivered.ago(_ago(0), NOW) == "just now"
    assert delivered.ago((NOW + timedelta(minutes=5)).isoformat(), NOW) == "just now"  # a clock a little ahead
    assert delivered.ago(_ago(1), NOW) == "1 minute ago"
    assert delivered.ago(_ago(180), NOW) == "3 hours ago"
    assert delivered.ago(_ago(60 * 24), NOW) == "1 day ago"
    assert delivered.ago(_ago(60 * 24 * 90), NOW) == "3 months ago"
    assert delivered.ago("nonsense", NOW) == "" and delivered.ago(None, NOW) == ""


def test_gates_report_the_two_numbers_and_list_only_visible_memories(db, monkeypatch) -> None:
    ok, hid = _fact(db, "Harmful visible."), _fact(db, "Harmful hidden.")
    from superlocalmemory.outcomes import memory_report

    monkeypatch.setattr(memory_report, "_need_gate", lambda c, p, w: {
        "turns": 40, "skipped": 12, "alpha": 0.05, "rules_bound": 0.03})
    monkeypatch.setattr(memory_report, "_harm_gate", lambda c, p, w, h: {
        "state": "off", "tests": 4, "facts_both_arms": 4, "facts_ever_left_out": 6,
        "would_withhold": [ok.fact_id, hid.fact_id, "not-a-fact"]})
    conn = open_read_only(str(db.db_path))
    got = delivered.gates(conn, "default", now=NOW, hidden=lambda ids: {i for i in ids if i == hid.fact_id})
    conn.close()
    assert got["need_gate"]["skipped"] == 12 and got["need_gate"]["turns"] == 40
    assert got["need_gate"]["rules_bound"] == 0.03
    assert got["harm_gate"]["acting"] is False and got["harm_gate"]["would_withhold_count"] == 1
    assert got["harm_gate"]["tests"] == 4 and got["harm_gate"]["facts_both_arms"] == 4
    assert got["harm_gate"]["facts_ever_left_out"] == 6
    assert [x["preview"] for x in got["harm_gate"]["would_withhold"]] == ["Harmful visible."]
    assert "Harmful hidden" not in str(got)
