# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Every remote write tool changes the key's profile, and only it (4.1.21).

The host computer is using "personal"; the remote key is bound to "work". Each
tool is called the way remote access calls it (``profile_id="work"``) through
the real daemon routes and canonical writer. It must change "work" only, must
not reach a "personal" memory by id, and must leave the host's active profile
where it was. A tool that ignored the routed profile would change "personal"
(or find nothing in it) and fail here.
"""

from __future__ import annotations

import asyncio
import datetime
import sqlite3
import uuid

import pytest

from tests.test_security._routed_host import HOST, OTHER, WORK, open_host


@pytest.fixture
def host(tmp_path, monkeypatch, mock_embedder):
    with open_host(tmp_path, monkeypatch, mock_embedder) as opened:
        yield opened


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pair(host) -> tuple[str, str]:
    work = host.save("The work release train leaves on Tuesday at 14:00 UTC.", WORK, "w")
    personal = host.save("The personal dentist visit is on Friday at noon.", HOST, "p")
    return work, personal


def _profile_of(host, fact_id: str) -> str | None:
    rows = host.rows("SELECT profile_id FROM atomic_facts WHERE fact_id=?", (fact_id,))
    return rows[0]["profile_id"] if rows else None


# -- memories ------------------------------------------------------------------------------


def test_delete_memory_deletes_in_the_keys_profile_only(host) -> None:
    work, personal = _pair(host)
    refused = host.call("delete_memory", fact_id=personal)
    assert refused["success"] is False, refused
    assert _profile_of(host, personal) == HOST
    out = host.call("delete_memory", fact_id=work)
    assert out["success"] is True, out
    assert _profile_of(host, work) is None and _profile_of(host, personal) == HOST


def test_update_memory_proposes_a_correction_in_the_keys_profile(host) -> None:
    work, personal = _pair(host)
    out = host.call("update_memory", fact_id=work,
                    content="The work release train leaves on Thursday at 09:00 UTC.")
    assert out["success"] is True, out
    cases = host.rows("SELECT profile_id, predecessor_fact_id FROM correction_cases")
    assert cases == [{"profile_id": WORK, "predecessor_fact_id": work}], cases
    assert _profile_of(host, out["successor_fact_id"]) == WORK
    refused = host.call("update_memory", fact_id=personal, content="Rewritten elsewhere.")
    assert refused["success"] is False, refused
    assert len(host.rows("SELECT 1 FROM correction_cases")) == 1


def test_set_and_confirm_memory_kind_change_the_keys_profile(host) -> None:
    work, personal = _pair(host)
    out = host.call("set_memory_kind", fact_id=work, kind="decision")
    assert out["success"] is True, out
    assert host.rows("SELECT memory_kind FROM atomic_facts WHERE fact_id=?",
                     (work,)) == [{"memory_kind": "decision"}]
    refused = host.call("set_memory_kind", fact_id=personal, kind="rule")
    assert refused["success"] is False, refused
    confirmed = host.call("confirm_memory_kinds",
                          items=[{"fact_id": work, "kind": "status"},
                                 {"fact_id": personal, "kind": "status"}])
    assert confirmed["success"] is True, confirmed
    assert [i["ok"] for i in confirmed["items"]] == [True, False], confirmed
    assert host.rows("SELECT memory_kind FROM atomic_facts WHERE fact_id=?",
                     (personal,)) == [{"memory_kind": None}]


def test_core_memory_pins_only_the_keys_facts(host) -> None:
    work, personal = _pair(host)
    assert host.call("core_memory", action="pin", fact_id=work)["success"] is True
    refused = host.call("core_memory", action="pin", fact_id=personal)
    assert refused["success"] is False, refused
    listed = host.call("core_memory", action="list")
    assert [p["fact_id"] for p in listed["pinned"]] == [work], listed
    assert host.call("core_memory", action="unpin", fact_id=work)["success"] is True
    assert host.call("core_memory", action="list")["count"] == 0


def test_observe_captures_into_the_keys_profile_as_personal(host) -> None:
    out = host.call("observe", content=(
        "We decided to use PostgreSQL instead of MySQL for the work billing service "
        "because we need transactional DDL."))
    assert out.get("captured") is True, out
    rows = host.rows("SELECT profile_id, scope FROM atomic_facts WHERE content LIKE ?",
                     ("%PostgreSQL%",))
    assert rows and {(r["profile_id"], r["scope"]) for r in rows} == {(WORK, "personal")}


def test_close_session_summarises_the_keys_session(host) -> None:
    from superlocalmemory.storage.models import AtomicFact, CanonicalEntity, MemoryRecord

    db = host.engine._db
    for profile in (WORK, HOST):
        entity = db.store_entity(CanonicalEntity(profile_id=profile, canonical_name="Vendor",
                                                 entity_type="org"))
        memory_id = db.store_memory(MemoryRecord(profile_id=profile, content="s",
                                                 session_id="sess-1"))
        db.store_fact(AtomicFact(profile_id=profile, memory_id=memory_id,
                                 session_id="sess-1", canonical_entities=[entity],
                                 content=f"{profile} met the vendor about the contract."))
    out = host.call("close_session", session_id="sess-1")
    assert out["success"] is True and out["summary_events_created"] >= 1, out
    events = {r["profile_id"] for r in host.rows("SELECT profile_id FROM temporal_events")}
    assert events == {WORK}, events


# -- learning signals ----------------------------------------------------------------------


def _assertion(host, profile: str) -> str:
    assertion_id = uuid.uuid4().hex
    host.engine._db.execute(
        "INSERT INTO behavioral_assertions (id, profile_id, trigger_condition, action, "
        "confidence, created_at, updated_at) VALUES (?, ?, 't', 'a', 0.5, ?, ?)",
        (assertion_id, profile, _now(), _now()))
    return assertion_id


@pytest.mark.parametrize("tool", ["reinforce_assertion", "contradict_assertion"])
def test_assertion_feedback_changes_the_keys_assertion_only(host, tool) -> None:
    mine, theirs = _assertion(host, WORK), _assertion(host, HOST)
    assert host.call(tool, assertion_id=mine)["success"] is True
    assert host.call(tool, assertion_id=theirs)["success"] is False
    confidence = {r["id"]: r["confidence"] for r in
                  host.rows("SELECT id, confidence FROM behavioral_assertions")}
    assert confidence[mine] != 0.5 and confidence[theirs] == 0.5, confidence


def test_log_tool_event_is_logged_for_the_keys_profile(host) -> None:
    out = host.call("log_tool_event", tool_name="Bash", session_id="s1")
    assert out["success"] is True, out
    assert host.rows("SELECT profile_id FROM tool_events") == [{"profile_id": WORK}]


def test_correct_pattern_is_recorded_for_the_keys_profile(host) -> None:
    from superlocalmemory.learning.behavioral import BehavioralPatternStore

    # 4.2.3: a correction must name a pattern that exists in the key's profile.
    BehavioralPatternStore(host.engine._db.db_path).record_pattern(
        WORK, pattern_type="preference", data={"pattern_key": "pat-1"})
    out = host.call("correct_pattern", pattern_id="pat-1", correction="prefer tabs")
    assert out["success"] is True, out
    with sqlite3.connect(host.engine._db.db_path) as conn:
        rows = conn.execute("SELECT profile_id, pattern_key FROM _store_patterns "
                            "WHERE pattern_type='correction'").fetchall()
    assert rows == [(WORK, "pat-1")], rows


def test_report_outcome_is_recorded_for_the_keys_profile(host) -> None:
    work, _personal = _pair(host)
    out = host.call("report_outcome", memory_ids=work, outcome="success")
    assert out["success"] is True, out
    assert host.rows("SELECT profile_id FROM action_outcomes") == [{"profile_id": WORK}]


def test_report_feedback_trains_the_keys_profile(host) -> None:
    from superlocalmemory.mcp.tools_active import _canonical_feedback_count

    work, _personal = _pair(host)
    before = (_canonical_feedback_count(WORK) or 0, _canonical_feedback_count(HOST) or 0)
    out = host.call("report_feedback", fact_id=work, feedback="relevant", query="train")
    assert out["success"] is True, out
    after = (_canonical_feedback_count(WORK), _canonical_feedback_count(HOST) or 0)
    assert after == (before[0] + 1, before[1]), (before, after)


def test_settle_session_outcomes_settles_the_keys_session_only(host) -> None:
    db = host.engine._db
    now_ms = int(datetime.datetime.now().timestamp() * 1000)
    for profile in (WORK, HOST):
        db.execute(
            "INSERT INTO pending_outcomes (outcome_id, profile_id, session_id, "
            "recall_query_id, fact_ids_json, query_text_hash, created_at_ms, "
            "expires_at_ms) VALUES (?, ?, 'sess-1', 'q', '[]', 'h', ?, ?)",
            (f"o-{profile}", profile, now_ms, now_ms + 3_600_000))
    out = host.call("settle_session_outcomes", session_id="sess-1", finalize=True)
    assert out["success"] is True and out["selected"] == 1, out
    status = {r["profile_id"]: r["status"] for r in
              host.rows("SELECT profile_id, status FROM pending_outcomes")}
    assert status[HOST] == "pending" and status[WORK] != "pending", status


# -- views and receipts --------------------------------------------------------------------


def test_manage_view_edits_the_keys_views(host) -> None:
    from superlocalmemory.views import default_store

    out = host.call("manage_view", action="create", name="Rota", query="rota")
    assert out["success"] is True and out["profile"] == WORK, out
    store = default_store()
    assert [v.name for v in store.list(WORK)] == ["Rota"]
    assert list(store.list(HOST)) == []
    assert host.call("manage_view", action="delete", name="Rota")["success"] is True
    assert list(store.list(WORK)) == []


def _receipt_tables(host) -> None:
    from superlocalmemory.storage.migrations import M040_agent_experience_receipts as m040
    from superlocalmemory.storage.migrations import M041_external_evidence_receipts as m041

    with sqlite3.connect(host.engine._db.db_path.parent / "learning.db") as conn:
        m040.apply(conn)
        m041.apply(conn)


def _experience(profile: str) -> dict:
    return {"experience_id": f"exp-{profile}", "profile_id": profile,
            "occurred_at": "2026-08-15T00:00:00+00:00", "task_class": "code",
            "project_scope": "project-digest",
            "route": {"harness": "h", "provider": "p", "model": "m", "effort": "high",
                      "machine": "m"},
            "verification": {"authority": "deterministic_gate", "evidence_digest": "a" * 64},
            "producer_claim": "success", "terminal_status": "succeeded"}


def _turn(profile: str) -> dict:
    return {"receipt_id": f"turn-{profile}", "task_id": "task-1", "profile_id": profile,
            "project_scope": "project-digest", "query_digest": "b" * 64,
            "fact_decisions": {"fact-1": "used"}, "state": "open"}


def test_brain_receipts_are_recorded_for_the_keys_profile(host) -> None:
    _receipt_tables(host)
    assert host.call("record_agent_experience", payload=_experience(WORK))["success"] is True
    assert host.call("record_agent_experience", payload=_experience(HOST))["success"] is False
    assert host.call("record_cognitive_turn", payload=_turn(WORK))["success"] is True
    status = host.call("get_brain_evidence_status")
    assert status["agent_experience"]["experiences_total"] == 1, status
    assert status["agent_experience"]["turns_by_state"] == {"open": 1}, status
    finalized = host.call("finalize_cognitive_turn", receipt_id=f"turn-{WORK}",
                          outcome={"authority": "deterministic_gate",
                                   "receipt_digest": "c" * 64, "reference": "gate-1"})
    assert finalized == {"success": True, "durable": True, "finalized": True}, finalized
    assert host.call("get_brain_evidence_status")["agent_experience"][
        "turns_by_state"] == {"finalized": 1}
    host_status = asyncio.run(host.tools["get_brain_evidence_status"]())
    assert host_status["profile_id"] == HOST
    assert host_status["agent_experience"]["experiences_total"] == 0, host_status


# -- a profile that does not exist ---------------------------------------------------------


@pytest.mark.parametrize("tool, args", [
    ("delete_memory", {"fact_id": "f"}),
    ("update_memory", {"fact_id": "f", "content": "A new version of it."}),
])
def test_a_routed_delete_or_update_for_a_deleted_profile_says_unknown_profile(
        host, tool, args) -> None:
    """The daemon's answer, not "daemon returned 404": a key whose profile was
    deleted is told so, and that retrying will not help."""
    out = asyncio.run(host.tools[tool](**args, profile_id=OTHER))
    assert out["success"] is False and out["retryable"] is False, out
    assert out["code"] == "unknown_profile", out
    assert OTHER in out["error"] and "daemon returned 404" not in out["error"], out


def test_the_404_reader_keeps_detail_codes_and_adds_the_routes_error_code() -> None:
    """Every other caller reads ``code`` exactly as before."""
    from superlocalmemory.cli.daemon import not_found_from
    from superlocalmemory.server.routed_profile import unknown_profile_body

    routed = not_found_from(unknown_profile_body(OTHER), "/api/memories/f")
    assert (routed.code, routed.message) == ("not_found", "daemon returned 404")
    assert routed.error_code == "unknown_profile" and OTHER in routed.error_message
    plain = not_found_from({"detail": "Memory not found"}, "/api/memories/f")
    assert (plain.code, plain.message, plain.error_code) == (
        "not_found", "Memory not found", "")
    typed = not_found_from({"detail": {"code": "view_not_found", "message": "No view"}}, "/v")
    assert (typed.code, typed.message, typed.error_code) == ("view_not_found", "No view", "")
    assert not_found_from(None, "/x").code == "not_found"


@pytest.mark.parametrize("tool, args", [
    ("delete_memory", {"fact_id": "f"}), ("update_memory", {"fact_id": "f", "content": "c"}),
    ("set_memory_kind", {"fact_id": "f", "kind": "rule"}),
    ("core_memory", {"action": "list"}), ("log_tool_event", {"tool_name": "Bash"}),
    ("manage_view", {"action": "create", "name": "n", "query": "q"}),
    ("observe", {"content": "We decided to use PostgreSQL for billing."}),
    ("close_session", {"session_id": "s"}), ("report_outcome",
                                            {"memory_ids": "f", "outcome": "success"}),
    ("record_agent_experience", {"payload": _experience(OTHER)}),
])
def test_a_write_to_a_profile_that_does_not_exist_changes_nothing(host, tool, args) -> None:
    before = len(host.rows("SELECT 1 FROM atomic_facts"))
    out = asyncio.run(host.tools[tool](**args, profile_id=OTHER))
    assert out.get("success") is False or out.get("captured") is False, (tool, out)
    assert len(host.rows("SELECT 1 FROM atomic_facts")) == before
    assert not host.rows("SELECT 1 FROM profiles WHERE profile_id=?", (OTHER,))
