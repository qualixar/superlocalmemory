# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""I-10 restated (D-49): no memory change without a certificate id, a rule-certificate id, or a
recorded decision (who + channel). ``slm doctor --truth`` checks it beyond judge receipts: for
firewall promotions, proposal sets and dispute settlements (Fable A-09 d)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.truth import invariants as inv
from tests.test_core.test_coverage_backfill import _db

NOW = "2026-11-18T10:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    db = _db(tmp_path / "memory.db")
    c = sqlite3.connect(str(db.db_path))
    yield c
    c.commit()
    c.close()


def _bad(conn) -> set[str]:
    return {r.invariant for r in inv.check(conn) if r.status == inv.VIOLATED}


def _i10(conn):
    (result,) = [r for r in inv.check(conn) if r.invariant == "I-10"]
    return result


def _dispute(conn, did, state, by) -> None:
    conn.execute("INSERT INTO slot_disputes (dispute_id, slot_id, profile_id, revision_a, "
                 "revision_b, state, reason_code, opened_by, opened_at, resolved_by, resolved_at) "
                 "VALUES (?, 's', 'default', 'a', 'b', ?, 'x', 'slot_pass', ?, ?, ?)",
                 (did, state, NOW, by, NOW if by else None))


def _set(conn, sid, state, kind, ref, by=None, channel=None) -> None:
    conn.execute("INSERT INTO proposal_sets (set_id, profile_id, title, origin, state, "
                 "basis_kind, basis_ref, decided_by, decided_channel, created_at) VALUES "
                 "(?, 'default', 't', 'dream', ?, ?, ?, ?, ?, ?)",
                 (sid, state, kind, ref, by, channel, NOW))


def _trust(conn, fid, basis) -> None:
    conn.execute("INSERT INTO fact_trust (fact_id, profile_id, origin_class, taint_class, basis, "
                 "computed_at) VALUES (?, 'default', 'tool_output', 'user_confirmed', ?, ?)",
                 (fid, basis, NOW))


def test_a_clean_store_passes(conn) -> None:
    assert "I-10" not in _bad(conn)


def test_a_dispute_settled_by_an_agent_over_mcp_is_a_recorded_decision(conn) -> None:
    for i, who in enumerate(["agent:claude-code@mcp", "alice@cli", "dashboard@dashboard",
                             "client:local@http", "judge_ladder"]):
        _dispute(conn, f"ok{i}", "resolved", who)
    _dispute(conn, "open1", "open", None)
    _dispute(conn, "stale1", "dismissed", "agent:x@mcp:stale")  # dismissed: no memory changed
    assert "I-10" not in _bad(conn)


def test_a_settled_dispute_with_no_actor_or_channel_is_a_violation(conn) -> None:
    for i, who in enumerate(["", "alice", "alice@nowhere", "@cli", "person"]):
        _dispute(conn, f"bad{i}", "resolved", who)
    result = _i10(conn)
    assert result.status == inv.VIOLATED and result.count == 5


def test_a_proposal_set_applied_on_a_decision_a_certificate_or_a_rule_certificate(conn) -> None:
    conn.execute(
        "INSERT INTO judge_certificates (cert_id, profile_id, question_id, question_version, "
        "tier, model, verdict, population, regime, method, alpha, delta, n, errors, threshold, "
        "label_set_hash, issued_at) VALUES ('c1', 'default', 'SLOT_RELATION', '2', 'rules', 'm', "
        "'replaces', 'slot_dispute', 'single', 'eprocess', 0.05, 0.05, 80, 0, 0.9, 'h', ?)",
        (NOW,))
    _set(conn, "s1", "applied", "decision", "agent:x@mcp", "agent:x", "mcp")
    _set(conn, "s2", "rolled_back", "decision", "alice@dashboard", "alice", "dashboard")
    _set(conn, "s3", "applied", "rule_certificate", "c1")
    _set(conn, "s4", "applied", "person", "alice@cli", "alice", "cli")  # the name before D-49
    _set(conn, "s5", "proposed", None, None)
    _set(conn, "s6", "rejected", "decision", "agent:x@mcp", "agent:x", "mcp")
    assert "I-10" not in _bad(conn)


def test_a_proposal_set_applied_without_a_basis_that_holds_is_a_violation(conn) -> None:
    _set(conn, "b1", "applied", "certificate", "no-such-cert")
    _set(conn, "b2", "applied", "decision", "x", None, "mcp")        # no actor
    _set(conn, "b3", "applied", "decision", "x", "agent:x", None)    # no channel
    _set(conn, "b4", "applied", "decision", "x", "agent:x", "tty")   # not a channel
    result = _i10(conn)
    assert result.status == inv.VIOLATED and result.count == 4


def test_a_promotion_names_who_and_channel_or_was_grandfathered(conn) -> None:
    _trust(conn, "p1", "confirmed:mcp:agent:claude-code")
    _trust(conn, "p2", "confirmed:dashboard:dashboard")
    _trust(conn, "p3", "confirmed:grandfathered:pre-4.2.30")
    _trust(conn, "p4", "v3:remember.agent")  # not a promotion at all
    assert "I-10" not in _bad(conn)


def test_a_promotion_without_a_recorded_actor_or_channel_is_a_violation(conn) -> None:
    _trust(conn, "q1", "confirmed:tty:alice")       # not a channel
    _trust(conn, "q2", "confirmed:mcp:")            # no actor
    _trust(conn, "q3", "confirmed:")
    result = _i10(conn)
    assert result.status == inv.VIOLATED and result.count == 3


def test_an_approved_row_whose_channel_is_not_a_channel_is_a_violation(conn) -> None:
    conn.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
                 "requested_at, decided_by, decided_channel, decided_at) VALUES "
                 "('a1', 'f1', 'default', 'pin', 'approved', ?, 'x', 'mcp', ?)", (NOW, NOW))
    assert "I-10" not in _bad(conn)
    conn.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
                 "requested_at, decided_by, decided_channel, decided_at) VALUES "
                 "('a2', 'f2', 'default', 'pin', 'approved', ?, 'x', 'tty', ?)", (NOW, NOW))
    assert _i10(conn).count == 1


def test_the_sample_names_what_broke(conn) -> None:
    _dispute(conn, "d-bad", "resolved", "")
    _set(conn, "s-bad", "applied", "certificate", "nope")
    sample = " ".join(_i10(conn).sample)
    assert "d-bad" in sample and "s-bad" in sample


def test_a_rolled_back_set_still_needs_the_basis_it_was_applied_on(conn) -> None:
    _set(conn, "rb", "rolled_back", "certificate", "no-such-cert")
    assert _i10(conn).count == 1


def test_an_approval_with_an_empty_actor_is_a_violation(conn) -> None:
    conn.execute("INSERT INTO trust_approvals (approval_id, fact_id, profile_id, wanted, state, "
                 "requested_at, decided_by, decided_channel, decided_at) VALUES "
                 "('a3', 'f3', 'default', 'pin', 'approved', ?, '', 'mcp', ?)", (NOW, NOW))
    assert _i10(conn).count == 1


def test_a_store_without_a_table_skips_that_part_and_still_checks_the_rest(conn) -> None:
    conn.execute("DROP TABLE trust_approvals")
    conn.execute("DROP TABLE proposal_sets")
    _dispute(conn, "d-bad", "resolved", "")
    result = _i10(conn)
    assert result.status == inv.VIOLATED and result.count == 1
