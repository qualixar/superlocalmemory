# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-13: an acted receipt names the certificate it relied on (I-10, I-14)."""

from __future__ import annotations

import sqlite3

from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.core.judge_ladder import RulesTier
from superlocalmemory.core.judge_questions import SLOT_RELATION
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from superlocalmemory.storage.migrations import M077_judge_certificates as M077
from superlocalmemory.truth import invariants as inv
from tests.test_truth import test_backlog_pass as _backlog_tests
from tests.test_truth import test_dispute_pass as _dispute_tests
from tests.test_truth.test_backlog_pass import _calibrated
from tests.test_truth.test_backlog_pass import _run as _backlog_run
from tests.test_truth.test_dispute_pass import _calibrate, _certify_explicit, _pass, _run

# The stores these tests share with the dispute and backlog tests (fixtures by alias).
disputed = _dispute_tests.disputed
explicit = _dispute_tests.explicit
backlog = _backlog_tests.backlog

ON = {"judge.ladder": "on", "truth.slots": "on", "truth.slots.auto_apply": "on"}


def _open_cert(db, tier="laya") -> str:
    (row,) = db.execute("SELECT cert_id FROM judge_certificates WHERE tier = ? AND "
                        "voided_at IS NULL", (tier,))
    return row["cert_id"]


def test_the_dispute_action_names_its_certificate(disputed) -> None:
    db, _old, _new = disputed
    counts, _ = _run(db, "replaces", **ON)
    assert counts["resolved_replaces"] == 1
    (acted,) = db.execute("SELECT action_taken, action_ref FROM judge_receipts "
                          "WHERE action_taken IS NOT NULL")
    assert acted["action_taken"] == "resolved_replaces"
    assert acted["action_ref"] == _open_cert(db)


def test_an_explicit_key_action_names_its_rule_certificate(explicit) -> None:
    db, _old, _new = explicit
    _certify_explicit(db)
    _pass(db, RulesTier(), **ON)
    (acted,) = db.execute("SELECT action_ref FROM judge_receipts WHERE action_taken = "
                          "'resolved_replaces'")
    assert acted["action_ref"] == _open_cert(db, "rules")


def test_the_backlog_action_names_its_certificate(backlog) -> None:
    _calibrated(backlog, "both_true")
    counts, _ = _backlog_run(backlog, "both_true", **{"judge.ladder": "on"})
    assert counts["closed"] == 1
    (acted,) = backlog.execute("SELECT action_ref FROM judge_receipts WHERE action_taken = "
                               "'closed_both_true'")
    assert acted["action_ref"] == _open_cert(backlog)


def test_a_receipt_that_did_not_act_has_no_action_ref(disputed) -> None:
    db, _old, _new = disputed
    _run(db, "replaces", **{"judge.ladder": "on"})  # held for review: no auto_apply
    assert db.execute("SELECT COUNT(*) AS n FROM judge_receipts WHERE action_ref IS NOT NULL")[0]["n"] == 0


def test_the_gate_names_the_open_certificate_and_refuses_without_a_record(
        disputed, monkeypatch) -> None:
    from superlocalmemory.core import judge_certificates as records

    db, _old, _new = disputed
    _calibrate(db, "replaces")
    cell = dict(population="slot_dispute", regime="single")
    args = dict(profile_id="default", question_id=SLOT_RELATION.question_id,
                version=SLOT_RELATION.version)
    with db.raw_connection() as conn:
        gate = ActingGate(conn, **args)
        permit = gate.permit("laya", "fake", "replaces", 0.95, **cell)
        assert permit is not None and permit.cert_id == _open_cert(db)
        assert gate.allows("laya", "fake", "replaces", 0.95, **cell) is True
        conn.execute("DELETE FROM judge_certificates")  # the record is gone
        monkeypatch.setattr(records, "reconcile", lambda *a, **k: records.NONE)
        fresh = ActingGate(conn, **args)  # the labels still hold, but nothing is on record
        assert fresh.permit("laya", "fake", "replaces", 0.95, **cell) is None
        assert fresh.allows("laya", "fake", "replaces", 0.95, **cell) is False


def test_a_store_without_the_certificate_table_never_allows() -> None:
    from tests.test_core.test_gate_phase_review_fixes import _receipts

    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    _receipts(conn, 80, verdict="replaces", tier="jev", model="m")  # labels that would certify

    def gate() -> ActingGate:
        return ActingGate(conn, profile_id="p", question_id="Q", version="2",
                          alpha_for=lambda verdict: 0.05, delta=0.05)

    args = ("jev", "m", "replaces", 0.95)
    cell = dict(population="slot_dispute", regime="single")
    assert gate().allows(*args, **cell) is False  # no table, so no certificate to name
    M077.apply(conn)
    assert gate().allows(*args, **cell) is True


# -- the doctor: an acted receipt without a lawful basis is a violation -------------------------

_RECEIPT = ("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version, "
            "subject_kind, subject_id, population, tier, model, verdict, confidence, threshold, "
            "accepted, calibration_id, inputs_hash, decided_at, audit_selected, audit_probability, "
            "batching_regime, inbox_selected, action_taken, action_ref) VALUES (?, 'default', "
            "'SLOT_RELATION', '2', 's', ?, 'slot_dispute', 'laya', 'm', 'replaces', 0.9, 0.7, 1, "
            "'c', ?, '2026-11-16T00:00:00', 0, 0.1, 'single', 0, ?, ?)")


def _violated(db) -> set[str]:
    with db.raw_connection() as conn:
        return {r.invariant for r in inv.check(conn) if r.status == inv.VIOLATED}


def _acted(db, rid, ref, action="resolved_replaces") -> None:
    with db.raw_connection() as conn:
        conn.execute(_RECEIPT, (rid, rid, "h" + rid, action, ref))


def _cert(db, cert_id="cert_1", profile="default", voided=False) -> None:
    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO judge_certificates (cert_id, profile_id, question_id, question_version, "
            "tier, model, verdict, population, regime, method, alpha, delta, n, errors, "
            "threshold, label_set_hash, issued_at, voided_at, void_reason) VALUES (?, ?, "
            "'SLOT_RELATION', '2', 'laya', 'm', 'replaces', 'slot_dispute', 'single', 'eprocess', "
            "0.05, 0.05, 80, 0, 0.9, 'h', '2026-11-15', ?, ?)",
            (cert_id, profile, "2026-11-16" if voided else None, "labels" if voided else None))


def test_i10_an_acted_receipt_without_an_action_ref_is_a_violation(disputed) -> None:
    db, _old, _new = disputed
    assert "I-10" not in _violated(db)
    _acted(db, "r1", None)
    assert "I-10" in _violated(db)


def test_i10_a_certificate_a_rule_certificate_or_a_person_is_a_lawful_basis(disputed) -> None:
    db, _old, _new = disputed
    _cert(db, "cert_1")
    _cert(db, "cert_old", voided=True)  # voided later: it held when the receipt acted
    for i, ref in enumerate(["cert_1", "cert_old", "alice@cli", "alice@dashboard"]):
        _acted(db, f"ok{i}", ref)
    _acted(db, "noact", None, action=None)  # never acted: needs nothing
    assert "I-10" not in _violated(db)


def test_i10_a_made_up_other_profile_or_agent_basis_is_a_violation(disputed) -> None:
    db, _old, _new = disputed
    _cert(db, "cert_other", profile="someone-else")
    for i, ref in enumerate(["cert_missing", "cert_other", "agent@mcp", "judge_ladder", "", "@cli"]):
        _acted(db, f"bad{i}", ref)
    with db.raw_connection() as conn:
        (result,) = [r for r in inv.check(conn) if r.invariant == "I-10"]
    assert result.status == inv.VIOLATED and result.count == 6


def test_i10_is_no_longer_listed_as_not_checked() -> None:
    assert "I-10" not in inv.NOT_CHECKED_HERE
