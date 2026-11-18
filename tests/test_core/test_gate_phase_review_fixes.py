# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The acting gate after the W0 phase review (W0-03, -05, -07, -10, -12; D-35)."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.judge_calibration import CANNOT_TELL, ActingGate, eprocess_certificate
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from superlocalmemory.storage.migrations import M077_judge_certificates as M077

_INSERT = ("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version,"
           " subject_kind, subject_id, population, tier, model, verdict, confidence, threshold,"
           " accepted, calibration_id, inputs_hash, decided_at, audit_selected,"
           " audit_probability, batching_regime, inbox_selected)"
           " VALUES (?, 'p', 'Q', '2', 's', ?, 'slot_dispute', ?, ?, ?, ?, 0.7, 1, 'c', ?, ?, 1,"
           " 0.1, ?, ?)")


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    return conn


def _receipts(conn, n, *, verdict="same", tier="jev", model="m", conf=0.95, regime="single",
              inbox=1, labels=None, tag="r", start=0):
    """``n`` drawn receipts decided in order; ``labels[i]`` is receipt i's label (None: none)."""
    for i in range(n):
        rid = f"{tag}{start + i}"
        conn.execute(_INSERT, (rid, rid, tier, model, verdict, conf, f"h{tag}{start + i}",
                               f"2026-11-16T{(start + i) // 3600:02d}:"
                               f"{(start + i) // 60 % 60:02d}:{(start + i) % 60:02d}",
                               regime, inbox))
        label = verdict if labels is None else labels[i]
        if label is not None:
            conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind,"
                         " labelled_at) VALUES (?, ?, 'audit', 'a', 1, ?)",
                         (rid, label, f"2027-01-01T00:00:{(n - i) % 60:02d}"))


def _gate(conn, alpha=0.05) -> ActingGate:
    return ActingGate(conn, profile_id="p", question_id="Q", version="2",
                      alpha_for=lambda verdict: alpha, delta=0.05)


def _allows(conn, *, verdict="same", regime="single", alpha=0.05, **kw) -> bool:
    return _gate(conn, alpha).allows("jev", "m", verdict, 0.95, population="slot_dispute",
                                     regime=regime, **kw)


def test_an_unlabelled_drawn_receipt_stops_the_count_whatever_was_labelled_after_it() -> None:
    conn = _store()  # W0-03: a labeller who skips the hard one cannot certify the cell
    labels = [None] + ["replaces"] * 120
    _receipts(conn, 121, verdict="replaces", labels=labels)
    assert not _allows(conn, verdict="replaces")
    conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind,"
                 " labelled_at) VALUES ('r0', 'replaces', 'audit', 'a', 1, 't')")
    assert _allows(conn, verdict="replaces")


def test_cannot_tell_counts_as_an_error_not_a_skip() -> None:
    conn = _store()  # a closing verdict, where "not a replacement" would read as clean
    _receipts(conn, 373, tier="rules", model="rules-v4", conf=None, regime="unbatched",
              labels=[CANNOT_TELL] + ["same"] * 372)
    gate = _gate(conn, alpha=0.01)
    assert not gate.allows("rules", "rules-v4", "same", None, population="slot_dispute",
                           regime="unbatched")


def test_a_certificate_stops_acting_when_new_errors_pull_e_back() -> None:
    clean = eprocess_certificate([0.99] * 73, [False] * 73, alpha=0.05, delta=0.05)
    later = eprocess_certificate([0.99] * 77, [False] * 73 + [True] * 4, alpha=0.05, delta=0.05)
    assert clean.acts and not later.acts and not later.revoked  # W0-12: no latch


def test_the_base_rate_comes_only_from_the_per_pair_inbox_sample() -> None:
    conn = _store()  # W0-05
    _receipts(conn, 110)  # a model's closing cell: 100 clean at delta/3 (W0-07)
    _receipts(conn, 300, verdict="replaces", model="inbox-model", inbox=0, tag="x")
    assert not _allows(conn)  # replacements outside the inbox sample give no base rate
    conn.execute("UPDATE judge_receipts SET inbox_selected = 1 WHERE model = 'inbox-model'")
    assert _allows(conn)


def test_a_rule_closes_on_its_own_audits_without_a_base_rate_but_reports_its_miss_rate() -> None:
    conn = _store()  # D-35: 373 clean at alpha_close 0.01 and the full delta
    _receipts(conn, 372, tier="rules", model="rules-v4", conf=None, regime="unbatched")
    gate = _gate(conn, alpha=0.01)
    assert not gate.allows("rules", "rules-v4", "same", None, population="slot_dispute",
                           regime="unbatched")
    _receipts(conn, 1, tier="rules", model="rules-v4", conf=None, regime="unbatched",
              start=372)
    gate = _gate(conn, alpha=0.01)
    assert gate.allows("rules", "rules-v4", "same", None, population="slot_dispute",
                       regime="unbatched")
    cert = gate.certificate("rules", "rules-v4", "same", "slot_dispute", "unbatched")
    assert cert.base_rate_lower is None and 0 < cert.false_keep_upper < 0.01


def test_a_model_closing_verdict_still_needs_the_base_rate() -> None:
    conn = _store()
    _receipts(conn, 600)  # plenty of clean audits, no replacement in the inbox
    assert not _allows(conn)


@pytest.mark.parametrize("routing,ok", [(None, True), (0.90, True), (0.97, False)])
def test_the_routing_threshold_must_sit_inside_the_certified_chain(routing, ok) -> None:
    conn = _store()  # W0-10
    _receipts(conn, 80, verdict="replaces")
    assert _gate(conn).certificate("jev", "m", "replaces", "slot_dispute", "single"
                                   ).threshold == 0.50
    assert _allows(conn, verdict="replaces", routing_threshold=routing) is ok


def test_audit_probability_is_one_for_a_retiring_verdict() -> None:
    from superlocalmemory.core.risk_policy import AUDIT_PROBABILITY, audit_probability_for

    assert audit_probability_for("replaces", AUDIT_PROBABILITY) == 1.0  # D-35
    assert audit_probability_for("same", AUDIT_PROBABILITY) == AUDIT_PROBABILITY
    assert audit_probability_for("replaces", None) is None


def test_the_inbox_draw_is_per_pair_whatever_tiers_answered_it() -> None:
    from superlocalmemory.core.audit_sampler import inbox_select

    key = b"k" * 32
    assert inbox_select(key, "pair-1", 0.5) == inbox_select(key, "pair-1", 0.5)
    hits = sum(inbox_select(key, f"pair-{i}", 0.1) for i in range(20_000))
    assert abs(hits - 2000) < 5 * (20_000 * 0.1 * 0.9) ** 0.5
    assert not inbox_select(key, "pair-1", None)


def test_an_earlier_decision_in_one_profile_is_never_reused_in_another() -> None:
    from superlocalmemory.core.judge_questions import SLOT_RELATION
    from superlocalmemory.truth import judge_reuse  # shared by the dispute and backlog passes

    conn = _store()  # W0-15
    _receipts(conn, 1, tag="a")  # profile 'p', pair hash 'ha0', accepted
    seen = judge_reuse.earlier(conn, SLOT_RELATION.version, [("p", "ha0"), ("q", "ha0")])
    assert ("p", "ha0") in seen and ("q", "ha0") not in seen
