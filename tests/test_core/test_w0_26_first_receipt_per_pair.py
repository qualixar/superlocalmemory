# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W0-26: a pair counts once in a cell, by its first drawn receipt."""

from __future__ import annotations

import sqlite3

from superlocalmemory.core.judge_calibration import ActingGate
from superlocalmemory.storage.migrations import M072_judge_receipts as M072
from superlocalmemory.storage.migrations import M077_judge_certificates as M077

_INSERT = ("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, question_version,"
           " subject_kind, subject_id, population, tier, model, verdict, confidence, threshold,"
           " accepted, calibration_id, inputs_hash, decided_at, audit_selected,"
           " audit_probability, batching_regime, inbox_selected)"
           " VALUES (?, 'p', 'Q', '2', 's', ?, 'slot_dispute', 'jev', 'm', 'replaces', 0.95, 0.7,"
           " 1, 'c', ?, ?, 1, 0.1, 'single', 1)")


def _store() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    M077.apply(conn)
    return conn


def _receipt(conn, rid, pair, second, label) -> None:
    conn.execute(_INSERT, (rid, rid, pair, f"2026-11-16T{second // 3600:02d}:"
                                           f"{second // 60 % 60:02d}:{second % 60:02d}"))
    if label is not None:
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind,"
                     " labelled_at) VALUES (?, ?, 'audit', 'a', 1, 't')", (rid, label))


def _gate(conn) -> ActingGate:
    return ActingGate(conn, profile_id="p", question_id="Q", version="2",
                      alpha_for=lambda verdict: 0.05, delta=0.05)


def _allows(conn) -> bool:
    return _gate(conn).allows("jev", "m", "replaces", 0.95, population="slot_dispute",
                              regime="single")


def _cert_n(conn):
    return conn.execute("SELECT n, errors FROM judge_certificates").fetchall()


def test_a_pair_re_asked_in_the_cell_counts_once_whatever_the_second_receipt_says() -> None:
    conn = _store()
    for i in range(80):
        _receipt(conn, f"first{i}", f"pair{i}", i, "replaces")
    for i in range(40):  # the same pairs asked again later (a tier became available)
        _receipt(conn, f"again{i}", f"pair{i}", 1000 + i, "unrelated")  # and labelled wrong
    assert _allows(conn) is True
    assert _cert_n(conn) == [(80, 0)]  # 80 pairs, not 120 receipts; no error counted twice


def test_an_unlabelled_second_receipt_of_a_pair_does_not_stall_the_count() -> None:
    conn = _store()
    for i in range(80):
        _receipt(conn, f"first{i}", f"pair{i}", i, "replaces")
    _receipt(conn, "again0", "pair0", 2000, None)
    assert _allows(conn) is True


def test_the_first_drawn_receipt_decides_not_a_later_one() -> None:
    conn = _store()
    for i in range(130):
        _receipt(conn, f"first{i}", f"pair{i}", 10 + i, "replaces")
    _receipt(conn, "wrong-first", "pair-x", 2, "unrelated")  # a real error, drawn early
    _receipt(conn, "right-later", "pair-x", 3000, "replaces")  # a later "correction" is ignored
    assert _allows(conn) is True
    assert _cert_n(conn) == [(131, 1)]  # the error counts once and is not hidden


def test_different_pairs_are_all_counted() -> None:
    conn = _store()
    for i in range(72):
        _receipt(conn, f"r{i}", f"pair{i}", i, "replaces")
    assert _allows(conn) is False  # 72 clean labels are one short at alpha = delta = 0.05
