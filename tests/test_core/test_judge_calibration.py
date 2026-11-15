# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Exact bounds, agreement and Learn-then-Test threshold selection."""

from __future__ import annotations

import math
import random

import pytest

from superlocalmemory.core.judge_calibration import (
    binom_cdf,
    clean_decisions_needed,
    clopper_pearson_lower,
    clopper_pearson_upper,
    cohen_kappa,
    select_threshold,
)


def test_zero_errors_bound_has_its_closed_form() -> None:
    for n in (1, 10, 59, 300):
        assert clopper_pearson_upper(0, n, 0.05) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)
    assert clean_decisions_needed(0.05, 0.05) == 59
    assert clopper_pearson_upper(0, 59, 0.05) <= 0.05 < clopper_pearson_upper(0, 58, 0.05)


def test_bounds_match_scipy_when_available() -> None:
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((0, 20), (3, 40), (10, 100), (39, 40)):
        assert binom_cdf(k, n, 0.3) == pytest.approx(stats.binom.cdf(k, n, 0.3), rel=1e-9)
        exact = stats.beta.ppf(0.95, k + 1, n - k)
        assert clopper_pearson_upper(k, n, 0.05) == pytest.approx(exact, abs=1e-8)


def test_bounds_are_ordered_and_edge_cases_hold() -> None:
    assert clopper_pearson_upper(5, 5, 0.05) == 1.0 and clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert clopper_pearson_lower(10, 10, 0.05) == pytest.approx(0.05 ** 0.1, abs=1e-9)
    assert clopper_pearson_lower(3, 10, 0.05) < 0.3 < clopper_pearson_upper(3, 10, 0.05)
    with pytest.raises(ValueError):
        clopper_pearson_upper(4, 3, 0.05)


def test_kappa() -> None:
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohen_kappa(list("abab"), list("baba")) == pytest.approx(-1.0)
    assert cohen_kappa(["x"] * 4, ["x"] * 4) == 1.0
    assert 0.0 < cohen_kappa(list("aaab"), list("aabb")) < 1.0


def test_too_few_labels_select_nothing() -> None:
    sel = select_threshold([0.99] * 30, [False] * 30, alpha=0.05, delta=0.05)
    assert not sel.acts and sel.trail == ()  # fewer than n_min = 59: nothing is even tested


def test_selection_stops_at_the_first_threshold_it_cannot_support() -> None:
    scores = [0.99] * 80 + [0.7] * 40
    wrong = [False] * 80 + [True] * 20 + [False] * 20
    sel = select_threshold(scores, wrong, alpha=0.05, delta=0.05)
    assert sel.threshold == 0.99
    assert [t.rejected for t in sel.trail] == [True, False]


def test_the_guarantee_holds_in_simulation() -> None:
    """Over many draws, the chosen threshold's TRUE error rate exceeds alpha <= delta of the time."""
    rng = random.Random(7)
    alpha, delta, failures, trials = 0.1, 0.1, 0, 400
    for _ in range(trials):
        scores = [rng.random() for _ in range(300)]
        wrong = [rng.random() < 0.35 * (1 - s) ** 2 * 3 for s in scores]  # error falls with score
        sel = select_threshold(scores, wrong, alpha=alpha, delta=delta,
                               grid=[i / 20 for i in range(19, 0, -1)])
        if sel.acts:
            t = sel.threshold
            true_rate = (sum(min(1, 1.05 * (1 - s) ** 2) for s in [t + (1 - t) * i / 999
                         for i in range(1000)]) / 1000)
            failures += true_rate > alpha
    assert failures / trials <= delta + 0.05


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        select_threshold([0.5], [True, False], alpha=0.05, delta=0.05)
    with pytest.raises(ValueError):
        select_threshold([0.5], [True], alpha=0.0, delta=0.05)
    assert math.isfinite(binom_cdf(500, 1000, 0.5))


def _gate_store(n, verdict="same", label=None, model="m", conf=0.9, population="slot_dispute",
                source="audit", blind=1):
    import sqlite3

    from superlocalmemory.storage.migrations import M072_judge_receipts as M072

    conn = sqlite3.connect(":memory:")
    M072.apply(conn)
    for i in range(n):
        conn.execute("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, "
                     "question_version, subject_kind, subject_id, population, tier, model, verdict, "
                     "confidence, threshold, accepted, calibration_id, inputs_hash, decided_at) "
                     "VALUES (?, 'p', 'Q', '2', 's', ?, ?, 'jev', ?, ?, ?, 0.7, 1, 'c', ?, 't')",
                     (f"r{i}", f"s{i}", population, model, verdict, conf, f"h{i}"))
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                     "labelled_at) VALUES (?, ?, ?, 'a', ?, ?)",
                     (f"r{i}", label or verdict, source, blind, f"2026-11-15T00:{i // 60:02d}:"
                      f"{i % 60:02d}"))
    return conn


def _gate(conn, alpha=0.05):
    from superlocalmemory.core.judge_calibration import ActingGate

    return ActingGate(conn, profile_id="p", question_id="Q", version="2",
                      alpha_for=lambda verdict: alpha, delta=0.05)


def test_acting_gate_certifies_from_blind_audit_labels_only() -> None:
    gate = _gate(_gate_store(80))
    assert gate.allows("jev", "m", "same", 0.95, population="slot_dispute")
    assert not gate.allows("jev", "m", "same", 0.45, population="slot_dispute")  # below the grid
    assert not gate.allows("jev", "m", "replaces", 0.99, population="slot_dispute")
    assert not gate.allows("jev", "other", "same", 0.99, population="slot_dispute")
    assert not gate.allows("jev", "m", "same", 0.99, population="backlog")  # per population
    assert not _gate(_gate_store(80, source="review")).allows(
        "jev", "m", "same", 0.95, population="slot_dispute")
    assert not _gate(_gate_store(80, blind=0)).allows(
        "jev", "m", "same", 0.95, population="slot_dispute")


def test_72_clean_audits_are_one_short() -> None:
    assert not _gate(_gate_store(72)).allows("jev", "m", "same", 0.95, population="slot_dispute")
    assert _gate(_gate_store(73)).allows("jev", "m", "same", 0.95, population="slot_dispute")


def test_a_closing_verdict_is_held_when_its_errors_are_hidden_replacements() -> None:
    conn = _gate_store(400, verdict="same", label="replaces")
    assert not _gate(conn).allows("jev", "m", "same", 0.95, population="slot_dispute")


def test_disagreeing_audits_count_as_an_error() -> None:
    conn = _gate_store(73)
    conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labeller, blind, "
                 "labelled_at) VALUES ('r0', 'replaces', 'audit', 'b', 1, '2026-11-15T01:00:00')")
    assert not _gate(conn).allows("jev", "m", "same", 0.95, population="slot_dispute")


def test_eprocess_boundaries() -> None:
    from superlocalmemory.core.judge_calibration import eprocess_certificate as e

    assert e([0.99] * 73, [False] * 73, alpha=0.05, delta=0.05).acts
    assert not e([0.99] * 72, [False] * 72, alpha=0.05, delta=0.05).acts
    assert e([0.99] * 113, [True] + [False] * 112, alpha=0.05, delta=0.05).acts
    assert not e([0.99] * 112, [True] + [False] * 111, alpha=0.05, delta=0.05).acts
    assert e([0.99] * 151 + [0.99] * 2, [True, True] + [False] * 151, alpha=0.05, delta=0.05).acts


def test_eprocess_revokes_on_later_errors() -> None:
    from superlocalmemory.core.judge_calibration import eprocess_certificate as e

    cert = e([0.99] * 133, [False] * 73 + [True] * 12 + [False] * 48, alpha=0.05, delta=0.05)
    assert cert.revoked and not cert.acts


def test_eprocess_skips_thresholds_too_sparse_to_certify() -> None:
    from superlocalmemory.core.judge_calibration import eprocess_certificate as e

    cert = e([0.9] * 80 + [0.6] * 80, [False] * 80 + [True] * 20 + [False] * 60,
             alpha=0.05, delta=0.05)
    assert cert.threshold == 0.65


def test_eprocess_guarantee_in_simulation() -> None:
    """At a true error rate exactly alpha, certification happens at most delta of the time
    (Ville's inequality; checked within 3 Monte-Carlo standard errors)."""
    rng = random.Random(11)
    alpha, delta, trials, hits = 0.05, 0.1, 600, 0
    for _ in range(trials):
        wrong = [rng.random() < alpha for _ in range(400)]
        hits += e_cert([0.99] * 400, wrong, alpha, delta)
    assert hits / trials <= delta + 3 * math.sqrt(delta * (1 - delta) / trials)


def e_cert(scores, wrong, alpha, delta):
    from superlocalmemory.core.judge_calibration import eprocess_certificate

    return eprocess_certificate(scores, wrong, alpha=alpha, delta=delta).acts


def test_the_default_grid_starts_where_a_threshold_can_pass() -> None:
    scores = [1 - i / 1000 for i in range(200)]
    sel = select_threshold(scores, [False] * 200, alpha=0.05, delta=0.05)
    assert sel.acts and sel.trail[0].acted == 59
    assert select_threshold(scores[:58], [False] * 58, alpha=0.05, delta=0.05).trail == ()


def test_two_risks() -> None:
    from superlocalmemory.core.judge_calibration import is_wrong

    assert is_wrong("replaces", "same") and not is_wrong("replaces", "replaces")
    assert is_wrong("same", "replaces") and not is_wrong("same", "both_true")


