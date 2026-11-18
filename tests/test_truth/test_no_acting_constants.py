# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""R-04: no hand-picked number may decide a memory change (blueprint §0 rule 4).

Every float literal in the code that judges, gates or changes memory is listed
here with what it is. A new one fails this test until it is registered: as a
policy value (blueprint §2, ``risk_policy``), a routing-only constant (which
tier to ask next, never whether to act), a constant of a stated method (§6), or
a descriptive value nothing branches on. Acting thresholds come only from
certificates (``judge_calibration.ActingGate``), never from a literal.
"""

from __future__ import annotations

import ast
import pathlib

import superlocalmemory

ROOT = pathlib.Path(superlocalmemory.__file__).parent
MODULES = sorted((ROOT / "truth").glob("*.py")) + [
    ROOT / "core" / f"{name}.py" for name in (
        "judge_ladder", "judge_tiers", "judge_calibration", "judge_questions", "audit_sampler",
        "risk_policy", "judge_certificates", "shadow_escalation", "audit_value",
        "relation_scorer", "consent_store")] + [ROOT / "outcomes" / f"{name}.py" for name in ("effects", "utility")]

#: Bounds of a probability or score: never thresholds on their own.
BOUNDS = {0.0, 1.0}

#: (module, value) -> what it is. Keep the reason exact.
ALLOWED = {
    ("risk_policy.py", 0.05): "policy: alpha_retire and delta, D-25",
    ("risk_policy.py", 0.01): "policy: alpha_close, D-25",
    ("risk_policy.py", 0.1): "policy: audit probability, D-31 (proposed)",
    ("judge_calibration.py", 0.95): "method: e-process grid fixed in advance, M-03",
    ("judge_calibration.py", 0.9): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.85): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.8): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.75): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.7): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.65): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.6): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.55): "method: e-process grid, M-03",
    ("judge_calibration.py", 0.5): "method: grid floor and the alpha < 0.5 domain check, M-03",
    ("judge_calibration.py", 5.0): "method: alternative p1 = alpha / 5, M-03",
    ("judge_calibration.py", 2.0): "method: revocation tests alpha against 2 alpha, M-03",
    ("judge_questions.py", 0.7): "routing only: default stop-escalating confidence",
    ("judge_questions.py", 0.85): "routing only: escalate 'replaces' further",
    ("judge_questions.py", 0.8): "routing only: INJECT_VALUE 'helps' (W2, shadow)",
    ("judge_questions.py", 0.6): "routing only: INJECT_VALUE 'misleads' (W2, shadow)",
    ("judge_questions.py", 0.5): "routing only: INSTRUCTION_RISK 'injection' (W2, shadow)",
    ("audit_value.py", 3.0): "descriptive: queue order, harm of a wrong retirement vs 1",
    ("audit_value.py", 0.5): "descriptive: queue order, doubt of a rule that has no score",
    ("judge_certificates.py", 0.01): "policy: drift test level split over open certificates, M-07",
    ("judge_certificates.py", 0.12): "method: KS small-sample correction (Stephens), M-07",
    ("judge_certificates.py", 0.11): "method: KS small-sample correction (Stephens), M-07",
    ("judge_certificates.py", 0.001): "method: KS series cut-off for a tiny statistic, M-07",
    ("relation_scorer.py", 0.1): "method: Tier 0.5 L2 penalty (shadow only)",
    ("relation_scorer.py", 0.5): "method: Tier 0.5 gradient step (shadow only)",
    ("relation_scorer.py", 30.0): "method: logistic overflow clamp",
    ("judge_tiers.py", 0.8): "routing only: Laya 'same' floor",
    ("slot_keys.py", 0.6): "descriptive: PATTERN_CONFIDENCE, nothing branches on it (D-10)",
    ("slot_pass.py", 60.0): "routing only: retry backoff base (seconds)",
    ("slot_pass.py", 86400.0): "routing only: retry backoff cap (seconds)",
    ("effects.py", 24.0): "policy: outcome window in hours, M-12 estimand (W0-17)",
    ("effects.py", 0.05): "policy: two-sided error level of every effect interval, M-12",
    ("effects.py", 0.1): "method: power-line effect size; tunes where the anytime interval "
                         "is tightest, never a threshold on data, M-12",
    ("effects.py", 0.8): "method: power-line power, M-12",
    ("effects.py", 0.25): "method: largest variance of a 0/1 outcome, p(1 - p) <= 1/4",
    ("effects.py", 2.0): "method: two-sided alpha / 2 and Robbins' normal-mixture boundary, M-12",
    ("utility.py", 30.0): "descriptive: a delivery's weight halves every 30 days, M-13",
    ("utility.py", 86400.0): "method: seconds per day",
    ("utility.py", 2.0): "method: half-life decay base and two-sided alpha / 2, M-13",
    ("utility.py", 0.01): "method: lower bound of the fitted concentration (optimiser only)",
    ("utility.py", 10000.0): "method: upper bound of the fitted concentration (optimiser only)",
    ("utility.py", 12.0): "method: bound of the fitted logit mean (optimiser only)",
    ("utility.py", 5.0): "method: sqrt(5) in the golden ratio of golden-section search",
    ("utility.py", 1e-07): "method: golden-section search tolerance (optimiser only)",
    ("utility.py", 1e-06): "method: golden-section search tolerance (optimiser only)",
    ("utility.py", 1e-05): "method: golden-section search tolerance (optimiser only)",
    ("effects.py", 1e-09): "method: ridge on the normal equations' diagonal (solver only)",
    ("effects.py", 1e-300): "method: a zero pivot in the normal-equations solve (solver only)",
}


def _floats(path: pathlib.Path) -> set[float]:
    return {node.value for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and type(node.value) is float}


def test_every_float_on_the_acting_path_is_registered() -> None:
    unknown = sorted((p.name, v) for p in MODULES for v in _floats(p) - BOUNDS
                     if (p.name, v) not in ALLOWED)
    assert not unknown, (
        f"unregistered numeric literals {unknown}: an acting threshold must come from a "
        "certificate; register a policy, routing-only, method or descriptive constant here")


def test_the_allow_list_has_no_stale_entries() -> None:
    present = {(p.name, v) for p in MODULES for v in _floats(p)}
    assert not set(ALLOWED) - present, sorted(set(ALLOWED) - present)


def test_routing_constants_are_labelled_routing_only_in_the_code() -> None:
    for module, value in ALLOWED:
        if ALLOWED[(module, value)].startswith("routing only"):
            text = next(p for p in MODULES if p.name == module).read_text(encoding="utf-8")
            assert "outing only" in text, f"{module} must label its routing constants"
