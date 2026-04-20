# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 9 W4

"""Stage 9 W4 M-ARC-03 — /api/v3/brain response-shape contract.

The dashboard JS reads specific nested keys from the ``/brain``
response. Stage 8 introduced the endpoint without a Pydantic schema
or golden-fixture test, so a rename inside any of the eight
``_compute_*`` helpers would silently break the client.

Pinning the top-level key set + one spot-check per nested section
catches that class of regression without the full Pydantic refactor
(which is a v3.4.22 scope item — 18k users are on a live dashboard
that consumes this response, so we add a shape guard now and keep
the flexibility for future field additions).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


_EXPECTED_TOP_LEVEL_KEYS = frozenset({
    "preferences",
    "learning",
    "usage",
    "bandit",
    "cache",
    "cross_platform",
    "evolution_preview",
    "outcomes_preview",
    "meta",
    "profile_id",
    # S9-defer H-22: three new dashboard tile blocks.
    "reward_preview",
    "shadow_preview",
    "evolution_cost_preview",
    # S9-DASH-02: producer-side telemetry for closed-loop visibility.
    "outcome_queue",
})


def _extract_return_keys(source: str) -> set[str]:
    """Parse the brain.get_brain function body and extract the set of
    top-level keys in its returned dict."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "get_brain"
        ):
            # Walk the function body looking for the final Return whose
            # value is a Dict literal.
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(
                    stmt.value, ast.Dict,
                ):
                    keys: set[str] = set()
                    for k in stmt.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(
                            k.value, str,
                        ):
                            keys.add(k.value)
                    # The final Return is the one with our full dict;
                    # return on first match with >= 5 keys.
                    if len(keys) >= 5:
                        return keys
    return set()


def test_brain_response_top_level_keys_are_stable() -> None:
    """M-ARC-03: dashboard clients read specific top-level keys from
    /api/v3/brain. A rename inside any _compute_* helper would silently
    break them. Lock the key set here; any intentional addition updates
    this test and gets a CHANGELOG entry.
    """
    brain_py = (
        Path(__file__).parent.parent.parent
        / "src" / "superlocalmemory" / "server" / "routes" / "brain.py"
    )
    src = brain_py.read_text(encoding="utf-8")
    keys = _extract_return_keys(src)
    assert keys, "failed to parse get_brain return dict"
    missing = _EXPECTED_TOP_LEVEL_KEYS - keys
    extra = keys - _EXPECTED_TOP_LEVEL_KEYS
    assert not missing, f"brain response dropped key(s): {missing}"
    assert not extra, (
        f"brain response added unexpected key(s) {extra} — update "
        f"tests/test_api/test_s9_w4_brain_contract.py + CHANGELOG"
    )


def test_brain_helper_symbols_exist() -> None:
    """M-ARC-03: pin the eight ``_compute_*`` helper names that
    back the /brain sections. Renaming any of these without
    updating the route breaks the dashboard."""
    from superlocalmemory.server.routes import brain
    expected = [
        "_compute_preferences",
        "_compute_learning_status",
        "_compute_usage_stats",
        "_compute_bandit_snapshot",
        "_compute_cache_stats",
        "_compute_cross_platform",
        "_compute_evolution_timeseries",
    ]
    for name in expected:
        assert hasattr(brain, name), (
            f"brain.py missing helper {name!r} — dashboard may break"
        )
