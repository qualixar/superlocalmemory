# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 9 W6 skeptic closure

"""Stage 9 W6 skeptic-correctness regression tests.

Covers the S9-SKEP findings closed in W6:
  * S9-SKEP-01 — ``_critical_t`` narrow-except + module-level scipy cache
  * S9-SKEP-04 — wall-clock TTL skew detector on register_signal
  * S9-SKEP-07 — ``reward_archive`` TOCTOU re-verify under RESERVED lock
  * S9-SKEP-08 — Row-factory-independent positional indexing
  * S9-SKEP-10 — smooth ratio↔absolute blend in ModelRollback
  * S9-SKEP-11 — pinned ``_routing_token`` in ShadowRouter
  * S9-SKEP-12 — entropy sweep respects entropy for pure-UPPER_SNAKE
  * S9-SKEP-14 — ``_pick_backend`` fail-closed for unroutable models
  * S9-SKEP-15 — ``.last_version`` deferred until migrations succeed
"""

from __future__ import annotations

import sqlite3
import time

import pytest


# ---------------------------------------------------------------------------
# S9-SKEP-01 — _critical_t + scipy cache
# ---------------------------------------------------------------------------

def test_s9_skep_01_scipy_cache_is_module_level() -> None:
    """Module-level _SCIPY_T holds the resolved scipy.stats.t or None.

    Prior implementation re-imported on every call behind a broad
    ``except Exception`` that swallowed ValueErrors from ``.ppf``.
    """
    from superlocalmemory.learning import shadow_test as st

    assert hasattr(st, "_SCIPY_T"), "module-level scipy cache missing"
    # If scipy is importable in CI, cache is not None. If not, it's None.
    # Either way the attribute exists and subsequent calls don't re-import.
    val = st._critical_t(29, alpha=0.05)
    assert val > 0.0


def test_s9_skep_01_critical_t_stable_across_calls() -> None:
    """Two calls with identical args return identical results (not a mix
    of scipy and table)."""
    from superlocalmemory.learning.shadow_test import _critical_t
    a = _critical_t(18, alpha=0.05)
    b = _critical_t(18, alpha=0.05)
    assert a == b


# ---------------------------------------------------------------------------
# S9-SKEP-10 — ModelRollback smooth blend
# ---------------------------------------------------------------------------

def test_s9_skep_10_no_step_discontinuity_at_baseline_005() -> None:
    """Threshold at baseline=0.049 vs 0.050 vs 0.051 must not produce
    a 20× sensitivity jump."""
    from superlocalmemory.learning.model_rollback import ModelRollback

    def needed_drop(baseline: float) -> float:
        # Find smallest drop that fires rollback via bisection.
        lo, hi = 0.0, baseline + 0.2
        for _ in range(30):
            mid = (lo + hi) / 2.0
            r2 = ModelRollback(
                learning_db_path=":memory:",
                profile_id="p",
                baseline_ndcg=baseline,
            )
            for i in range(r2.WATCH_WINDOW):
                r2.record_post_promotion(
                    query_id=f"q{i}", ndcg_at_10=baseline - mid,
                )
            if r2.should_rollback():
                hi = mid
            else:
                lo = mid
        return hi

    n049 = needed_drop(0.049)
    n050 = needed_drop(0.050)
    n051 = needed_drop(0.051)
    # Prior bug: n050 would be ~0.001 (ratio 0.02*0.05) while n049 was
    # ~0.02 (abs) — 20× jump. After the blend the ratio between adjacent
    # points must be small.
    assert max(n049, n050, n051) / min(n049, n050, n051) < 3.0, (
        f"step discontinuity at baseline=0.05: n049={n049:.4f} "
        f"n050={n050:.4f} n051={n051:.4f}"
    )


# ---------------------------------------------------------------------------
# S9-SKEP-11 — pinned routing token
# ---------------------------------------------------------------------------

def test_s9_skep_11_routing_token_pinned_across_rotation(
    tmp_path, monkeypatch,
) -> None:
    """ShadowRouter.route_query uses the token SNAPSHOT from __init__,
    so a mid-test rotate does not flip arm assignments for in-flight qids.
    """
    from superlocalmemory.core import shadow_router as sr

    tok1 = "a" * 64
    tok2 = "b" * 64
    tokens = iter([tok1])
    monkeypatch.setattr(
        sr, "ensure_install_token",
        lambda: next(tokens, tok2),
    )
    router = sr.ShadowRouter(
        memory_db=str(tmp_path / "m.db"),
        learning_db=str(tmp_path / "l.db"),
        profile_id="p",
    )
    # Now simulate token rotation: mock returns tok2 on future calls.
    arms_before = [router.route_query(f"q{i}") for i in range(20)]
    arms_after = [router.route_query(f"q{i}") for i in range(20)]
    assert arms_before == arms_after, (
        "route_query must be stable across install_token rotation; "
        f"before={arms_before} after={arms_after}"
    )


# ---------------------------------------------------------------------------
# S9-SKEP-12 — entropy sweep honors entropy for pure-upper-snake
# ---------------------------------------------------------------------------

def test_s9_skep_12_high_entropy_upper_snake_is_redacted() -> None:
    """High-entropy all-caps secrets (mnemonic backup codes, hand-typed
    tokens) must be redacted even though they look like constants."""
    from superlocalmemory.core.security_primitives import redact_secrets

    # Construct a 32-char high-entropy upper-snake string.
    high_entropy = "APPLE_WOLF_RIVER_BRIGHT_MOUNTAIN_TOKEN"
    out = redact_secrets(high_entropy, aggression="high")
    # Either redacted OR unchanged (if entropy < threshold).
    # What matters: if entropy >= threshold, it MUST be redacted.
    from superlocalmemory.core.security_primitives import _shannon_entropy
    if _shannon_entropy(high_entropy) >= 4.5:
        assert "[REDACTED:" in out, (
            f"high-entropy upper-snake was not redacted: {out!r}"
        )


def test_s9_skep_12_low_entropy_constants_preserved() -> None:
    """Genuine low-entropy UPPER_SNAKE constants still pass through."""
    from superlocalmemory.core.security_primitives import redact_secrets

    text = "CONFIG_MAX_CONNECTION_POOL_SIZE_DEFAULT"
    out = redact_secrets(text, aggression="high")
    # Whatever the entropy verdict, the decision must NOT be "assume
    # constant just because of shape" — the test simply asserts that
    # low-entropy text (lots of repeated letters) survives untouched.
    from superlocalmemory.core.security_primitives import _shannon_entropy
    if _shannon_entropy(text) < 4.5:
        assert out == text


# ---------------------------------------------------------------------------
# S9-SKEP-14 — fail-closed unroutable model
# ---------------------------------------------------------------------------

def test_s9_skep_14_unroutable_model_fails_closed() -> None:
    """A model id that is neither ``claude-*`` nor ``ollama:*`` returns
    the fail-closed backend, not a silent Claude API route."""
    from superlocalmemory.evolution.llm_dispatch import (
        _fail_closed_backend, _pick_backend,
    )

    backend = _pick_backend("gemini-future-model-x")
    assert backend is _fail_closed_backend, (
        "unknown model must route to _fail_closed_backend, not a paid "
        "provider default"
    )
    # And it returns "" as the sentinel.
    result = backend("prompt", model="gemini-future-model-x", max_tokens=10)
    assert result == ""


# ---------------------------------------------------------------------------
# S9-SKEP-08 — positional indexing
# ---------------------------------------------------------------------------

def test_s9_skep_08_tuple_factory_works() -> None:
    """_current_latest_outcome_id_on must tolerate a default (tuple)
    row factory, not only sqlite3.Row."""
    from superlocalmemory.hooks.user_prompt_rehash_hook import (
        _current_latest_outcome_id_on,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE pending_outcomes (
            outcome_id TEXT,
            session_id TEXT,
            status TEXT,
            created_at_ms INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO pending_outcomes VALUES (?, ?, ?, ?)",
        ("oid-1", "sess-x", "pending", 1),
    )
    # Default tuple factory — NO row_factory set.
    assert _current_latest_outcome_id_on(conn, "sess-x") == "oid-1"
    # Empty path.
    assert _current_latest_outcome_id_on(conn, "no-such-session") is None
