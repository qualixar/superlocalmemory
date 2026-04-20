# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-11 §Budget

"""Tests for ``superlocalmemory.evolution.budget``.

Covers:
  - Wall-time cap (30 min/cycle)
  - LLM call cap (10/cycle)
  - Cycles-per-day cap (3/profile/day)
  - Single-flight lock via evolution.lock (safe_resolve_identifier)
  - ``BudgetExhausted`` signalling

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from superlocalmemory.evolution.budget import (
    BudgetExhausted,
    EvolutionBudget,
    MAX_CYCLES_PER_DAY,
    MAX_LLM_CALLS_PER_CYCLE,
    MAX_WALL_TIME_SEC,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_learning_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE evolution_config (
            profile_id        TEXT PRIMARY KEY,
            enabled           INTEGER NOT NULL DEFAULT 0,
            llm_backend       TEXT NOT NULL DEFAULT 'haiku',
            llm_model         TEXT NOT NULL DEFAULT 'claude-haiku-4-5',
            last_cycle_at     TEXT,
            cycles_this_week  INTEGER NOT NULL DEFAULT 0,
            disabled_until    TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE evolution_llm_cost_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id    TEXT NOT NULL,
            ts            TEXT NOT NULL,
            model         TEXT NOT NULL,
            tokens_in     INTEGER NOT NULL DEFAULT 0,
            tokens_out    INTEGER NOT NULL DEFAULT 0,
            cost_usd      REAL NOT NULL DEFAULT 0.0,
            cycle_id      TEXT
        )
        """
    )
    conn.commit()
    return conn


@pytest.fixture()
def learning_db(tmp_path: Path) -> Path:
    db = tmp_path / "learning.db"
    conn = _make_learning_db(db)
    conn.close()
    return db


@pytest.fixture()
def lock_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".superlocalmemory"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# test_evolution_single_flight_lock
# ---------------------------------------------------------------------------


def test_evolution_single_flight_lock(learning_db: Path, lock_dir: Path) -> None:
    """Two EvolutionBudget instances cannot both acquire the same profile lock.

    Resolves the lock file via ``safe_resolve_identifier`` — the file name
    pattern is ``evolution-<profile>.lock`` so the untrusted profile_id
    passes the identifier regex.
    """
    b1 = EvolutionBudget(
        profile_id="default",
        learning_db=learning_db,
        lock_dir=lock_dir,
    )
    b2 = EvolutionBudget(
        profile_id="default",
        learning_db=learning_db,
        lock_dir=lock_dir,
    )

    with b1.cycle() as ctx1:
        assert ctx1 is b1
        # Second acquire must fail fast — raises BudgetExhausted subclass.
        with pytest.raises(BudgetExhausted):
            with b2.cycle():
                pass  # pragma: no cover — should not reach here

    # After b1 released, b2 can acquire.
    with b2.cycle():
        pass


# ---------------------------------------------------------------------------
# test_evolution_budget_30min_wall_time
# ---------------------------------------------------------------------------


def test_evolution_budget_30min_wall_time(
    learning_db: Path, lock_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After wall-time elapses, ``check_time()`` raises ``BudgetExhausted``."""
    assert MAX_WALL_TIME_SEC == 1800  # 30 min contract

    t = [1_000_000.0]

    def _fake_monotonic() -> float:
        return t[0]

    monkeypatch.setattr(
        "superlocalmemory.evolution.budget.time.monotonic",
        _fake_monotonic,
    )

    budget = EvolutionBudget(
        profile_id="default",
        learning_db=learning_db,
        lock_dir=lock_dir,
    )

    with budget.cycle():
        # Advance time past the cap.
        t[0] += MAX_WALL_TIME_SEC - 1
        budget.check_time()  # still ok
        t[0] += 2
        with pytest.raises(BudgetExhausted, match="wall_time"):
            budget.check_time()


# ---------------------------------------------------------------------------
# test_evolution_budget_10_llm_calls
# ---------------------------------------------------------------------------


def test_evolution_budget_10_llm_calls(
    learning_db: Path, lock_dir: Path,
) -> None:
    """After 10 LLM calls are charged, the next charge raises."""
    assert MAX_LLM_CALLS_PER_CYCLE == 10

    budget = EvolutionBudget(
        profile_id="default",
        learning_db=learning_db,
        lock_dir=lock_dir,
    )

    with budget.cycle():
        for _ in range(MAX_LLM_CALLS_PER_CYCLE):
            budget.charge_llm_call()
        with pytest.raises(BudgetExhausted, match="llm_calls"):
            budget.charge_llm_call()


# ---------------------------------------------------------------------------
# test_evolution_cycles_per_day_cap_3
# ---------------------------------------------------------------------------


def test_evolution_cycles_per_day_cap_3(
    learning_db: Path, lock_dir: Path,
) -> None:
    """A 4th cycle in the same UTC day must refuse to start."""
    assert MAX_CYCLES_PER_DAY == 3

    budget = EvolutionBudget(
        profile_id="default",
        learning_db=learning_db,
        lock_dir=lock_dir,
    )

    # Three cycles succeed.
    for _ in range(MAX_CYCLES_PER_DAY):
        with budget.cycle():
            pass

    # Fourth attempt raises.
    with pytest.raises(BudgetExhausted, match="cycles_per_day"):
        with budget.cycle():
            pass  # pragma: no cover
