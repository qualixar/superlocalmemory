# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — LLD-11 §Firing

"""Tests for opt-in firing of ``SkillEvolver.run_post_session`` + Stop hook.

Covers MASTER-PLAN D3: evolution is OFF by default, so the Stop hook is a
no-op on fresh installs. Only after ``evolution.enabled=True`` is set does
the Stop hook trigger ``run_post_session``.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import EvolutionConfig, SLMConfig
from superlocalmemory.evolution.skill_evolver import SkillEvolver


# ---------------------------------------------------------------------------
# test_evolution_disabled_by_default
# ---------------------------------------------------------------------------


def test_evolution_disabled_by_default() -> None:
    """A fresh ``SLMConfig`` must have ``evolution.enabled=False``."""
    config = SLMConfig.default()
    assert isinstance(config.evolution, EvolutionConfig)
    assert config.evolution.enabled is False

    # And ``run_post_session`` must short-circuit when config.enabled is False.
    evolver = SkillEvolver(db_path=":memory:", config=config)
    result = evolver.run_post_session(session_id="s1", profile_id="default")
    assert result.get("enabled") is False


# ---------------------------------------------------------------------------
# Stop-hook integration
# ---------------------------------------------------------------------------


@pytest.fixture()
def stop_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Configure the Stop hook to run in an isolated tempdir.

    Swaps ``_daemon_post`` so no real daemon call happens, and sets
    CLAUDE_PROJECT_DIR + CLAUDE_SESSION_ID so the hook has identifiers.
    """
    from superlocalmemory.hooks import hook_handlers as hh

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-test-abc")

    # Isolate tmp markers so Stop doesn't touch real /tmp entries.
    tmp_area = tmp_path / "tmp"
    tmp_area.mkdir()
    monkeypatch.setattr(hh, "_TMP", str(tmp_area))
    monkeypatch.setattr(
        hh, "_MARKER", str(tmp_area / "slm-session-initialized"),
    )
    monkeypatch.setattr(
        hh, "_START_TIME", str(tmp_area / "slm-session-start-time"),
    )
    monkeypatch.setattr(
        hh, "_ACTIVITY_LOG", str(tmp_area / "slm-session-activity"),
    )
    monkeypatch.setattr(
        hh, "_LAST_CONSOLIDATION", str(tmp_area / ".last-consolidation"),
    )

    # Stub daemon_post — returns success, records calls.
    daemon_calls: list[tuple[str, dict]] = []

    def _fake_daemon_post(
        path: str, body: dict, timeout: float = 3.0,
    ) -> bool:
        daemon_calls.append((path, body))
        return True

    monkeypatch.setattr(hh, "_daemon_post", _fake_daemon_post)

    # Record SkillEvolver.run_post_session calls.
    evolver_calls: list[dict[str, Any]] = []

    def _record(**kwargs: Any) -> dict[str, Any]:
        evolver_calls.append(kwargs)
        return {"enabled": True, "candidates": 0, "evolved": 0, "rejected": 0}

    # Patch the module-level launcher we will introduce.
    monkeypatch.setattr(
        hh, "_launch_post_session_evolution", _record, raising=False,
    )

    return {
        "tmp_path": tmp_path,
        "daemon_calls": daemon_calls,
        "evolver_calls": evolver_calls,
        "hh": hh,
    }


def test_stop_hook_calls_run_post_session_when_enabled(
    stop_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``evolution.enabled=True``, Stop hook must trigger evolution."""
    hh = stop_env["hh"]

    # Signal enabled state via env var — the hook reads it at call time
    # so we don't need to touch config files.
    monkeypatch.setenv("SLM_EVOLUTION_ENABLED", "1")

    # Invoke the Stop handler. It calls sys.exit(0); catch it.
    with pytest.raises(SystemExit) as exc:
        hh._hook_stop()
    assert exc.value.code == 0

    assert len(stop_env["evolver_calls"]) == 1
    call = stop_env["evolver_calls"][0]
    assert call.get("session_id") == "sess-test-abc"


def test_stop_hook_noop_when_disabled(
    stop_env: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``evolution.enabled`` is false/unset, Stop hook is a no-op."""
    hh = stop_env["hh"]
    monkeypatch.delenv("SLM_EVOLUTION_ENABLED", raising=False)

    with pytest.raises(SystemExit) as exc:
        hh._hook_stop()
    assert exc.value.code == 0

    assert stop_env["evolver_calls"] == []


# ---------------------------------------------------------------------------
# SB-2: SkillEvolver._llm_call must funnel through _dispatch_llm
# ---------------------------------------------------------------------------


class _StubEvoConfig:
    """Minimal stand-in for EvolutionConfig — lets us toggle enabled freely."""
    def __init__(self) -> None:
        self.enabled = True
        self.backend = "auto"


class _StubConfig:
    def __init__(self) -> None:
        self.evolution = _StubEvoConfig()


def _enabled_config() -> _StubConfig:
    """Config with evolution.enabled=True for funnel tests (mutable stub)."""
    return _StubConfig()


def _provision_cost_log(db_path: Path) -> None:
    """Create the ``evolution_llm_cost_log`` table the budget reads from."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_llm_cost_log (
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
    finally:
        conn.close()


def test_skill_evolver_calls_dispatch_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_llm_call must delegate to evolution.llm_dispatch._dispatch_llm.

    Records the call and asserts kwargs carry model/learning_db/profile_id
    so SB-2 wiring is observable.
    """
    from superlocalmemory.evolution import skill_evolver as se_mod

    captured: list[dict[str, Any]] = []

    def _fake_dispatch(
        prompt: str, *, model: str, learning_db, profile_id: str,
        max_tokens: int = 500, cycle_id: str | None = None,
    ) -> str:
        captured.append({
            "prompt": prompt,
            "model": model,
            "learning_db": str(learning_db),
            "profile_id": profile_id,
            "max_tokens": max_tokens,
            "cycle_id": cycle_id,
        })
        return "DISPATCHED"

    monkeypatch.setattr(se_mod, "_dispatch_llm", _fake_dispatch, raising=False)

    cfg = _enabled_config()
    evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    # Force backend so _llm_call doesn't short-circuit on 'none'.
    evolver._backend = "claude"

    out = evolver._llm_call("prompt body", max_tokens=100, model="haiku")
    assert out == "DISPATCHED"
    assert len(captured) == 1
    call = captured[0]
    # The evolver translates 'haiku' -> allow-listed model id.
    assert call["model"] in {"claude-haiku-4-5", "claude-sonnet-4-6",
                             "ollama:llama3", "ollama:qwen2.5"}
    assert call["profile_id"]  # default or whatever evolver tracks
    assert call["max_tokens"] == 100


def test_skill_evolver_does_not_call_anthropic_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SkillEvolver must NEVER construct anthropic.Anthropic()/openai.OpenAI()
    itself. All API access is funneled through _dispatch_llm.
    """
    from superlocalmemory.evolution import skill_evolver as se_mod

    # Spy on dispatch — if it's called, the evolver went through the funnel.
    funnel_hits: list[str] = []

    def _fake_dispatch(prompt, *, model, learning_db, profile_id,
                       max_tokens=500, cycle_id=None):
        funnel_hits.append(model)
        return "R"

    monkeypatch.setattr(se_mod, "_dispatch_llm", _fake_dispatch, raising=False)

    # Poison anthropic import — if evolver tries to construct it, test fails.
    poison_calls: list[str] = []

    class _PoisonedAnthropic:
        def __init__(self, *a, **kw) -> None:
            poison_calls.append("anthropic")
            raise AssertionError("direct anthropic.Anthropic() forbidden")

    import types
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = _PoisonedAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic)

    cfg = _enabled_config()
    evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    evolver._backend = "anthropic"
    evolver._llm_call("hello", max_tokens=50, model="haiku")
    assert funnel_hits, "dispatch was not called — evolver bypassed funnel"
    assert poison_calls == [], "evolver constructed anthropic.Anthropic directly"


def test_skill_evolver_does_not_call_subprocess_directly_outside_safe_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SkillEvolver must not call subprocess.run directly. When the claude
    CLI backend is exercised via _dispatch_llm, the call must go through
    run_subprocess_safe.
    """
    from superlocalmemory.evolution import skill_evolver as se_mod

    direct_subproc_calls: list[str] = []

    import subprocess as real_subprocess
    real_run = real_subprocess.run

    def _trap_run(*args, **kw):  # pragma: no cover — should not be hit
        direct_subproc_calls.append(repr(args))
        return real_run(*args, **kw)

    monkeypatch.setattr(real_subprocess, "run", _trap_run)

    # Provide a fake dispatch so we don't touch real backends.
    monkeypatch.setattr(
        se_mod, "_dispatch_llm",
        lambda prompt, *, model, learning_db, profile_id,
               max_tokens=500, cycle_id=None: "ok",
        raising=False,
    )

    cfg = _enabled_config()
    evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    evolver._backend = "claude"
    evolver._llm_call("prompt", max_tokens=50, model="haiku")
    assert direct_subproc_calls == [], (
        "SkillEvolver called subprocess.run directly: "
        f"{direct_subproc_calls}"
    )


# ---------------------------------------------------------------------------
# SB-3: EvolutionBudget wired into SkillEvolver
# ---------------------------------------------------------------------------


def test_skill_evolver_holds_budget_by_default(tmp_path: Path) -> None:
    """__init__ must construct a default EvolutionBudget when none is passed."""
    from superlocalmemory.evolution.budget import EvolutionBudget

    cfg = _enabled_config()
    evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    assert hasattr(evolver, "_budget")
    assert isinstance(evolver._budget, EvolutionBudget)


def test_run_consolidation_cycle_honours_budget_cycle_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_consolidation_cycle must open a budget.cycle() context.

    We observe this by patching EvolutionBudget.cycle to count entries.
    """
    from superlocalmemory.evolution import budget as budget_mod
    from contextlib import contextmanager

    cycle_entries: list[str] = []

    real_cycle = budget_mod.EvolutionBudget.cycle

    @contextmanager
    def _spy_cycle(self, cycle_id=None):
        cycle_entries.append(cycle_id or "auto")
        with real_cycle(self, cycle_id=cycle_id) as b:
            yield b

    monkeypatch.setattr(budget_mod.EvolutionBudget, "cycle", _spy_cycle)

    cfg = _enabled_config()
    db_path = tmp_path / "x.db"
    evolver = SkillEvolver(db_path=str(db_path), config=cfg)
    _provision_cost_log(db_path)  # budget reads this table
    # Pick a real backend so consolidation enters the cycle body.
    evolver._backend = "claude"

    # Stub trigger scans so the cycle body is cheap and deterministic.
    evolver._degradation.scan = lambda _p: []  # type: ignore[assignment]
    evolver._degradation.get_active_degraded = lambda _p: []  # type: ignore[assignment]
    evolver._health.scan = lambda _p: []  # type: ignore[assignment]

    evolver.run_consolidation_cycle(profile_id="default")

    assert len(cycle_entries) == 1, (
        f"run_consolidation_cycle did not open a budget cycle: {cycle_entries}"
    )


def test_run_consolidation_cycle_aborts_on_budget_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If budget.cycle() raises BudgetExhausted, the evolver must return
    cleanly (no exception propagated to the caller)."""
    from superlocalmemory.evolution import budget as budget_mod

    def _raise(self, cycle_id=None):  # type: ignore[no-untyped-def]
        raise budget_mod.BudgetExhausted("cycles_per_day", "test")

    monkeypatch.setattr(budget_mod.EvolutionBudget, "cycle", _raise)

    cfg = _enabled_config()
    evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    evolver._backend = "claude"
    result = evolver.run_consolidation_cycle(profile_id="default")
    # Must return gracefully with a flag — no exception.
    assert isinstance(result, dict)
    assert result.get("aborted") is True or result.get("budget_exhausted") is True


def test_evolution_lock_uses_safe_resolve_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the default budget must route profile_id through
    safe_resolve_identifier (LLD-00 §4)."""
    from superlocalmemory.core import security_primitives as secp

    calls: list[tuple[Path, str]] = []
    real_resolve = secp.safe_resolve_identifier

    def _spy_resolve(base: Path, untrusted: str) -> Path:
        calls.append((base, untrusted))
        return real_resolve(base, untrusted)

    monkeypatch.setattr(secp, "safe_resolve_identifier", _spy_resolve)
    # Also patch where budget imported it (from ... import)
    from superlocalmemory.evolution import budget as budget_mod
    monkeypatch.setattr(budget_mod, "safe_resolve_identifier", _spy_resolve)

    cfg = _enabled_config()
    _evolver = SkillEvolver(db_path=str(tmp_path / "x.db"), config=cfg)
    assert calls, "safe_resolve_identifier was not used for lock path"
    # The untrusted portion must include the profile id.
    assert any("evolution-" in u for (_b, u) in calls)
