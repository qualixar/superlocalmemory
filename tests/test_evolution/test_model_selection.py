# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.7.9 — evolution model selection

"""Tests for ``superlocalmemory.evolution.model_selection``.

Covers the lowest-cost-per-backend defaults and the generator != verifier
independence guard (v3.7.9). Pure logic — no network, no LLM.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

from superlocalmemory.core.config import EvolutionConfig
from superlocalmemory.evolution.model_selection import (
    CHEAPEST_CLAUDE,
    CHEAPEST_OLLAMA,
    ALT_OLLAMA,
    QUALITY_CLAUDE,
    resolve_evolution_models,
)


def _cfg(**kw: object) -> EvolutionConfig:
    return EvolutionConfig(**kw)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Defaults: lowest cost per backend
# ---------------------------------------------------------------------------


def test_defaults_claude_no_ollama_reuses_cheapest_and_warns() -> None:
    """Claude backend, no Ollama: mutation=haiku, verify=haiku (== gen).

    Per the design decision, we do NOT escalate verify to a pricier model;
    we reuse the cheapest and flag reduced independence.
    """
    m = resolve_evolution_models(_cfg(), "claude", ollama_available=False)
    assert m.mutation == CHEAPEST_CLAUDE
    assert m.confirm == CHEAPEST_CLAUDE
    assert m.verify == CHEAPEST_CLAUDE
    assert m.independent is False


def test_defaults_claude_with_ollama_uses_local_verifier() -> None:
    """Claude backend + Ollama up: verify runs on free local model."""
    m = resolve_evolution_models(_cfg(), "claude", ollama_available=True)
    assert m.mutation == CHEAPEST_CLAUDE
    assert m.verify == CHEAPEST_OLLAMA
    assert m.independent is True


def test_defaults_ollama_backend_uses_two_distinct_local_models() -> None:
    m = resolve_evolution_models(_cfg(), "ollama", ollama_available=True)
    assert m.mutation == CHEAPEST_OLLAMA
    assert m.verify == ALT_OLLAMA
    assert m.independent is True


def test_anthropic_api_backend_treated_as_claude() -> None:
    m = resolve_evolution_models(_cfg(), "anthropic", ollama_available=False)
    assert m.mutation == CHEAPEST_CLAUDE


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------


def test_sonnet_optup_keeps_cheap_independent_verifier() -> None:
    """Opt up mutation to sonnet: verify drops to haiku (cheaper + independent)."""
    m = resolve_evolution_models(
        _cfg(mutation_model="sonnet"), "claude", ollama_available=False,
    )
    assert m.mutation == QUALITY_CLAUDE
    assert m.verify == CHEAPEST_CLAUDE
    assert m.independent is True


def test_explicit_verify_override_is_respected() -> None:
    m = resolve_evolution_models(
        _cfg(verify_model="sonnet"), "claude", ollama_available=True,
    )
    assert m.verify == QUALITY_CLAUDE


def test_aliases_resolve_to_allowlisted_ids() -> None:
    m = resolve_evolution_models(
        _cfg(mutation_model="haiku", confirm_model="claude-haiku-4-5"),
        "claude", ollama_available=False,
    )
    assert m.mutation == CHEAPEST_CLAUDE
    assert m.confirm == CHEAPEST_CLAUDE


def test_explicit_equal_models_flag_not_independent() -> None:
    m = resolve_evolution_models(
        _cfg(mutation_model="haiku", verify_model="haiku"),
        "claude", ollama_available=True,
    )
    assert m.mutation == m.verify == CHEAPEST_CLAUDE
    assert m.independent is False


def test_invalid_model_falls_back_to_cheapest() -> None:
    """A forbidden/unknown configured model never dispatches; fall back safe."""
    m = resolve_evolution_models(
        _cfg(mutation_model="gpt-4-turbo"), "claude", ollama_available=False,
    )
    assert m.mutation == CHEAPEST_CLAUDE


# ---------------------------------------------------------------------------
# SkillEvolver wiring — the evolver must resolve models from its config
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


def test_evolver_defaults_to_cheapest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.evolution import skill_evolver as se

    config = SLMConfig.default()  # evolution.*_model all "" (auto)
    evolver = se.SkillEvolver(db_path=":memory:", config=config)
    monkeypatch.setattr(evolver, "_get_backend", lambda: "claude")
    monkeypatch.setattr(se, "_ollama_running", lambda: False)

    models = evolver._get_models()
    assert models.mutation == CHEAPEST_CLAUDE  # not the old hardcoded sonnet
    assert models.verify == CHEAPEST_CLAUDE
    assert models.independent is False
    # Cached on second call.
    assert evolver._get_models() is models


def test_evolver_honours_configured_mutation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core.config import SLMConfig, EvolutionConfig
    from superlocalmemory.evolution import skill_evolver as se

    config = SLMConfig.default()
    config.evolution = EvolutionConfig(mutation_model="sonnet")
    evolver = se.SkillEvolver(db_path=":memory:", config=config)
    monkeypatch.setattr(evolver, "_get_backend", lambda: "claude")
    monkeypatch.setattr(se, "_ollama_running", lambda: False)

    models = evolver._get_models()
    assert models.mutation == QUALITY_CLAUDE          # opt-up honoured
    assert models.verify == CHEAPEST_CLAUDE           # cheaper + independent
    assert models.independent is True
