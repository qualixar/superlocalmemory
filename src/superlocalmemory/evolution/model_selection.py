# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.7.9 — evolution model selection

"""Per-step LLM model selection for the skill-evolution pipeline.

v3.7.9: skill evolution defaults to the *lowest-cost* capable model for
whatever backend is active, and lets users override each step
(``evolution.mutation_model`` / ``verify_model`` / ``confirm_model``).

Two invariants this module enforces:

1. **Lowest cost by default.** Claude backend → Haiku; Ollama → a local
   (free) model. No step silently defaults to a premium model.
2. **Generator != verifier.** Blind verification must run on a *different*
   model from the generator, otherwise the generator effectively grades
   its own homework. When an independent verifier isn't available for
   free (Claude backend, no local Ollama), we reuse the cheapest model
   and set ``independent=False`` so the caller can log the reduced-safety
   condition — we never escalate the verifier to a premium model just to
   force independence (that would defeat the cost goal).

This module is pure: it takes ``ollama_available`` as an argument instead
of probing, so it is fully unit-testable without network access.

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Allow-listed model ids (must stay a subset of
# ``llm_dispatch.ALLOWED_LLM_MODELS``).
CHEAPEST_CLAUDE = "claude-haiku-4-5"
QUALITY_CLAUDE = "claude-sonnet-4-6"
CHEAPEST_OLLAMA = "ollama:llama3"
ALT_OLLAMA = "ollama:qwen2.5"

# Short-name → allow-listed model id. Single source of truth for aliasing;
# ``skill_evolver`` re-exports these for backward compatibility.
_MODEL_ALIASES: dict[str, str] = {
    "haiku": CHEAPEST_CLAUDE,
    "sonnet": QUALITY_CLAUDE,
    "ollama": CHEAPEST_OLLAMA,
    "ollama:llama3": CHEAPEST_OLLAMA,
    "ollama:qwen2.5": ALT_OLLAMA,
    CHEAPEST_CLAUDE: CHEAPEST_CLAUDE,
    QUALITY_CLAUDE: QUALITY_CLAUDE,
}

# Models this subsystem is allowed to dispatch. Anything resolving outside
# this set falls back to the cheapest safe default rather than dispatching
# an unvetted (or forbidden) model.
_ALLOWED: frozenset[str] = frozenset(_MODEL_ALIASES.values())


def _resolve_model_alias(alias: str) -> str:
    """Translate a short alias (``haiku``/``sonnet``) to an allow-listed id.

    Unknown aliases fall back to the cheapest Claude model — the same
    fail-safe the evolver has always used.
    """
    return _MODEL_ALIASES.get(alias, CHEAPEST_CLAUDE)


@dataclass(frozen=True)
class ResolvedModels:
    """Concrete allow-listed model id for each pipeline step."""

    mutation: str
    verify: str
    confirm: str
    independent: bool  # True when verify != mutation (blind-verify is sound)


def _cheapest_for_backend(backend: str) -> str:
    """Lowest-cost allow-listed model for a detected backend."""
    if backend == "ollama":
        return CHEAPEST_OLLAMA
    # claude CLI, anthropic API, and any non-Ollama backend evolution can
    # actually dispatch route through the cheapest Claude model.
    return CHEAPEST_CLAUDE


def _resolve_field(value: str, default: str) -> str:
    """Resolve a config field: empty → default; else alias→id, validated."""
    if not value:
        return default
    resolved = _resolve_model_alias(value)
    if resolved not in _ALLOWED:  # pragma: no cover — defensive
        logger.warning(
            "evolution: configured model %r is not allow-listed; "
            "falling back to %s", value, default,
        )
        return default
    return resolved


def _independent_verifier(
    mutation: str, *, ollama_available: bool,
) -> str:
    """Pick a cheap verifier that differs from the generator when possible.

    - Ollama generator → the *other* local model (both free, distinct).
    - Claude generator + local Ollama up → free local verifier.
    - Claude generator, no Ollama → the cheapest *different* Claude tier if
      the generator was a premium model; otherwise reuse the cheapest model
      (caller flags reduced independence — we don't pay for a premium
      verifier just to force distinctness).
    """
    if mutation.startswith("ollama:"):
        return ALT_OLLAMA if mutation != ALT_OLLAMA else CHEAPEST_OLLAMA
    if ollama_available:
        return CHEAPEST_OLLAMA
    if mutation != CHEAPEST_CLAUDE:
        return CHEAPEST_CLAUDE
    return CHEAPEST_CLAUDE


def resolve_evolution_models(
    config: object, backend: str, *, ollama_available: bool,
) -> ResolvedModels:
    """Resolve mutation/verify/confirm models for a backend.

    ``config`` is an ``EvolutionConfig`` (duck-typed: reads
    ``mutation_model`` / ``verify_model`` / ``confirm_model`` attributes,
    each an alias, allow-listed id, or "" for auto).
    """
    cheapest = _cheapest_for_backend(backend)

    mutation = _resolve_field(getattr(config, "mutation_model", ""), cheapest)
    confirm = _resolve_field(getattr(config, "confirm_model", ""), cheapest)

    verify_cfg = getattr(config, "verify_model", "")
    if verify_cfg:
        verify = _resolve_field(verify_cfg, cheapest)
    else:
        verify = _independent_verifier(
            mutation, ollama_available=ollama_available,
        )

    independent = verify != mutation
    if not independent:
        logger.warning(
            "evolution: blind-verify model == generator model (%s) — "
            "blind-verification independence is reduced. Set "
            "evolution.verify_model to a different model, or run Ollama "
            "locally, for an independent verifier.",
            mutation,
        )
    return ResolvedModels(
        mutation=mutation,
        verify=verify,
        confirm=confirm,
        independent=independent,
    )
