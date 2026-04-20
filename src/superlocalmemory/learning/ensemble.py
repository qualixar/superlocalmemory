# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §5.4

"""Bandit / LightGBM ensemble blender.

LLD reference: ``.backup/active-brain/lld/LLD-03-contextual-bandit-and-ensemble.md``
Section 5.4.

D8 blend policy (``choose_ensemble``):
  - 0..199 signals OR model is None → ``EnsembleWeights(1.0, 0.0)`` (bandit-only).
  - 200..499 signals + model         → ``EnsembleWeights(0.4, 0.6)`` (warm blend).
  - 500+ signals + model             → ``EnsembleWeights(0.2, 0.8)`` (mature).

Hard rules:
  - E1: ``bandit + lgbm == 1.0`` — asserted at construction.
  - E2: ``booster.predict`` called exactly ONCE per rerank (batched).
  - E3: no predict call when ``lgbm_weight == 0.0`` or ``model is None``.
  - E4: both score streams normalised to [0, 1] before blending.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# Thresholds come from env for ops-time override; defaults match LLD-03 §10.
_MIN_SIGNALS = int(os.environ.get("SLM_ENSEMBLE_LGBM_MIN_SIGNALS", "200"))
_DOMINANT_SIGNALS = int(
    os.environ.get("SLM_ENSEMBLE_DOMINANT_MIN_SIGNALS", "500")
)


def _parse_blend(value: str, fallback: tuple[float, float]) -> tuple[float, float]:
    """Parse 'bandit:lgbm' env var into a (bandit, lgbm) tuple."""
    try:
        a_s, b_s = value.split(":", 1)
        a, b = float(a_s), float(b_s)
        if abs((a + b) - 1.0) > 1e-6:
            return fallback
        return (a, b)
    except (ValueError, AttributeError):
        return fallback


_WARM = _parse_blend(
    os.environ.get("SLM_ENSEMBLE_BLEND_WARM", "0.4:0.6"), (0.4, 0.6),
)
_MATURE = _parse_blend(
    os.environ.get("SLM_ENSEMBLE_BLEND_MATURE", "0.2:0.8"), (0.2, 0.8),
)


# ---------------------------------------------------------------------------
# EnsembleWeights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnsembleWeights:
    """Blend weights for the bandit/LGBM ensemble.

    E1: ``bandit + lgbm`` must equal 1.0 (±1e-6 float tolerance).
    """

    bandit: float
    lgbm: float

    def __post_init__(self) -> None:
        total = self.bandit + self.lgbm
        if abs(total - 1.0) > 1e-6:
            raise AssertionError(
                f"EnsembleWeights must sum to 1.0, got {total}"
            )
        if self.bandit < 0.0 or self.lgbm < 0.0:
            raise AssertionError(
                f"EnsembleWeights must be non-negative, got "
                f"bandit={self.bandit}, lgbm={self.lgbm}"
            )


def choose_ensemble(
    signal_count: int,
    model: Any | None,
) -> EnsembleWeights:
    """Select bandit/LGBM blend per D8.

    ``model`` is typed ``Any`` to avoid importing ``ActiveModel`` at module
    load; in practice it's an ``ActiveModel | None``. Only ``model is None``
    is checked.
    """
    try:
        count = int(signal_count)
    except (TypeError, ValueError):
        count = 0
    if model is None or count < _MIN_SIGNALS:
        return EnsembleWeights(1.0, 0.0)
    if count < _DOMINANT_SIGNALS:
        return EnsembleWeights(_WARM[0], _WARM[1])
    return EnsembleWeights(_MATURE[0], _MATURE[1])


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _softmax_unit(scores: Sequence[float]) -> list[float]:
    """Normalise a score stream to [0, 1] via softmax, numerically stable.

    Preserves ordering. Returns uniform 1/N when all scores are identical.
    """
    if not scores:
        return []
    xs = list(scores)
    n = len(xs)
    m = max(xs)
    # Subtract max for numerical stability before exp.
    exps = []
    for v in xs:
        try:
            exps.append(pow(2.718281828459045, v - m))
        except OverflowError:  # pragma: no cover — m subtraction avoids this
            exps.append(0.0)
    total = sum(exps)
    if total <= 0.0:  # pragma: no cover — defensive
        return [1.0 / n] * n
    return [e / total for e in exps]


def _apply_weights_score(candidate: Any, weights: dict[str, float]) -> float:
    """Compute a scalar bandit score for a candidate under the arm weights.

    Input shape: candidate has either ``.channel_scores`` attr OR ``score``.
    For v3.4.22 the bandit-only path simply uses the already-weighted ordering
    from ``apply_channel_weights``; this helper only matters when we blend.
    """
    # Prefer pre-weighted score on the object.
    score = getattr(candidate, "score", None)
    if score is None and isinstance(candidate, dict):
        score = candidate.get("score")
    if score is None:
        # Fallback: sum channel contributions × weights.
        cs = getattr(candidate, "channel_scores", None)
        if cs is None and isinstance(candidate, dict):
            cs = candidate.get("channel_scores", {}) or {}
        cs = cs or {}
        score = sum(
            float(cs.get(name, 0.0)) * float(weights.get(name, 1.0))
            for name in ("semantic", "bm25", "entity_graph", "temporal")
        )
        ce = None
        if hasattr(candidate, "cross_encoder_score"):
            ce = getattr(candidate, "cross_encoder_score", None)
        elif isinstance(candidate, dict):
            ce = candidate.get("cross_encoder_score")
        if ce is not None:
            score += float(ce) * float(
                weights.get("cross_encoder_bias", 1.0)
            )
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# ensemble_rerank
# ---------------------------------------------------------------------------


def ensemble_rerank(
    candidates: list[Any],
    bandit_choice: Any,
    model: Any | None,
    weights: EnsembleWeights,
    query_context: dict[str, Any],
) -> list[Any]:
    """Blend bandit + LGBM scores and reorder candidates.

    E2: ``booster.predict`` called at most ONCE, via a single batched input.
    E3: short-circuits when ``weights.lgbm == 0.0`` or ``model is None``.
    E4: softmax-unit normalisation per stream before blending.

    Never raises. On error (import / predict), returns input unchanged.
    """
    if not candidates:
        return candidates

    # E3: short-circuit.
    if weights.lgbm == 0.0 or model is None:
        return list(candidates)

    try:
        import numpy as np  # noqa: PLC0415 — optional heavy dep
    except ImportError:  # pragma: no cover — optional
        logger.debug("ensemble_rerank: numpy unavailable; bandit-only path")
        return list(candidates)

    # Lazy import so the unit tests don't require lightgbm at import time.
    try:
        from superlocalmemory.learning.features import FeatureExtractor
    except ImportError:  # pragma: no cover — defensive
        return list(candidates)

    # Build batch feature matrix ONCE. PERF-v2-02: also stash a
    # ``{fact_id: features_json}`` dict on ``query_context`` under the
    # reserved key ``_precomputed_features_json`` so the downstream
    # signal_worker (which would otherwise call ``FeatureExtractor.extract``
    # again when recording signals) can reuse this work. No schema change;
    # purely a caller-opt-in cache the signal writer probes.
    try:
        import json as _json  # noqa: PLC0415 — local import keeps hot-path clean

        rows = []
        feats_cache: dict[str, str] = {}
        for c in candidates:
            result = _candidate_to_result(c)
            fv = FeatureExtractor.extract(result, query_context)
            rows.append(fv.to_list())
            fid = getattr(c, "fact_id", None) or result.get("fact_id", "")
            if fid:
                feats_cache[fid] = _json.dumps(
                    fv.features, separators=(",", ":"),
                )
        X = np.asarray(rows, dtype=np.float32)
        if isinstance(query_context, dict) and feats_cache:
            # Merge into caller's dict; do not clobber a pre-existing cache.
            existing = query_context.get("_precomputed_features_json") or {}
            if isinstance(existing, dict):
                merged = {**existing, **feats_cache}
                query_context["_precomputed_features_json"] = merged
    except Exception as exc:
        logger.debug("ensemble_rerank: feature build failed: %s", exc)
        return list(candidates)

    # E2: single batched predict call.
    booster = getattr(model, "booster", None)
    if booster is None or not hasattr(booster, "predict"):
        return list(candidates)
    try:
        lgbm_scores = booster.predict(X)
    except Exception as exc:
        logger.warning("ensemble_rerank: predict failed: %s", exc)
        return list(candidates)
    try:
        lgbm_scores = list(map(float, lgbm_scores))
    except (TypeError, ValueError):  # pragma: no cover — defensive
        return list(candidates)

    arm_weights = (
        bandit_choice.weights if hasattr(bandit_choice, "weights") else {}
    )
    bandit_scores = [
        _apply_weights_score(c, arm_weights) for c in candidates
    ]

    # E4: normalise each stream to [0, 1] via softmax before blending.
    n_lgbm = _softmax_unit(lgbm_scores)
    n_bandit = _softmax_unit(bandit_scores)

    blended = [
        weights.bandit * b + weights.lgbm * l
        for b, l in zip(n_bandit, n_lgbm)
    ]

    # Stable-sort descending so equal scores preserve original order.
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: -blended[pair[0]])
    return [c for _, c in indexed]


def _candidate_to_result(c: Any) -> dict[str, Any]:
    """Coerce a candidate (dict / dataclass / ORM row) to a feature result."""
    if isinstance(c, dict):
        return c
    if hasattr(c, "to_result_dict") and callable(c.to_result_dict):
        try:
            return c.to_result_dict()
        except Exception:  # pragma: no cover — defensive
            pass
    # Last resort: assemble from common attributes.
    return {
        "fact_id": getattr(c, "fact_id", ""),
        "score": getattr(c, "score", 0.0),
        "channel_scores": getattr(c, "channel_scores", {}) or {},
        "cross_encoder_score": getattr(c, "cross_encoder_score", None),
    }


__all__ = (
    "EnsembleWeights",
    "choose_ensemble",
    "ensemble_rerank",
)
