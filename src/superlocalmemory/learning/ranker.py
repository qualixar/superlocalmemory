# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""3-phase adaptive ranker — from heuristic to ML.

LLD reference: ``.backup/active-brain/lld/LLD-02-signal-pipeline-and-lightgbm.md``
Sections 4.4 + 4.5.

Phase 1: cross-encoder score only (cold start)
Phase 2: heuristic boosts (some data)
Phase 3: LightGBM **lambdarank** Booster (native, not LGBMRanker sklearn
         wrapper) scoring on numpy feature matrices.

Transitions are automatic based on accumulated training data. Feature-name
drift is handled per LLD-02 §4.5 (``drift_mode``):
    - ``aligned`` — score normally.
    - ``subset``  — pad missing features with 0.0 in FEATURE_NAMES order.
    - ``unknown`` — refuse to score; fall back to pre-model order.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from superlocalmemory.learning.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
)

logger = logging.getLogger(__name__)

# Phase thresholds
PHASE_2_THRESHOLD = 50   # signals needed to enter Phase 2
PHASE_3_THRESHOLD = 200  # signals needed to enter Phase 3


class AdaptiveRanker:
    """3-phase adaptive re-ranker for V3 retrieval results."""

    def __init__(
        self,
        signal_count: int = 0,
        model_state: bytes | None = None,
        *,
        active_model: Any = None,
    ) -> None:
        """Build a ranker.

        ``active_model`` (``model_cache.ActiveModel``) is preferred when
        available — it carries verified booster + feature_names. The legacy
        ``model_state`` bytes path remains for backward compatibility with
        3.4.20 callers; it does NOT perform SHA-256 verification and should
        not be used by the 3.4.22 recall path.
        """
        self._signal_count = signal_count
        self._active = active_model
        # Back-compat: only fill in from raw bytes when no active_model given.
        if active_model is None and model_state:
            self._load_legacy_bytes(model_state)

    # --- public properties ---------------------------------------------

    @property
    def phase(self) -> int:
        if self._signal_count >= PHASE_3_THRESHOLD and self._active is not None:
            return 3
        if self._signal_count >= PHASE_2_THRESHOLD:
            return 2
        return 1

    @property
    def signal_count(self) -> int:
        return self._signal_count

    @signal_count.setter
    def signal_count(self, value: int) -> None:
        self._signal_count = value

    @property
    def active_model(self) -> Any:
        return self._active

    # --- re-rank entry points ------------------------------------------

    def rerank(self, results: list[dict], query_context: dict) -> list[dict]:
        """Re-rank retrieval results based on current phase."""
        if not results:
            return results

        if self.phase == 3:
            return self._rerank_ml(results, query_context)
        if self.phase == 2:
            return self._rerank_heuristic(results, query_context)
        return self._rerank_baseline(results)

    def rank(self, candidates: list, query_context: dict) -> list:
        """LLD-02 §4.5 native inference path.

        Accepts an iterable of objects that implement ``to_result_dict()``
        (the signal-pipeline candidates) AND plain dicts (legacy).
        """
        if self._active is None or not candidates:
            return list(candidates)

        # Build result dicts in a uniform shape.
        result_dicts: list[dict] = []
        for c in candidates:
            if hasattr(c, "to_result_dict"):
                result_dicts.append(c.to_result_dict())
            elif isinstance(c, dict):
                result_dicts.append(c)
            else:
                # Unknown candidate type — return original order.
                return list(candidates)

        from superlocalmemory.learning.model_cache import drift_mode

        mode = drift_mode(self._active)
        if mode == "unknown":
            logger.info(
                "ranker.rank: feature-name drift unknown; "
                "falling back to pre-model order",
            )
            return list(candidates)

        # Order matrix by CURRENT FEATURE_NAMES; if subset, missing names
        # pad with 0.0 (FeatureExtractor already does this via .get(name, 0)).
        try:
            import numpy as np
        except ImportError:  # pragma: no cover — numpy is required dep
            return list(candidates)

        try:
            rows = []
            for rd in result_dicts:
                fv = FeatureExtractor.extract(rd, query_context)
                rows.append(fv.to_list())
            X = np.asarray(rows, dtype=np.float32)
            scores = self._active.booster.predict(X)
        except Exception as exc:  # pragma: no cover — booster.predict path
            logger.warning("ranker.rank: booster.predict failed: %s", exc)
            return list(candidates)

        order = np.argsort(-scores, kind="stable")
        return [candidates[int(i)] for i in order]

    # --- phase implementations -----------------------------------------

    def _rerank_baseline(self, results: list[dict]) -> list[dict]:
        return sorted(
            results,
            key=lambda r: r.get("cross_encoder_score", r.get("score", 0)),
            reverse=True,
        )

    def _rerank_heuristic(
        self, results: list[dict], query_context: dict,
    ) -> list[dict]:
        scored: list[dict] = []
        for r in results:
            base = r.get("cross_encoder_score", r.get("score", 0))
            age_days = r.get("fact", {}).get("age_days", 30)
            access_count = r.get("fact", {}).get("access_count", 0)
            recency_boost = 0.1 * math.exp(-age_days / 30)
            access_boost = 0.05 * min(access_count / 10, 1.0)
            trust_boost = 0.1 * (r.get("trust_score", 0.5) - 0.5)
            final = base + recency_boost + access_boost + trust_boost
            scored.append({**r, "_adaptive_score": final})
        return sorted(scored, key=lambda r: r["_adaptive_score"], reverse=True)

    def _rerank_ml(
        self, results: list[dict], query_context: dict,
    ) -> list[dict]:
        """Phase 3 prediction via native Booster."""
        if self._active is None:  # pragma: no cover — guarded by phase()
            return self._rerank_heuristic(results, query_context)

        from superlocalmemory.learning.model_cache import drift_mode

        mode = drift_mode(self._active)
        if mode == "unknown":
            logger.info(
                "ranker._rerank_ml: unknown drift → heuristic fallback",
            )
            return self._rerank_heuristic(results, query_context)

        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return self._rerank_heuristic(results, query_context)

        try:
            feature_vectors = FeatureExtractor.extract_batch(
                results, query_context,
            )
            X = np.asarray(
                [fv.to_list() for fv in feature_vectors],
                dtype=np.float32,
            )
            scores = self._active.booster.predict(X)
        except Exception as exc:  # pragma: no cover — booster.predict path
            logger.warning("_rerank_ml failed: %s", exc)
            return self._rerank_heuristic(results, query_context)

        order = np.argsort(-scores, kind="stable")
        return [results[int(i)] for i in order]

    # --- legacy load path (back-compat) --------------------------------

    def _load_legacy_bytes(self, state: bytes) -> None:
        """Best-effort load from raw bytes — NO SHA-256 verify.

        Kept for 3.4.20 callers. The 3.4.22 recall path uses
        ``model_cache.load_active`` which enforces verification.
        """
        try:
            import lightgbm as lgb  # noqa: PLC0415

            booster = lgb.Booster(model_str=state.decode("utf-8"))
        except Exception as exc:
            logger.warning("Legacy model load failed: %s", exc)
            self._active = None
            return

        from superlocalmemory.learning.model_cache import ActiveModel

        self._active = ActiveModel(
            profile_id="legacy",
            booster=booster,
            feature_names=tuple(FEATURE_NAMES),
            trained_at="",
            sha256="",
        )

    # --- legacy train() shim (3.4.20 API) ------------------------------

    def train(self, training_data: list) -> bool:
        """Deprecated — v3.4.22 training lives in ``consolidation_worker``.

        Kept as a guard for 3.4.20 callers: returns False when
        training_data is below the Phase-3 threshold, True after a best-
        effort native booster fit on the legacy feature dict shape
        (never persists to disk). Production training must go through
        ``consolidation_worker._retrain_ranker`` which uses real features
        + ``lambdarank`` + group + integrity persistence.
        """
        if not training_data or len(training_data) < PHASE_3_THRESHOLD:
            return False
        # Best-effort legacy path — does NOT persist, does NOT promote.
        try:
            import lightgbm as lgb  # noqa: PLC0415
            import numpy as np
        except ImportError:
            return False
        X = np.asarray(
            [[float((d.get("features") or {}).get(n, 0.0))
              for n in FEATURE_NAMES]
             for d in training_data],
            dtype=np.float32,
        )
        y = np.asarray(
            [float(d.get("label", 0.0)) for d in training_data],
            dtype=np.float32,
        )
        ds = lgb.Dataset(X, label=y, feature_name=list(FEATURE_NAMES),
                         free_raw_data=False)
        try:
            booster = lgb.train(
                {"objective": "regression", "metric": "rmse",
                 "verbosity": -1, "min_data_in_leaf": 1},
                ds, num_boost_round=10,
            )
        except Exception:  # pragma: no cover — defensive
            return False
        from superlocalmemory.learning.model_cache import ActiveModel

        self._active = ActiveModel(
            profile_id="legacy",
            booster=booster,
            feature_names=tuple(FEATURE_NAMES),
            trained_at="",
            sha256="",
        )
        return True

    # --- legacy serialiser (used by external code in 3.4.20) -----------

    def get_model_state(self) -> bytes | None:
        if self._active is None:
            return None
        try:
            return self._active.booster.model_to_string().encode("utf-8")
        except Exception:  # pragma: no cover — defensive
            return None
