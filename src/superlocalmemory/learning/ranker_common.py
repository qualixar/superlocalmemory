# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — F4.A Stage-8 H-01 fix

"""Ranker retraining helpers shared by legacy + online paths.

These functions predate the LLD-10 online retrain wiring and remain
identical in behaviour; they are factored out so both
``ranker_retrain_legacy.py`` and ``ranker_retrain_online.py`` can call
them without importing from each other.

Contract refs:
  - LLD-02 §4.6 — lambdarank retraining groups + shadow gate.
  - LLD-10 §3.2 — in-sample NDCG gate before persisting a candidate.
  - Stage 8 H-01 (architect) — file split.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = (
    "_build_training_matrix",
    "_shadow_test_improved",
    "_compute_eval_metrics",
)


def _build_training_matrix(rows: list[dict], feature_names):
    """Group rows by ``query_id``, preserve order by ``position``.

    Returns ``(X, y_int, group_counts)``. ``group_counts`` is ``None``
    when no groups are discoverable (empty input).
    """
    import numpy as np
    from superlocalmemory.learning.labeler import label_for_row

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        qid = row.get("query_id") or ""
        grouped.setdefault(qid, []).append(row)
    if not grouped:
        return np.zeros((0, len(feature_names)), dtype=np.float32), [], None

    xs: list[list[float]] = []
    ys: list[int] = []
    group_counts: list[int] = []
    for qid, group_rows in grouped.items():
        # Sort by position ascending; missing positions land at the end.
        group_rows = sorted(
            group_rows,
            key=lambda r: (
                r.get("position") if r.get("position") is not None else 10**9
            ),
        )
        for r in group_rows:
            feats = r.get("features") or {}
            xs.append([float(feats.get(n, 0.0)) for n in feature_names])
            ys.append(label_for_row(r))
        group_counts.append(len(group_rows))

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int32)
    return X, y, group_counts


def _shadow_test_improved(prior_row, booster_new, rows, feature_names) -> bool:
    """Return True iff new booster beats prior on NDCG@10 with p<0.05.

    Lightweight paired t-test across per-query NDCG@10 scores.
    ``prior_row`` is the dict returned by ``load_active_model`` — it
    may be unusable (missing state_bytes / unparseable); in that case
    we promote.
    """
    try:
        import numpy as np
        import lightgbm as lgb
    except ImportError:  # pragma: no cover
        return True

    try:
        prior_booster = lgb.Booster(
            model_str=bytes(prior_row["state_bytes"]).decode("utf-8"),
        )
    except Exception:
        return True  # prior unusable → promote new.

    X, y, groups = _build_training_matrix(rows, feature_names)
    if groups is None or not groups:
        return True

    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)

    def _ndcg_at_k(scores, labels, k=10):
        order = np.argsort(-scores)
        gains_map = [0, 1, 3, 7, 15]
        dcg = 0.0
        for i, idx in enumerate(order[:k]):
            l = int(labels[idx])
            if 0 <= l < len(gains_map):
                dcg += gains_map[l] / np.log2(i + 2)
        ideal = sorted(labels.tolist(), reverse=True)[:k]
        idcg = sum(
            (gains_map[int(l)] if 0 <= int(l) < len(gains_map) else 0)
            / np.log2(i + 2)
            for i, l in enumerate(ideal)
        )
        return dcg / idcg if idcg > 0 else 0.0

    old_ndcgs: list[float] = []
    new_ndcgs: list[float] = []
    for i in range(len(groups)):
        lo, hi = offsets[i], offsets[i + 1]
        if hi - lo < 2:
            continue
        Xg, yg = X[lo:hi], y[lo:hi]
        try:
            s_old = prior_booster.predict(Xg)
            s_new = booster_new.predict(Xg)
        except Exception:
            return False
        old_ndcgs.append(_ndcg_at_k(s_old, yg))
        new_ndcgs.append(_ndcg_at_k(s_new, yg))

    if not old_ndcgs:
        return True
    old_arr = np.asarray(old_ndcgs)
    new_arr = np.asarray(new_ndcgs)
    delta = float(np.mean(new_arr - old_arr))
    if delta < 0.02:
        return False

    # Paired t-test — small-sample safe.
    diff = new_arr - old_arr
    n = len(diff)
    if n < 2:
        return True
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return mean > 0
    t_stat = mean / (std / np.sqrt(n))
    # Rough threshold: t > 2.0 (~p<0.05 for n ≥ 10 two-tailed).
    return t_stat > 2.0


def _compute_eval_metrics(booster, rows, feature_names) -> dict:
    """Lightweight training metrics snapshot."""
    try:
        import numpy as np
        X, y, groups = _build_training_matrix(rows, feature_names)
        preds = booster.predict(X) if X.size else np.zeros(0)
        return {
            "n_rows": int(X.shape[0]),
            "n_groups": int(len(groups or [])),
            "mean_score": float(np.mean(preds)) if preds.size else 0.0,
        }
    except Exception:  # pragma: no cover
        return {}
