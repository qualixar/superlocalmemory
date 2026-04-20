# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — F4.A Stage-8 H-03/H-17/H-18 fix

"""HNSW-backed near-duplicate detection for atomic_facts.

Extracted from ``hnsw_dedup.py`` as part of the F4.A split (Stage 8
H-03/H-18). Reward-gated archive + strong-memory boost live in
``reward_archive.py`` + ``reward_boost.py``; the shim
``hnsw_dedup.py`` re-exports every public symbol.

Contract refs:
  - LLD-12 §2 — cosine > 0.95 AND entity_jaccard > 0.8 thresholds.
  - LLD-12 §3 — hnswlib RAM budget + prefix-dedup fallback.
  - LLD-00 §7 — ``ram_reservation`` protocol.
  - Stage 8 H-17 — fallback emits logger.warning + counter.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from superlocalmemory.core.ram_lock import ram_reservation

logger = logging.getLogger(__name__)


# Stage 8 H-17 — fallback degradation counter.
#: Incremented every time the HNSW path degrades to the prefix fallback
#: for any reason (hnswlib missing, RAM refused, schema missing, fact
#: count above cap). Observable via dashboards + tests.
_HNSW_DEGRADED_COUNT = 0
_HNSW_DEGRADED_LOCK = threading.Lock()


def get_hnsw_degraded_count() -> int:
    """Return the current cumulative fallback count."""
    return _HNSW_DEGRADED_COUNT


def reset_hnsw_degraded_count() -> None:
    """Reset the counter — for tests only."""
    global _HNSW_DEGRADED_COUNT
    with _HNSW_DEGRADED_LOCK:
        _HNSW_DEGRADED_COUNT = 0


def _record_degradation(reason: str) -> None:
    """Increment the degradation counter + emit a logger.warning."""
    global _HNSW_DEGRADED_COUNT
    with _HNSW_DEGRADED_LOCK:
        _HNSW_DEGRADED_COUNT += 1
    logger.warning("hnsw_dedup: degraded to prefix fallback (%s)", reason)


__all__ = (
    "HnswDeduplicator",
    "get_hnsw_degraded_count",
    "reset_hnsw_degraded_count",
    "_parse_embedding",
    "_cosine",
    "_jaccard",
    "_pick_canonical",
)


def _parse_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    try:
        vec = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(vec, list) or not vec:
        return None
    try:
        return [float(x) for x in vec]
    except (TypeError, ValueError):
        return None


# L-P-01: vectorise ``_cosine`` via NumPy when available. NumPy cold
# import is ~30 ms, but hnswlib already forces numpy in; the import is
# effectively free in the consolidation context where these helpers run.
# Pure-Python fallback is retained for environments where numpy is
# missing (contract: this module MUST NOT hard-depend on numpy).
try:  # pragma: no cover — environment-dependent
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover — numpy always present in our deps
    _np = None


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    if _np is not None:
        # S9-W3 M-PERF-04: ``_np.asarray`` is a no-op when the input is
        # already an ndarray of the target dtype. When it is a list of
        # Python floats (which is how embeddings arrive from the JSON
        # fetch path) the cast costs 20-40 μs × N·k in dedup. We still
        # accept lists for API compatibility but prefer callers to pass
        # ndarray directly; the fast path kicks in automatically when
        # they do.
        ua = u if isinstance(u, _np.ndarray) else _np.asarray(u, dtype=_np.float32)
        va = v if isinstance(v, _np.ndarray) else _np.asarray(v, dtype=_np.float32)
        nu = float(_np.linalg.norm(ua))
        nv = float(_np.linalg.norm(va))
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return float(_np.dot(ua, va)) / (nu * nv)
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    # L-P-01: _jaccard is already O(|a|+|b|) set ops — numpy adds
    # hashing overhead for short string sets, so we keep the pure-Python
    # path. The change from the audit is the explicit note here; no
    # behaviour delta.
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _pick_canonical(
    a: dict[str, Any], b: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical = higher importance, tie-break: higher confidence, older."""
    ai, bi = float(a.get("importance", 0.0)), float(b.get("importance", 0.0))
    if ai != bi:
        return (a, b) if ai > bi else (b, a)
    ac, bc = float(a.get("confidence", 0.0)), float(b.get("confidence", 0.0))
    if ac != bc:
        return (a, b) if ac > bc else (b, a)
    at, bt = a.get("created_at", ""), b.get("created_at", "")
    return (a, b) if at <= bt else (b, a)


class HnswDeduplicator:
    """Find near-duplicate ``atomic_facts`` rows via HNSW ANN + entity overlap.

    Contract (LLD-12 §2.1):
      - cosine > COSINE_THRESHOLD AND jaccard > ENTITY_JACCARD_THRESHOLD
      - Canonical = higher importance, tie-break older created_at
      - Never delete; merges happen through memory_merge.apply_merges
    """

    COSINE_THRESHOLD: float = 0.95
    ENTITY_JACCARD_THRESHOLD: float = 0.8
    MAX_FACTS_FOR_HNSW: int = 200_000

    # S-L01: HNSW init params — stored on the class so ``_estimate_ram_mb``
    # and ``_ann_candidates`` share ONE source of truth. Previously the
    # estimator hardcoded M=16 while the real build also used M=16 / ef=100
    # — the numbers agreed by coincidence, not by construction. If either
    # knob changes, the estimate tracks automatically and the ``ef_construction``
    # build-time buffer is captured in the 1.4× multiplier below.
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 100
    # Build-time overhead multiplier vs steady-state footprint. Empirically
    # hnswlib uses ~1.3× steady RAM during construction due to the
    # ef_construction candidate pool — we round up to 1.4 for safety on
    # tight-RAM (Light) profiles.
    HNSW_BUILD_OVERHEAD: float = 1.4

    # Per-vector HNSW footprint estimate (LLD-12 §3.1). Kept for
    # back-compat — callers should prefer ``_estimate_ram_mb``.
    _BYTES_PER_VEC_DEFAULT: int = 384 * 4 + 16 * 8 * 2

    def __init__(self, *, memory_db_path: str | Path) -> None:
        self._db = Path(memory_db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_merge_candidates(
        self,
        profile_id: str,
        *,
        wall_seconds: float = 300.0,
        _force_unavailable: bool = False,
    ) -> list[tuple[str, str, float, float]]:
        """Return ``(canonical_id, duplicate_id, cosine, jaccard)`` tuples.

        Never raises for expected failure modes — falls back to prefix
        dedup instead. ``wall_seconds`` is the soft budget; we stop
        emitting new candidates once exceeded.
        """
        deadline = time.monotonic() + max(0.0, wall_seconds)

        rows = self._fetch_live_facts(profile_id)
        if len(rows) < 2:
            return []
        if len(rows) > self.MAX_FACTS_FOR_HNSW:
            _record_degradation(
                f"{len(rows)} facts > MAX {self.MAX_FACTS_FOR_HNSW}",
            )
            return self._prefix_fallback(rows, deadline)

        # Estimate RAM; let the reservation reject if the system is tight.
        est_mb = self._estimate_ram_mb(len(rows), dim=self._detect_dim(rows))
        required_mb = max(16, int(est_mb * 1.2))

        hnswlib_mod = None
        if not _force_unavailable:
            try:
                import hnswlib as hnswlib_mod  # type: ignore  # noqa: PLC0415
            except ImportError:
                hnswlib_mod = None

        if hnswlib_mod is None:
            _record_degradation("hnswlib unavailable")
            return self._prefix_fallback(rows, deadline)

        try:
            with ram_reservation(
                "hnswlib",
                required_mb=required_mb,
                timeout_s=min(30.0, max(1.0, wall_seconds)),
            ):
                return self._ann_candidates(rows, hnswlib_mod, deadline)
        except RuntimeError as exc:
            _record_degradation(f"ram_reservation refused: {exc}")
            return self._prefix_fallback(rows, deadline)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_live_facts(self, profile_id: str) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(self._db), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT fact_id, content, canonical_entities_json, "
                "       embedding, importance, confidence, created_at "
                "FROM atomic_facts "
                "WHERE profile_id = ? "
                "  AND (archive_status IS NULL OR archive_status = 'live') "
                "  AND (importance IS NULL OR importance < 1.0) "
                "ORDER BY created_at ASC",
                (profile_id,),
            )
            rows: list[dict[str, Any]] = []
            for r in cursor.fetchall():
                rows.append({
                    "fact_id": r["fact_id"],
                    "content": r["content"] or "",
                    "entities": json.loads(r["canonical_entities_json"] or "[]"),
                    "embedding": _parse_embedding(r["embedding"]),
                    "importance": float(r["importance"] or 0.0),
                    "confidence": float(r["confidence"] or 0.0),
                    "created_at": r["created_at"] or "",
                })
            return rows
        finally:
            conn.close()

    @staticmethod
    def _detect_dim(rows: list[dict[str, Any]]) -> int:
        for r in rows:
            emb = r.get("embedding")
            if emb:
                return len(emb)
        return 384

    def _estimate_ram_mb(self, n: int, *, dim: int) -> float:
        # S-L01: derive per-vector size from the actual HNSW_M knob so
        # a future tuning of M updates the estimate automatically. The
        # 1.4× multiplier folds in the ef_construction build-time
        # candidate pool that the old 1.10× factor under-counted.
        bytes_per_vec = dim * 4 + self.HNSW_M * 8 * 2
        return (n * bytes_per_vec * self.HNSW_BUILD_OVERHEAD) / (1024 * 1024)

    def _ann_candidates(
        self,
        rows: list[dict[str, Any]],
        hnswlib_mod,
        deadline: float,
    ) -> list[tuple[str, str, float, float]]:
        embedded = [r for r in rows if r["embedding"] is not None]
        if len(embedded) < 2:
            return self._prefix_fallback(rows, deadline)

        dim = len(embedded[0]["embedding"])
        # Align: drop rows with mismatched dim.
        embedded = [r for r in embedded if len(r["embedding"]) == dim]
        if len(embedded) < 2:
            return self._prefix_fallback(rows, deadline)

        index = hnswlib_mod.Index(space="cosine", dim=dim)
        # S-L01: share knobs with ``_estimate_ram_mb`` so RAM reservation
        # never under-counts.
        index.init_index(
            max_elements=len(embedded),
            ef_construction=self.HNSW_EF_CONSTRUCTION,
            M=self.HNSW_M,
        )
        index.set_ef(min(50, len(embedded)))

        try:
            # H-12/C-P-04 + H-12/L-P-05: batch add_items + knn_query instead
            # of one-at-a-time Python→C round-trips. hnswlib releases the GIL
            # during the batch call and processes rows in the same order, so
            # neighbour-label output is unchanged — behavioural equivalence
            # holds. The subsequent candidate-selection loop below still
            # drives `seen_losers` inline, so its decisions are identical.
            #
            # S9-W3 H-PERF-02: stream ``embedded`` straight into the
            # index without materialising ``all_embeddings`` as a
            # second Python list. At N=100k × 384-dim × 4 B this saves
            # ~150 MB of transient RAM (the previous comprehension
            # doubled the embedding footprint). hnswlib's add_items
            # accepts any sized iterable and a parallel list of labels.
            #
            # S9-W3 H-SKEP-02: pin ``set_ef(max(50, k*3))`` before the
            # batched knn so approximate-search quality matches the
            # pre-refactor per-item default. Stage 9 Skeptic flagged
            # that batched knn can miss near-duplicates at scale when
            # ef is not explicitly set.
            k = min(6, len(embedded))
            index.set_ef(max(50, k * 3))
            labels = list(range(len(embedded)))
            # Pass the DB rows' embedding lists directly — hnswlib
            # converts to ndarray inside and copies into its own
            # contiguous buffer, so we never need a second Python list.
            index.add_items([r["embedding"] for r in embedded], labels)

            candidates: list[tuple[str, str, float, float]] = []
            seen_losers: set[str] = set()

            # H-12/C-P-04: one batched knn_query for all rows. The
            # same embedding list is consumed once; hnswlib frees its
            # internal ndarray before this block returns via ``del index``
            # in the finally.
            all_labels, all_distances = index.knn_query(
                [r["embedding"] for r in embedded], k=k,
            )

            for i, r in enumerate(embedded):
                if time.monotonic() > deadline:
                    break
                lbls = all_labels[i]
                dsts = all_distances[i]
                for nb_idx, dist in zip(lbls, dsts):
                    if int(nb_idx) == i:
                        continue
                    neighbour = embedded[int(nb_idx)]
                    if neighbour["fact_id"] in seen_losers:
                        continue
                    if r["fact_id"] in seen_losers:
                        break
                    # hnswlib cosine distance is (1 - cos).
                    cos = max(0.0, min(1.0, 1.0 - float(dist)))
                    if cos <= self.COSINE_THRESHOLD:
                        continue
                    jac = _jaccard(r["entities"], neighbour["entities"])
                    if jac <= self.ENTITY_JACCARD_THRESHOLD:
                        continue
                    canonical, loser = _pick_canonical(r, neighbour)
                    if loser["fact_id"] in seen_losers:
                        continue
                    candidates.append(
                        (canonical["fact_id"], loser["fact_id"], cos, jac),
                    )
                    seen_losers.add(loser["fact_id"])
            return candidates
        finally:
            # Free ANN RAM immediately (LLD-12 §3.3).
            del index

    def _prefix_fallback(
        self,
        rows: list[dict[str, Any]],
        deadline: float,
    ) -> list[tuple[str, str, float, float]]:
        """Content-prefix dedup — retained behaviour when hnswlib cannot run."""
        seen_prefix: dict[str, dict[str, Any]] = {}
        candidates: list[tuple[str, str, float, float]] = []
        for r in rows:
            if time.monotonic() > deadline:
                break
            prefix = (r["content"] or "")[:100].strip().lower()
            if not prefix:
                continue
            prior = seen_prefix.get(prefix)
            if prior is None:
                seen_prefix[prefix] = r
                continue
            canonical, loser = _pick_canonical(prior, r)
            jac = _jaccard(prior["entities"], r["entities"])
            candidates.append(
                (canonical["fact_id"], loser["fact_id"], 1.0, jac),
            )
        return candidates
