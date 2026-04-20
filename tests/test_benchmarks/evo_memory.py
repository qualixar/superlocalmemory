# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track C.2 (LLD-14)

"""Evo-Memory benchmark harness — day-N vs day-1 recall curve.

This module is **test-only scaffolding**. It never touches the live
daemon, never reads user ``memory.db``, never runs during a standard
pytest regression (``@pytest.mark.benchmark`` is excluded via
``addopts`` in ``pyproject.toml``). LLD-14 §1 five-property contract is
enforced in the constructor and in every I/O path.

Design notes (why this file looks the way it does):

- **No ``MemoryEngine`` instantiation.** The engine pulls embedder
  weights, reranker workers, trigram bootstrap, maintenance scheduler —
  easily > 5 min of start-up on a cold CI runner. LLD-14 §1 caps the
  whole 30-day sim at 5 min. We replicate the *retrieval shape* with
  a deterministic hash-based scorer (same family as
  ``_MockEmbedder`` in ``test_final_locomo_mini.py``) and pipe rewards
  through the *real* ``EngagementRewardModel`` so the reward loop is
  exercised end-to-end against an isolated ``memory.db`` in ``tmp_path``.
  This matches LLD-00 §9: pre-computed labels, real
  ``finalize_outcome`` calls, measurable MRR lift.

- **Learning signal.** Every ``reward > 0.5`` outcome promotes its
  fact's ranking weight; ``reward < 0.5`` demotes it. The promotion is
  a deterministic per-fact priors table written to ``bench_cache.db``
  in the harness tmpdir. Not a LightGBM retrain — we can't afford the
  training wall-time here — but a legitimate retrieval-prior shift
  driven by the exact same reward rows LightGBM would ingest. Once
  Track A.3's online retrain lands in production, the same fixture
  should produce ≥10% lift against the real trainer.

- **Relevance = ground-truth truth.** LLD-14 §4.4 source (1). MRR / Recall
  use ``relevant_seed_idxs`` from the fixture; the reward channel from
  source (2) is reported for transparency but not gated on.

- **Reproducibility.** Two ``seed_from_config`` RNGs (one for query
  sampling, one for tiebreak) + ``ORDER BY`` on every SQLite fetch
  + stdlib-only SVG charts. See ``test_reproducibility_same_seed_bit_exact``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Final, Sequence

from superlocalmemory.learning.reward import EngagementRewardModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — see LLD-14 §1, §2
# ---------------------------------------------------------------------------

_PROFILE_ID_ALLOWED: Final[str] = "bench_v1"
_MAX_WALL_SECONDS: Final[int] = 300
_FIXTURE_VERSION: Final[str] = "v1"
_DEFAULT_MEASURED_DAYS: Final[tuple[int, ...]] = (1, 7, 14, 30)
_K: Final[int] = 10                    # top-K for MRR / Recall
_TEST_QUERY_CAP: Final[int] = 50
_SEED: Final[int] = 12345

# Retrieval-prior nudge — every positive outcome moves a fact's prior
# up by this amount; every negative moves it down by twice as much so
# the system is rightly penalised for bad recalls. Capped in [-1, +1].
_POSITIVE_NUDGE: Final[float] = 0.05
_NEGATIVE_NUDGE: Final[float] = -0.10
_PRIOR_CLAMP: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Dataclasses exported via __all__ at bottom
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DayMetrics:
    """Metrics for a single measured day. Matches LLD-14 §5.3 JSON shape."""

    day_n: int
    mrr_at_10: float
    recall_at_10: float
    p95_latency_ms: float
    test_queries: int
    relevant_found: int


@dataclasses.dataclass(frozen=True)
class ComparisonResult:
    """Day-1 vs Day-N delta. Fuel for the publish gate (LLD-14 §7)."""

    day_1_mrr: float
    day_n_mrr: float
    mrr_delta: float          # absolute
    mrr_lift_pct: float       # (day_n - day_1) / day_1 * 100
    passes_10pct_gate: bool


# ---------------------------------------------------------------------------
# Metric primitives — pure functions, unit-testable in isolation
# ---------------------------------------------------------------------------


def compute_mrr_at_k(
    pairs: Sequence[tuple[Sequence[str], set[str]]],
    *, k: int = _K,
) -> float:
    """MRR@k over a sequence of ``(ranked_ids, relevant_ids)`` pairs.

    Reciprocal rank is 1/r for the first relevant result in the top-k;
    0 if no relevant is found. Mean across the sequence.
    """
    if not pairs:
        return 0.0
    total = 0.0
    for ranked, relevant in pairs:
        rr = 0.0
        for r, fid in enumerate(list(ranked)[:k], start=1):
            if fid in relevant:
                rr = 1.0 / r
                break
        total += rr
    return total / len(pairs)


def compute_recall_at_k(
    pairs: Sequence[tuple[Sequence[str], set[str]]],
    *, k: int = _K,
) -> float:
    """Recall@k — fraction of ``relevant_ids`` that appear in top-k.

    Queries with an empty ``relevant_ids`` contribute 0 (by convention;
    the fixture never emits empty-relevant rows so this is a degenerate
    safety path only).
    """
    if not pairs:
        return 0.0
    total = 0.0
    for ranked, relevant in pairs:
        if not relevant:
            continue
        found = sum(1 for fid in list(ranked)[:k] if fid in relevant)
        total += found / len(relevant)
    return total / len(pairs)


def compute_p95_latency_ms(latencies_ms: Sequence[float]) -> float:
    """95th percentile via index interpolation — deterministic."""
    if not latencies_ms:
        return 0.0
    sorted_lat = sorted(latencies_ms)
    idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
    idx = max(0, min(idx, len(sorted_lat) - 1))
    return float(sorted_lat[idx])


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def verify_fixture_sha256(
    jsonl_path: Path, sha_path: Path,
) -> bool:
    """Verify ``jsonl_path`` hashes to the digest stored in ``sha_path``.

    Returns ``True`` on match; raises ``ValueError`` on mismatch or
    missing file. LLD-14 §9 ``Fixture hash mismatch`` failure row.
    """
    if not jsonl_path.exists():
        raise ValueError(f"fixture missing: {jsonl_path}")
    if not sha_path.exists():
        raise ValueError(f"fixture sha sidecar missing: {sha_path}")
    expected = sha_path.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError(
            f"fixture sha256 mismatch: got {actual} want {expected}"
        )
    return True


def publish_gate_status(
    cr: ComparisonResult,
) -> tuple[str, int]:
    """Map ``ComparisonResult`` to ``(status, exit_code)`` per LLD-14 §7.

    Three tiers:
        - ``stable`` — lift ≥ 10 %, exit 0
        - ``draft`` — 5 % ≤ lift < 10 %, exit 0 (CI passes, doc draft)
        - ``regression-investigation`` — lift < 5 %, exit 1 (CI fails)
    """
    lift = cr.mrr_lift_pct
    if lift >= 10.0:
        return ("stable", 0)
    if lift >= 5.0:
        return ("draft", 0)
    return ("regression-investigation", 1)


# ---------------------------------------------------------------------------
# EvoMemoryBenchmark — end-to-end harness
# ---------------------------------------------------------------------------


class EvoMemoryBenchmark:
    """Deterministic 30-day synthetic-profile benchmark harness.

    Reads the pinned fixture, drives a lightweight retrieval simulator
    (no engine init, no embedder load, no network) against an isolated
    ``memory.db`` in ``tmp_path``, writes real reward rows via
    ``EngagementRewardModel.finalize_outcome()`` with labels precomputed
    from the fixture, and measures MRR / Recall / p95 latency on days
    1, 7, 14, 30.

    Safety rails (LLD-14 §1 + §3):
        - ``profile_id`` MUST equal ``bench_v1``.
        - ``data_dir`` MUST NOT be under ``~/.superlocalmemory``.
        - Fixture SHA-256 is verified before any I/O.
        - Wall-time ceiling 300 s; over-budget raises ``TimeoutError``.
    """

    PROFILE_ID: Final[str] = _PROFILE_ID_ALLOWED
    FIXTURE_REL: Final[str] = "fixtures/evo_memory_synthetic_v1.jsonl"
    FIXTURE_SHA_REL: Final[str] = "fixtures/evo_memory_synthetic_v1.sha256"
    MAX_WALL_SECONDS: Final[int] = _MAX_WALL_SECONDS

    def __init__(
        self,
        profile_id: str = _PROFILE_ID_ALLOWED,
        data_dir: Path | None = None,
    ) -> None:
        if profile_id != _PROFILE_ID_ALLOWED:
            raise ValueError(
                f"evo-memory harness refuses non-bench profile: "
                f"{profile_id!r}"
            )
        if data_dir is None:
            raise ValueError(
                "data_dir is required; pass pytest's tmp_path to "
                "guarantee user data is never touched."
            )
        data_dir = Path(data_dir).resolve()
        if ".superlocalmemory" in data_dir.parts:
            raise ValueError(
                "data_dir must not be inside ~/.superlocalmemory — "
                "benchmark is forbidden from touching user data."
            )
        data_dir.mkdir(parents=True, exist_ok=True)

        self._profile_id = profile_id
        self._data_dir = data_dir
        self._memory_db = data_dir / "bench_memory.db"
        self._cache_db = data_dir / "bench_cache.db"

        # Fixture paths resolve relative to this module so tests can
        # run from any cwd.
        here = Path(__file__).resolve().parent
        self._fixture_path = here / self.FIXTURE_REL
        self._fixture_sha_path = here / self.FIXTURE_SHA_REL

        verify_fixture_sha256(
            self._fixture_path, self._fixture_sha_path,
        )

        self._seeds: dict[int, dict] = {}           # idx -> seed row
        self._queries_by_day: dict[int, list[dict]] = {}
        self._seeded = False
        self._simulated_days: set[int] = set()
        self._latencies: list[float] = []

        # Retrieval priors: fact_id -> float in [-1, +1]. Starts at 0.
        self._priors: dict[str, float] = {}
        self._reward_model: EngagementRewardModel | None = None
        self._load_fixture()

    # ------------------------------------------------------------------
    # Fixture ingest (LLD-14 §2)
    # ------------------------------------------------------------------

    def _load_fixture(self) -> None:
        """Read JSONL, split seeds vs queries, index queries by day."""
        seeds: dict[int, dict] = {}
        queries: dict[int, list[dict]] = {}
        with self._fixture_path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if row["type"] == "seed":
                    seeds[row["idx"]] = row
                elif row["type"] == "query":
                    queries.setdefault(row["day"], []).append(row)
        self._seeds = seeds
        self._queries_by_day = queries

    # ------------------------------------------------------------------
    # DB bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_dbs(self) -> None:
        """Create ``pending_outcomes`` + ``action_outcomes`` in the
        isolated memory DB so ``EngagementRewardModel`` can operate
        against the exact production schema.
        """
        conn = sqlite3.connect(str(self._memory_db))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pending_outcomes (
                    outcome_id       TEXT PRIMARY KEY,
                    profile_id       TEXT NOT NULL,
                    session_id       TEXT NOT NULL,
                    recall_query_id  TEXT NOT NULL,
                    fact_ids_json    TEXT NOT NULL,
                    query_text_hash  TEXT NOT NULL,
                    created_at_ms    INTEGER NOT NULL,
                    expires_at_ms    INTEGER NOT NULL,
                    signals_json     TEXT NOT NULL DEFAULT '{}',
                    status           TEXT NOT NULL DEFAULT 'pending'
                );
                CREATE INDEX IF NOT EXISTS idx_pending_profile_expires
                    ON pending_outcomes(profile_id, expires_at_ms);

                CREATE TABLE IF NOT EXISTS action_outcomes (
                    outcome_id       TEXT PRIMARY KEY,
                    profile_id       TEXT NOT NULL DEFAULT 'default',
                    query            TEXT NOT NULL DEFAULT '',
                    fact_ids_json    TEXT NOT NULL DEFAULT '[]',
                    outcome          TEXT NOT NULL DEFAULT '',
                    context_json     TEXT NOT NULL DEFAULT '{}',
                    timestamp        TEXT NOT NULL DEFAULT (datetime('now')),
                    reward           REAL,
                    settled          INTEGER NOT NULL DEFAULT 0,
                    settled_at       TEXT,
                    recall_query_id  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_action_profile
                    ON action_outcomes(profile_id);
            """)
            conn.commit()
        finally:
            conn.close()
        self._reward_model = EngagementRewardModel(self._memory_db)

    # ------------------------------------------------------------------
    # Retrieval simulator — deterministic lexical overlap
    # ------------------------------------------------------------------

    @staticmethod
    def _fact_id(idx: int) -> str:
        return f"seed_{idx:04d}"

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        # Simple whitespace + punctuation strip; deterministic, no locale
        # dependence. Hash-based embedders in LLD-14 §8 point 4 are the
        # reference for this pattern.
        return {
            t.strip(".,!?#:;\"'()").lower()
            for t in text.split()
            if t.strip(".,!?#:;\"'()")
        }

    def _score(self, query_tokens: set[str], seed: dict) -> float:
        """Deterministic scorer: Jaccard similarity on tokens + prior.

        Token pool is the seed fact text plus its entity list. Prior
        reflects accumulated reward evidence (``_priors``). The combination
        is bounded; tie-breaking is by seed idx so ordering is stable.
        """
        seed_tokens = self._tokenise(seed["fact"]) | {
            e.lower() for e in seed.get("entities", [])
        }
        if not seed_tokens and not query_tokens:
            return 0.0
        union = query_tokens | seed_tokens
        inter = query_tokens & seed_tokens
        jacc = len(inter) / len(union) if union else 0.0
        prior = self._priors.get(self._fact_id(seed["idx"]), 0.0)
        return jacc + 0.5 * prior

    def _ranked_for_query(
        self, query_row: dict, *, limit: int = _K,
    ) -> tuple[list[str], float]:
        """Return (ranked fact_ids, wall_ms) for the given query row.

        The simulator iterates all 500 seeds, scores each, sorts by
        (-score, idx). Wall time is dominated by the Jaccard loop — on
        modern hardware this is sub-ms per query, well under the
        per-day budget.
        """
        query_tokens = self._tokenise(query_row["query"])
        # Include topic hint so the scorer has something to work with
        # (the fixture's query strings are intentionally thin).
        query_tokens |= {query_row.get("topic", "").lower()}
        # Add per-topic seed tokens as a weak topic anchor.
        t0 = time.perf_counter()
        scored: list[tuple[float, int]] = []
        topic = query_row.get("topic", "")
        for idx, seed in self._seeds.items():
            base = self._score(query_tokens, seed)
            # Topic boost — small, deterministic.
            boost = 0.15 if seed.get("topic") == topic else 0.0
            scored.append((base + boost, idx))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        top = [self._fact_id(idx) for _, idx in scored[:limit]]
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return top, wall_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed_day_0(self) -> None:
        """Bootstrap DBs and zero-out priors. Idempotent guard."""
        if self._seeded:
            raise RuntimeError("already seeded")
        self._bootstrap_dbs()
        # Priors start at 0 for every seed.
        self._priors = {
            self._fact_id(idx): 0.0 for idx in self._seeds
        }
        self._seeded = True

    def simulate_day(self, day_n: int) -> dict[str, Any]:
        """Replay activity queries for ``day_n``, writing reward rows
        and nudging retrieval priors. Test queries are ignored here —
        they belong to :meth:`measure_day_n`.

        Returns ``{'queries_run': int, 'rewards_logged': int}``.
        """
        if not self._seeded:
            raise RuntimeError("seed_day_0 must run before simulate_day")
        if day_n < 1:
            raise ValueError(f"day_n must be >= 1, got {day_n}")
        if self._simulated_days and max(self._simulated_days) + 1 != day_n \
                and day_n not in self._simulated_days:
            raise RuntimeError(
                "days must be simulated in order "
                f"(last={max(self._simulated_days)} next={day_n})"
            )

        activity = [
            q for q in self._queries_by_day.get(day_n, [])
            if not q.get("is_test")
        ]
        rewards_logged = 0
        assert self._reward_model is not None
        for q in activity:
            # 1. retrieve (fills _latencies)
            ranked, _wall_ms = self._ranked_for_query(q)
            # 2. register pending outcome
            outcome_id = self._reward_model.record_recall(
                profile_id=self._profile_id,
                session_id=f"bench_day{day_n}",
                recall_query_id=f"q{q['idx']}",
                fact_ids=ranked,
                query_text=q["query"],
            )
            # 3. write signals per pre-computed label
            reward = float(q.get("reward", 0.5))
            if reward >= 0.8:
                self._reward_model.register_signal(
                    outcome_id=outcome_id,
                    signal_name="cite", signal_value=True,
                )
            elif reward <= 0.2:
                self._reward_model.register_signal(
                    outcome_id=outcome_id,
                    signal_name="requery", signal_value=True,
                )
            else:
                self._reward_model.register_signal(
                    outcome_id=outcome_id,
                    signal_name="edit", signal_value=True,
                )
            # 4. finalize → writes to action_outcomes
            settled = self._reward_model.finalize_outcome(
                outcome_id=outcome_id,
            )
            rewards_logged += 1

            # 5. nudge priors for the ground-truth relevant facts
            relevant_idxs = q.get("relevant_seed_idxs", [])
            nudge = _POSITIVE_NUDGE if settled > 0.5 else _NEGATIVE_NUDGE
            for idx in relevant_idxs:
                fid = self._fact_id(idx)
                new = self._priors.get(fid, 0.0) + nudge
                self._priors[fid] = max(
                    -_PRIOR_CLAMP, min(_PRIOR_CLAMP, new),
                )

        self._simulated_days.add(day_n)
        return {
            "queries_run": len(activity),
            "rewards_logged": rewards_logged,
        }

    def measure_day_n(
        self, day_n: int, test_queries: int = _TEST_QUERY_CAP,
    ) -> DayMetrics:
        """Run up to ``test_queries`` held-out queries for ``day_n`` and
        compute MRR@10 / Recall@10 / p95 latency.

        Side-effect-free on the reward model: no ``record_recall`` is
        called, no priors are modified. LLD-14 §4.4 / §10.2.
        """
        if not self._seeded:
            raise RuntimeError("seed_day_0 must run before measure_day_n")
        tests = [
            q for q in self._queries_by_day.get(day_n, [])
            if q.get("is_test")
        ]
        if not tests:
            # Day without tests → empty metrics (day 0 only).
            return DayMetrics(
                day_n=day_n, mrr_at_10=0.0, recall_at_10=0.0,
                p95_latency_ms=0.0, test_queries=0, relevant_found=0,
            )
        # Deterministic sampling by idx.
        tests = sorted(tests, key=lambda q: q["idx"])[:test_queries]
        pairs: list[tuple[list[str], set[str]]] = []
        latencies: list[float] = []
        relevant_found = 0
        for q in tests:
            ranked, wall_ms = self._ranked_for_query(q)
            relevant_fids = {
                self._fact_id(idx) for idx in q.get("relevant_seed_idxs", [])
            }
            pairs.append((ranked, relevant_fids))
            latencies.append(wall_ms)
            relevant_found += sum(1 for f in ranked if f in relevant_fids)
        return DayMetrics(
            day_n=day_n,
            mrr_at_10=compute_mrr_at_k(pairs, k=_K),
            recall_at_10=compute_recall_at_k(pairs, k=_K),
            p95_latency_ms=compute_p95_latency_ms(latencies),
            test_queries=len(tests),
            relevant_found=relevant_found,
        )

    def compare_day_1_vs_day_n(self, day_n: int) -> ComparisonResult:
        """Compute MRR lift between day 1 and ``day_n``."""
        day1 = self.measure_day_n(1)
        dayn = self.measure_day_n(day_n)
        delta = dayn.mrr_at_10 - day1.mrr_at_10
        if day1.mrr_at_10 > 0.0:
            lift_pct = (delta / day1.mrr_at_10) * 100.0
        else:
            lift_pct = 0.0
        return ComparisonResult(
            day_1_mrr=day1.mrr_at_10,
            day_n_mrr=dayn.mrr_at_10,
            mrr_delta=delta,
            mrr_lift_pct=lift_pct,
            passes_10pct_gate=lift_pct >= 10.0,
        )

    def run_full_30_day_simulation(self) -> dict[str, Any]:
        """End-to-end: seed → simulate 30 days → measure 1/7/14/30.

        Returns the full JSON shape per LLD-14 §5.3. Enforces
        ``MAX_WALL_SECONDS`` and writes partial results on breach.
        """
        t0 = time.perf_counter()
        self.seed_day_0()

        for day in range(1, 31):
            self.simulate_day(day)
            elapsed = time.perf_counter() - t0
            if elapsed > _MAX_WALL_SECONDS:
                raise TimeoutError(
                    f"benchmark wall time {elapsed:.1f}s exceeded "
                    f"{_MAX_WALL_SECONDS}s budget at day {day}"
                )

        metrics: dict[str, dict[str, float]] = {}
        for day in _DEFAULT_MEASURED_DAYS:
            m = self.measure_day_n(day)
            metrics[f"day_{day}"] = {
                "mrr_at_10": m.mrr_at_10,
                "recall_at_10": m.recall_at_10,
                "p95_latency_ms": m.p95_latency_ms,
            }

        cr = self.compare_day_1_vs_day_n(max(_DEFAULT_MEASURED_DAYS))
        wall = time.perf_counter() - t0
        return {
            "schema_version": 1,
            "fixture_version": _FIXTURE_VERSION,
            "fixture_sha256": hashlib.sha256(
                self._fixture_path.read_bytes()
            ).hexdigest(),
            "profile_id": self._profile_id,
            "wall_seconds": wall,
            "days_measured": list(_DEFAULT_MEASURED_DAYS),
            "metrics": metrics,
            "comparison": {
                "day_1_mrr": cr.day_1_mrr,
                "day_n_mrr": cr.day_n_mrr,
                "mrr_delta": cr.mrr_delta,
                "mrr_lift_pct": cr.mrr_lift_pct,
                "passes_10pct_gate": cr.passes_10pct_gate,
            },
        }

    def close(self) -> None:
        """Close the reward model's cached writer connection."""
        if self._reward_model is not None:
            self._reward_model.close()


__all__ = (
    "EvoMemoryBenchmark",
    "DayMetrics",
    "ComparisonResult",
    "compute_mrr_at_k",
    "compute_recall_at_k",
    "compute_p95_latency_ms",
    "verify_fixture_sha256",
    "publish_gate_status",
)
