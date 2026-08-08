"""Experiment 6 — Temporal-memory micro-evaluation (three behaviours).

Measured directly on SLM's own temporal machinery (no external agent benchmark):

  6a  Superseded-fact demotion: a bi-temporally invalidated fact stays
      recallable but is scaled by the 0.25 demotion factor and re-ranked below
      still-valid facts. Driven through the real invalidate + real validity
      filter over the real ``get_invalidated_fact_ids`` query.
  6b  Recency decay monotonicity: Ebbinghaus retention is non-increasing in age
      for a fixed strength. Driven through the real ``EbbinghausCurve``.
  6c  Time-window inference: date-proximate events rank above distant ones and
      events beyond the horizon are excluded. Driven through the real
      ``TemporalChannel`` over a real database.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path

from _harness import TempWorkspace, TrialOutcome, add_profile, fresh_db, run_trials


# ---------------------------------------------------------------------------
# 6a — superseded-fact demotion
# ---------------------------------------------------------------------------


def _store_fact(db, pid: str, fact_id: str) -> None:
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    mem = MemoryRecord(
        memory_id=f"m_{fact_id}", profile_id=pid, scope="personal",
        shared_with=None, content=fact_id,
    )
    db.store_memory(mem)
    db.store_fact(AtomicFact(
        fact_id=fact_id, memory_id=mem.memory_id, profile_id=pid,
        scope="personal", shared_with=None, content=f"fact {fact_id}",
    ))


def _trial_6a(index: int) -> TrialOutcome:
    from superlocalmemory.retrieval.temporal_validity_filter import (
        TemporalValidityFilter,
    )

    with TempWorkspace() as ws:
        db = fresh_db(ws)
        try:
            pid = f"p_{uuid.uuid4().hex[:8]}"
            add_profile(db, pid)
            valid = f"valid_{uuid.uuid4().hex[:6]}"
            superseded = f"old_{uuid.uuid4().hex[:6]}"
            _store_fact(db, pid, valid)
            _store_fact(db, pid, superseded)

            # Register + invalidate the superseded fact through the real path.
            db.execute(
                "INSERT OR IGNORE INTO fact_temporal_validity "
                "(fact_id, profile_id) VALUES (?, ?)",
                (superseded, pid),
            )
            db.invalidate_fact_temporal(superseded, "newer-fact", "superseded")

            filt = TemporalValidityFilter(db, demotion_factor=0.25)
            out = filt.filter(
                {"semantic": [(valid, 0.9), (superseded, 0.8)]}, pid, None,
            )
            ranked = out["semantic"]
            scores = dict(ranked)

            held = (
                superseded in scores
                and abs(scores[superseded] - 0.8 * 0.25) < 1e-9
                and abs(scores[valid] - 0.9) < 1e-9
                and ranked[0][0] == valid
                and ranked[1][0] == superseded
            )
            detail = {"index": index}
            if not held:
                detail["ranked"] = ranked
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# 6b — recency decay monotonicity
# ---------------------------------------------------------------------------

_AGE_SWEEP_HOURS = (0.0, 1.0, 6.0, 24.0, 72.0, 168.0, 720.0, 2160.0)


def _trial_6b(index: int) -> TrialOutcome:
    from superlocalmemory.core.config import ForgettingConfig
    from superlocalmemory.math.ebbinghaus import EbbinghausCurve

    rng = random.Random(1000 + index)
    curve = EbbinghausCurve(ForgettingConfig())
    strength = rng.uniform(0.5, 50.0)
    series = [curve.retention(h, strength) for h in _AGE_SWEEP_HOURS]

    monotonic = all(series[i] >= series[i + 1] - 1e-12 for i in range(len(series) - 1))
    bounded = all(0.0 <= r <= 1.0 for r in series)
    fresh_highest = series[0] >= max(series)
    held = monotonic and bounded and fresh_highest
    detail = {"index": index, "strength": round(strength, 4)}
    if not held:
        detail["series"] = [round(r, 6) for r in series]
    return TrialOutcome(index=index, held=held, detail=detail)


# ---------------------------------------------------------------------------
# 6c — time-window inference
# ---------------------------------------------------------------------------


def _seed_event(db, pid: str, fact_id: str, ref_date: str) -> None:
    from superlocalmemory.storage.models import (
        AtomicFact,
        CanonicalEntity,
        MemoryRecord,
        TemporalEvent,
    )

    mem = MemoryRecord(
        memory_id=f"m_{fact_id}", profile_id=pid, scope="personal",
        shared_with=None, content=fact_id,
    )
    db.store_memory(mem)
    db.store_fact(AtomicFact(
        fact_id=fact_id, memory_id=mem.memory_id, profile_id=pid,
        scope="personal", shared_with=None,
        content=f"Nova acted ({fact_id})", referenced_date=ref_date,
    ))
    entity = CanonicalEntity(
        entity_id=f"e_{fact_id}", profile_id=pid,
        canonical_name="Nova", entity_type="person",
    )
    db.execute(
        "INSERT OR IGNORE INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, first_seen, "
        "last_seen, fact_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entity.entity_id, entity.profile_id, entity.canonical_name,
         entity.entity_type, entity.first_seen, entity.last_seen,
         entity.fact_count),
    )
    db.store_temporal_event(TemporalEvent(
        event_id=f"t_{fact_id}", profile_id=pid, entity_id=entity.entity_id,
        fact_id=fact_id, referenced_date=ref_date, description=fact_id,
        scope="personal", shared_with=None,
    ))


def _trial_6c(index: int) -> TrialOutcome:
    from superlocalmemory.retrieval.temporal_channel import TemporalChannel

    with TempWorkspace() as ws:
        db = fresh_db(ws)
        try:
            pid = f"p_{uuid.uuid4().hex[:8]}"
            add_profile(db, pid)
            near = f"near_{uuid.uuid4().hex[:6]}"
            far = f"far_{uuid.uuid4().hex[:6]}"
            # Query anchor 2026-03-11; near is 1 day off, far is >2 years off.
            _seed_event(db, pid, near, "2026-03-12")
            _seed_event(db, pid, far, "2023-01-01")

            channel = TemporalChannel(db)
            results = channel.search("What happened on 2026-03-11?", pid)
            scores = dict(results)

            near_score = scores.get(near, 0.0)
            far_score = scores.get(far, 0.0)
            held = (
                near in scores
                and near_score > 0.5          # date-proximate -> high
                and near_score > far_score     # near outranks far
                and far_score == 0.0           # beyond horizon -> excluded
            )
            detail = {"index": index,
                      "near_score": round(near_score, 4),
                      "far_score": round(far_score, 4)}
            if not held:
                detail["results"] = [(f, round(s, 4)) for f, s in results]
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(n_trials: int = 200, seed: int = 0) -> list:
    return [
        run_trials(
            name="exp6a_superseded_demotion",
            guarantee="superseded facts kept but demoted 0.25x and re-ranked below valid",
            metric_name="correct-demotion rate",
            n_trials=n_trials,
            trial_fn=_trial_6a,
            method=(
                "Real invalidate_fact_temporal + TemporalValidityFilter(0.25) "
                "over the real get_invalidated_fact_ids query."
            ),
        ),
        run_trials(
            name="exp6b_decay_monotonic",
            guarantee="Ebbinghaus retention non-increasing in age, bounded [0,1]",
            metric_name="monotonic-decay rate",
            n_trials=n_trials,
            trial_fn=_trial_6b,
            method="Real EbbinghausCurve.retention swept over an increasing age vector.",
        ),
        run_trials(
            name="exp6c_time_window",
            guarantee="date-proximate events outrank distant ones; horizon excludes far past",
            metric_name="correct-window rate",
            n_trials=n_trials,
            trial_fn=_trial_6c,
            method=(
                "Real TemporalChannel.search over a real DB with a near "
                "(+1 day) and a far (>2 year) dated event."
            ),
        ),
    ]


if __name__ == "__main__":
    from _harness import write_result

    for result in run():
        print(write_result(result, Path(__file__).parent / "results"))
        print(f"{result.name}: {result.held}/{result.trials} "
              f"({result.metric_value:.4f})")
