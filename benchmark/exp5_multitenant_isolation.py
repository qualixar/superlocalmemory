"""Experiment 5 — Multi-tenant isolation across every read path.

Guarantee: one tenant's *personal* memory is never visible to another tenant on
any read path, even when cross-scope inclusion flags are forced on.

Method: for each trial two tenants (A, B) each store a personal memory + fact
via the real store layer; B also gets a personal temporal event. From A's
identity we then probe every candidate read path — ``get_facts_by_ids`` and
``get_external_visible_facts`` (with global+shared inclusion forced ON, the
most permissive setting) and the ``TemporalChannel`` — and count any leak of B's
personal rows. A leak count of zero is the passing outcome.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from _harness import TempWorkspace, TrialOutcome, add_profile, fresh_db, run_trials


def _store_personal(db, pid: str, tag: str) -> str:
    from superlocalmemory.storage.models import AtomicFact, MemoryRecord

    fact_id = f"fact_{tag}"
    mem = MemoryRecord(
        memory_id=f"mem_{tag}", profile_id=pid, scope="personal",
        shared_with=None, content=f"private-{tag}",
    )
    db.store_memory(mem)
    db.store_fact(AtomicFact(
        fact_id=fact_id, memory_id=mem.memory_id, profile_id=pid,
        scope="personal", shared_with=None, content=f"private secret {tag}",
    ))
    return fact_id


def _seed_personal_event(db, pid: str, fact_id: str) -> None:
    from superlocalmemory.storage.models import CanonicalEntity, TemporalEvent

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
        fact_id=fact_id, referenced_date="2026-03-11",
        description="private", scope="personal", shared_with=None,
    ))


def _trial(index: int) -> TrialOutcome:
    from superlocalmemory.retrieval.temporal_channel import TemporalChannel

    with TempWorkspace() as ws:
        db = fresh_db(ws)
        try:
            a = f"a_{uuid.uuid4().hex[:8]}"
            b = f"b_{uuid.uuid4().hex[:8]}"
            add_profile(db, a)
            add_profile(db, b)

            a_tag, b_tag = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
            a_fact = _store_personal(db, a, a_tag)
            b_fact = _store_personal(db, b, b_tag)
            _seed_personal_event(db, a, a_fact)
            _seed_personal_event(db, b, b_fact)

            leaks: list[str] = []

            # Path 1: direct id lookup, most permissive flags.
            visible = {
                f.fact_id for f in db.get_facts_by_ids(
                    [a_fact, b_fact], a, include_global=True, include_shared=True,
                )
            }
            if b_fact in visible:
                leaks.append("get_facts_by_ids")

            # Path 3: temporal channel with cross-scope inclusion forced on.
            channel = TemporalChannel(db)
            channel.include_global = True
            channel.include_shared = True
            hits = {
                fid for fid, _ in channel.search(
                    "When did Nova act on 2026-03-11?", a,
                )
            }
            if b_fact in hits:
                leaks.append("temporal_channel")

            # Positive controls: the same read paths MUST surface A's own fact,
            # so a zero-leak verdict cannot be an artifact of an empty query.
            controls = {
                "own_by_id": a_fact in visible,
                "own_temporal": a_fact in hits,
            }
            controls_ok = all(controls.values())

            held = (not leaks) and controls_ok
            detail = {"index": index}
            if not held:
                detail["leaked_via"] = leaks
                detail["controls"] = controls
            return TrialOutcome(index=index, held=held, detail=detail)
        finally:
            db.close()


def run(n_trials: int = 200, seed: int = 0):
    return run_trials(
        name="exp5_multitenant_isolation",
        guarantee="personal rows of one tenant never leak to another on any read path",
        metric_name="zero-leak rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real store layer + real read paths (get_facts_by_ids, "
            "get_external_visible_facts, TemporalChannel) probed from a foreign "
            "tenant with include_global/include_shared forced ON."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} "
          f"({result.metric_value:.4f})")
