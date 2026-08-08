"""Experiment 7 — Generation fence: stale epoch rejection and positive commit.

Spine exercised:
  generation_fence module (storage/generation_fence.py)
    record_admission_epoch / admitted_epoch / clear_admission_epoch
  CanonicalRememberRuntime._handle_admission (remember_runtime.py line 465)
    epoch staleness check: admitted_epoch != self._generation → ValueError
  WriteCoordinator (storage/write_coordinator.py)
    wraps ValueError in WriteCoordinatorError.__cause__

Each trial uses a fresh CanonicalRememberRuntime (separate db + journal)
and exercises two controls in sequence:

  STALE control
    1. Record admission epoch 0 for a unique idempotency_key.
    2. Advance runtime._generation to 1 by direct assignment
       (runtime._generation = 1).  This models a post-rebind epoch without
       exercising the full rebind_engine path; the fence behaviour under an
       epoch mismatch is what this experiment covers, not the rebind mechanism.
    3. Submit a WriteCommand(ADMISSION) referencing that key via the
       coordinator directly — bypassing runtime.remember().
    4. Assert: WriteCoordinatorError raised, __cause__ is ValueError
       with "epoch is stale" in the message, writer never called
       (calls==[]), runtime_probe table has 0 rows.

  POSITIVE control (same runtime, generation already at 1)
    5. Call runtime.remember() with a fresh idempotency_key.
       runtime.remember() reads _generation=1 and records that as the
       admitted epoch — so when _handle_admission checks, 1 == 1 → clean.
    6. Assert: payload["status"]=="queryable", calls has exactly 1 entry,
       runtime_probe has 1 row.

A trial holds only when both controls pass.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from _harness import TrialOutcome, run_trials

_DEADLINE_MS = 1_500


# ---------------------------------------------------------------------------
# Fence reset helper (module-level global cleared between trials)
# ---------------------------------------------------------------------------


def _reset_fence() -> None:
    from superlocalmemory.storage import generation_fence as gf
    with gf._lock:
        gf._epochs.clear()


# ---------------------------------------------------------------------------
# DB bootstrap: only the two migrations the coordinator needs
# (mirrors test_generation_fence.py exactly — NOT fresh_db())
# ---------------------------------------------------------------------------


def _install_write_commits(db_path: Path) -> None:
    from superlocalmemory.storage.migrations import (
        M018_ingestion_operations,
        M032_write_coordinator_admission,
    )

    conn = sqlite3.connect(str(db_path))
    try:
        M018_ingestion_operations.apply(conn)
        M032_write_coordinator_admission.apply(conn)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def _trial(index: int) -> TrialOutcome:
    import shutil
    import tempfile

    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage import generation_fence as gf
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.write_coordinator import (
        CommandKind,
        WriteCommand,
        WriteCoordinatorError,
    )

    _reset_fence()
    ws = Path(tempfile.mkdtemp(prefix="slm-exp7-"))
    try:
        db_path = ws / "memory.db"
        _install_write_commits(db_path)
        db = DatabaseManager(db_path)
        db.initialize(schema)

        calls: list[str] = []

        def writer(request, operation_id: str) -> list[str]:
            calls.append(operation_id)
            db.execute(
                "INSERT INTO runtime_probe(operation_id, content) VALUES (?, ?)",
                (operation_id, request.content),
            )
            return ["fact-1"]

        runtime = CanonicalRememberRuntime(
            db=db,
            profile_id="default",
            writer=writer,
            journal_path=ws / "admission_journal.db",
        )
        db.execute("CREATE TABLE runtime_probe(operation_id TEXT, content TEXT)")

        runtime.start()
        try:
            # ----------------------------------------------------------------
            # STALE control
            # ----------------------------------------------------------------
            stale_key = f"stale-{uuid.uuid4().hex}"
            stale_req = RememberRequest(
                content="stale epoch witness",
                profile_id="default",
                source_type="http",
                idempotency_key=stale_key,
                trusted_actor_id="actor",
            )
            gf.record_admission_epoch("default", stale_key, 0)
            with runtime._binding_lock:
                runtime._generation = 1

            stale_cmd = WriteCommand.create(
                CommandKind.ADMISSION,
                {
                    "journal_id": f"jid-{stale_key}",
                    "request_hash": "hash-stale",
                    "profile_id": "default",
                    "idempotency_key": stale_key,
                    "request": stale_req.canonical_payload(),
                },
                command_id=f"jid-{stale_key}",
            )

            caught_err: WriteCoordinatorError | None = None
            try:
                runtime.coordinator.submit(stale_cmd, timeout=2.0)
            except WriteCoordinatorError as err:
                caught_err = err

            probe_after_stale = int(
                dict(db.execute("SELECT COUNT(*) AS c FROM runtime_probe")[0])["c"]
            )

            stale_ok = (
                caught_err is not None
                and isinstance(caught_err.__cause__, ValueError)
                and "epoch is stale" in str(caught_err.__cause__)
                and calls == []
                and probe_after_stale == 0
            )
            stale_detail: dict = {}
            if not stale_ok:
                stale_detail = {
                    "caught": caught_err is not None,
                    "cause_class": (
                        type(caught_err.__cause__).__name__
                        if caught_err and caught_err.__cause__
                        else None
                    ),
                    "cause_msg": (
                        str(caught_err.__cause__)
                        if caught_err and caught_err.__cause__
                        else None
                    ),
                    "calls_after_stale": list(calls),
                    "probe_after_stale": probe_after_stale,
                }

            # ----------------------------------------------------------------
            # POSITIVE control — same runtime, _generation still 1
            # ----------------------------------------------------------------
            actor = Actor("actor", frozenset({"default"}), frozenset({"personal"}))
            fresh_key = f"fresh-{uuid.uuid4().hex}"
            fresh_req = RememberRequest(
                content="post-transition write",
                profile_id="default",
                source_type="http",
                idempotency_key=fresh_key,
                trusted_actor_id="actor",
            )

            positive_ok = False
            positive_detail: dict = {}
            try:
                result = runtime.remember(fresh_req, actor, deadline_ms=_DEADLINE_MS)
                payload = result.payload
                probe_after_positive = int(
                    dict(db.execute("SELECT COUNT(*) AS c FROM runtime_probe")[0])["c"]
                )
                positive_ok = (
                    payload.get("status") == "queryable"
                    and len(calls) == 1
                    and probe_after_positive == 1
                )
                if not positive_ok:
                    positive_detail = {
                        "status": payload.get("status"),
                        "calls_count": len(calls),
                        "probe_after_positive": probe_after_positive,
                    }
            except Exception as exc:  # noqa: BLE001
                positive_detail = {"error": f"{type(exc).__name__}: {exc}"}

        finally:
            runtime.stop()

        held = stale_ok and positive_ok
        detail: dict = {"index": index}
        if not held:
            detail.update(stale=stale_detail, positive=positive_detail)
        return TrialOutcome(index=index, held=held, detail=detail)

    finally:
        shutil.rmtree(ws, ignore_errors=True)
        _reset_fence()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(n_trials: int = 200, seed: int = 0) -> object:
    return run_trials(
        name="exp7_generation_fence",
        guarantee=(
            "Generation fence rejects a stale-epoch ADMISSION (WriteCoordinatorError "
            "with ValueError __cause__ 'epoch is stale', writer never called) and "
            "admits a fresh-epoch ADMISSION from runtime.remember() in the same runtime"
        ),
        metric_name="fence-correct rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real CanonicalRememberRuntime with M018+M032 schema (mirrors "
            "test_generation_fence.py). Stale control: direct coordinator.submit() "
            "at epoch 0 after advancing _generation to 1 → WriteCoordinatorError. "
            "Positive control: runtime.remember() captures _generation=1, admitted "
            "epoch matches → commits, writer called once, probe has 1 row."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} ({result.metric_value:.4f})")
