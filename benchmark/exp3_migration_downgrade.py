"""Experiment 3 — Schema-downgrade refusal (no silent corruption).

Guarantee: a database stamped by a newer build than the running binary is
refused on the deferred migration pass, and the refusal mutates nothing.

Method: for each trial we create fresh learning + memory databases, stamp the
memory DB's schema_version to a value ahead of the binary's
``SUPPORTED_SCHEMA_VERSION``, snapshot the table set, then call the real
``migration_runner.apply_deferred``. We assert it raises ``SchemaVersionError``
and that the table set is byte-identical afterwards (zero mutation).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from _harness import TempWorkspace, TrialOutcome, run_trials


def _tables(db_path: Path) -> frozenset[str]:
    with sqlite3.connect(db_path) as conn:
        return frozenset(
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        )


def _trial(index: int) -> TrialOutcome:
    import superlocalmemory.storage.migration_runner as mr
    from superlocalmemory.storage._schema_version import (
        SUPPORTED_SCHEMA_VERSION,
        SchemaVersionError,
        ensure_schema_version_table,
        write_schema_version,
    )

    with TempWorkspace() as ws:
        learning_db = ws / "learning.db"
        memory_db = ws / "memory.db"
        sqlite3.connect(learning_db).close()
        with sqlite3.connect(memory_db) as conn:
            ensure_schema_version_table(conn)
            # Ahead of this binary by a margin -> must be refused.
            write_schema_version(conn, SUPPORTED_SCHEMA_VERSION + 5)

        before = _tables(memory_db)
        refused = False
        try:
            mr.apply_deferred(learning_db, memory_db)
        except SchemaVersionError:
            refused = True
        after = _tables(memory_db)

        held = refused and (before == after)
        detail = {"index": index}
        if not held:
            detail.update(
                refused=refused,
                added_tables=sorted(after - before),
                removed_tables=sorted(before - after),
            )
        return TrialOutcome(index=index, held=held, detail=detail)


def run(n_trials: int = 200, seed: int = 0):
    return run_trials(
        name="exp3_migration_downgrade",
        guarantee="newer-stamped DB refused on deferred pass with zero mutation",
        metric_name="refuse-and-preserve rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real migration_runner.apply_deferred against a DB stamped "
            "SUPPORTED_SCHEMA_VERSION+5; asserts SchemaVersionError and an "
            "unchanged table set."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} "
          f"({result.metric_value:.4f})")
