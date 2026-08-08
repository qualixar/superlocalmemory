"""Experiment 4 — Backup-restore atomicity (rollback on partial failure).

Guarantee: if a restore fails partway through, the live data set is rolled back
to exactly the bytes that were live before the restore began — never a mix of
old and new epochs, never the half-written backup epoch.

Method: for each trial we build a real ``BackupCoordinator`` over two live
SQLite stores, capture a backup set, replace the live stores with a distinct
"new epoch", then inject a disk error on the second store's staging copy. The
real restore must raise ``BackupRestoreError`` and leave both live stores at the
new-epoch bytes (the pre-restore snapshot), with no staging/snapshot residue.
Fault injection patches ``shutil.copy2`` — the coordinator itself is unmodified.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from unittest.mock import patch

from _harness import TempWorkspace, TrialOutcome, run_trials


def _make_sqlite(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS marker (val TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _trial(index: int) -> TrialOutcome:
    import superlocalmemory.infra.backup as backup_mod
    from superlocalmemory.infra.backup import BackupCoordinator, BackupRestoreError

    subset = ("memory.db", "learning.db")
    with TempWorkspace() as ws:
        base_dir = ws / "live"
        backup_dir = ws / "backups"
        base_dir.mkdir()
        backup_dir.mkdir()

        tag = uuid.uuid4().hex[:6]
        _make_sqlite(base_dir / "memory.db", f"orig_mem_{tag}")
        _make_sqlite(base_dir / "learning.db", f"orig_learn_{tag}")

        coord = BackupCoordinator(
            managed_databases=subset, base_dir=base_dir, backup_dir=backup_dir,
        )
        manifest = coord.create_backup_set()

        # Replace live stores with a distinguishable new epoch.
        _make_sqlite(base_dir / "memory.db", f"new_mem_{tag}")
        _make_sqlite(base_dir / "learning.db", f"new_learn_{tag}")
        new_mem = (base_dir / "memory.db").read_bytes()
        new_learn = (base_dir / "learning.db").read_bytes()

        # Inject a failure on the SECOND staging copy.
        real_copy2 = backup_mod.shutil.copy2
        calls = {"n": 0}

        def _failing_copy2(src, dst):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_copy2(src, dst)
            raise OSError("injected disk error on second store")

        raised = False
        with patch.object(backup_mod.shutil, "copy2", _failing_copy2):
            try:
                coord.restore_from_manifest(manifest)
            except BackupRestoreError:
                raised = True

        rolled_back = (
            (base_dir / "memory.db").read_bytes() == new_mem
            and (base_dir / "learning.db").read_bytes() == new_learn
        )
        no_residue = (
            not list(base_dir.glob("*.pre_restore"))
            and not list(base_dir.glob("*.restore_staging"))
        )
        held = raised and rolled_back and no_residue
        detail = {"index": index}
        if not held:
            detail.update(raised=raised, rolled_back=rolled_back,
                          no_residue=no_residue)
        return TrialOutcome(index=index, held=held, detail=detail)


def run(n_trials: int = 200, seed: int = 0):
    return run_trials(
        name="exp4_backup_restore_atomicity",
        guarantee="partial-restore failure rolls live data back to pre-restore bytes",
        metric_name="rollback+clean rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Real BackupCoordinator create/restore; failure injected on the "
            "second staging copy via shutil.copy2; asserts BackupRestoreError, "
            "pre-restore bytes intact, no staging/snapshot residue."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} "
          f"({result.metric_value:.4f})")
