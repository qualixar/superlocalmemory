# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com

"""Tests for BackupCoordinator restore atomicity and LanceDB directory handling.

Tests use real temp directories and real SQLite databases — no MagicMock on
any path under test. Failure injection is done via monkeypatching Path.rename
(lance/ directory) or the coordinator's write into a live store.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.infra.backup import (
    BackupCoordinator,
    BackupRestoreError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUBSET: tuple[str, ...] = ("memory.db", "learning.db")


def _make_sqlite(path: Path, marker: str) -> None:
    """Create a minimal SQLite database at *path* with a known marker row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS marker (val TEXT)")
    conn.execute("INSERT INTO marker VALUES (?)", (marker,))
    conn.commit()
    conn.close()


def _read_marker(path: Path) -> str:
    """Return the first marker value from a SQLite database."""
    conn = sqlite3.connect(str(path))
    row = conn.execute("SELECT val FROM marker").fetchone()
    conn.close()
    return row[0] if row else ""


def _all_markers(path: Path) -> list[str]:
    """Return every marker value, in insertion order."""
    conn = sqlite3.connect(str(path))
    try:
        return [r[0] for r in conn.execute("SELECT val FROM marker ORDER BY rowid")]
    finally:
        conn.close()


def _coord(base_dir: Path, backup_dir: Path) -> BackupCoordinator:
    return BackupCoordinator(
        managed_databases=_SUBSET,
        base_dir=base_dir,
        backup_dir=backup_dir,
    )


def _fail_live_write_on_call(
    coord: BackupCoordinator, monkeypatch: pytest.MonkeyPatch, failing_call: int
) -> list[str]:
    """Make the Nth write into a live store raise; every other write is real.

    Writes are counted across Phase C and the rollback, so ``failing_call=2``
    fails the second store's restore after the first store was written, and
    lets the rollback writes that follow go through. Returns the target names
    in call order.
    """
    real_write = coord._write_into_live_store
    calls: list[str] = []

    def _write(source: Path, target: Path) -> None:
        calls.append(target.name)
        if len(calls) == failing_call:
            raise OSError(f"simulated disk error writing {target.name}")
        real_write(source, target)

    monkeypatch.setattr(coord, "_write_into_live_store", _write)
    return calls


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_env(tmp_path: Path):
    """Yield (base_dir, backup_dir, coordinator) with two live SQLite stores."""
    base_dir = tmp_path / "live"
    backup_dir = tmp_path / "backups"
    base_dir.mkdir()
    backup_dir.mkdir()

    _make_sqlite(base_dir / "memory.db", "original_memory")
    _make_sqlite(base_dir / "learning.db", "original_learning")

    coord = _coord(base_dir, backup_dir)
    yield base_dir, backup_dir, coord


# ---------------------------------------------------------------------------
# Happy-path: full create + restore cycle
# ---------------------------------------------------------------------------

class TestHappyPathRestore:
    """Create a backup set, mutate live DBs, restore — verify round-trip."""

    def test_all_stores_restored_and_snapshots_cleaned_up(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        base_dir, backup_dir, coord = live_env

        manifest = coord.create_backup_set()

        # Mutate the live stores so we can verify restoration
        conn = sqlite3.connect(str(base_dir / "memory.db"))
        conn.execute("DELETE FROM marker")
        conn.execute("INSERT INTO marker VALUES ('mutated_memory')")
        conn.commit()
        conn.close()

        coord.restore_from_manifest(manifest)

        assert _read_marker(base_dir / "memory.db") == "original_memory"
        assert _read_marker(base_dir / "learning.db") == "original_learning"

        # Pre-restore snapshot files must be gone after a successful restore
        pre_restore_files = list(base_dir.glob("*.pre_restore"))
        assert pre_restore_files == [], (
            f"Pre-restore snapshots not cleaned up: {pre_restore_files}"
        )

    def test_restore_staging_files_not_left_behind(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        base_dir, backup_dir, coord = live_env
        manifest = coord.create_backup_set()
        coord.restore_from_manifest(manifest)

        staging_files = list(base_dir.glob("*.restore_staging"))
        assert staging_files == [], (
            f"Staging files left on disk: {staging_files}"
        )


# ---------------------------------------------------------------------------
# Atomicity: mid-restore failure triggers rollback
# ---------------------------------------------------------------------------

class TestMidRestoreRollback:
    """Inject a failure after the first store is written; verify rollback."""

    def test_live_content_restored_to_original_after_partial_write(
        self,
        live_env: tuple[Path, Path, BackupCoordinator],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_dir, backup_dir, coord = live_env

        manifest = coord.create_backup_set()

        # Replace the live stores with "different epoch" data so we can
        # clearly distinguish original vs backup
        _make_sqlite(base_dir / "memory.db", "new_epoch_memory")
        _make_sqlite(base_dir / "learning.db", "new_epoch_learning")
        new_memory_rows = _all_markers(base_dir / "memory.db")
        new_learning_rows = _all_markers(base_dir / "learning.db")

        # The backup holds the original epoch. Fail Phase C on the second
        # store, AFTER the first store (memory.db) has been written, so the
        # rollback has real work to do.
        calls = _fail_live_write_on_call(coord, monkeypatch, failing_call=2)

        with pytest.raises(BackupRestoreError, match="rolled back"):
            coord.restore_from_manifest(manifest)

        # memory.db was written, learning.db failed, then memory.db was rolled
        # back. Anything else means the injection did not hit a partial write.
        assert calls == ["memory.db", "learning.db", "memory.db"]
        # After rollback: live stores must hold the "new epoch" content, not
        # the backup epoch — the rollback restores what was live BEFORE the
        # restore attempt began. Content, not bytes: SQLite rewrites header
        # counters when it writes a database.
        assert _all_markers(base_dir / "memory.db") == new_memory_rows, (
            "memory.db was not rolled back: contains unexpected content"
        )
        assert _all_markers(base_dir / "learning.db") == new_learning_rows, (
            "learning.db was not rolled back to pre-restore state"
        )

    def test_pre_restore_snapshots_cleaned_up_after_rollback(
        self,
        live_env: tuple[Path, Path, BackupCoordinator],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_dir, backup_dir, coord = live_env
        manifest = coord.create_backup_set()

        _fail_live_write_on_call(coord, monkeypatch, failing_call=2)

        with pytest.raises(BackupRestoreError):
            coord.restore_from_manifest(manifest)

        # Snapshot files must be cleaned up even on the error path
        pre_restore_files = list(base_dir.glob("*.pre_restore"))
        assert pre_restore_files == [], (
            f"Pre-restore snapshots leaked after rollback: {pre_restore_files}"
        )

    def test_backup_restore_error_raised_not_swallowed(
        self,
        live_env: tuple[Path, Path, BackupCoordinator],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Phase C write failure raises BackupRestoreError, never swallowed."""
        base_dir, backup_dir, coord = live_env
        manifest = coord.create_backup_set()

        # Fail the first write into a live store, after the Phase B snapshots
        # have succeeded.
        _fail_live_write_on_call(coord, monkeypatch, failing_call=1)

        with pytest.raises(BackupRestoreError, match="Restore write phase failed"):
            coord.restore_from_manifest(manifest)


# ---------------------------------------------------------------------------
# LanceDB directory: backup + restore
# ---------------------------------------------------------------------------

class TestLanceDirBackupRestore:
    """Verify that a lance/ directory is captured during backup and restored."""

    def test_lance_dir_survives_backup_and_restore(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        base_dir, backup_dir, coord = live_env

        # Create a fake lance/ directory with a sentinel file
        lance_dir = base_dir / "lance"
        lance_dir.mkdir()
        sentinel = lance_dir / "vectors.lance"
        sentinel.write_bytes(b"fake-vector-data-v1")

        manifest = coord.create_backup_set()

        # Remove the live lance/ directory to simulate a loss scenario
        import shutil
        shutil.rmtree(str(lance_dir))
        assert not lance_dir.exists()

        coord.restore_from_manifest(manifest)

        assert lance_dir.exists(), "lance/ directory was not restored"
        restored_sentinel = lance_dir / "vectors.lance"
        assert restored_sentinel.exists(), "vectors.lance not found in restored lance/"
        assert restored_sentinel.read_bytes() == b"fake-vector-data-v1", (
            "Restored lance/ sentinel content does not match original"
        )

    def test_lance_dir_absent_on_both_sides_is_noop(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        """No lance/ in live OR backup — restore completes without error."""
        base_dir, backup_dir, coord = live_env
        assert not (base_dir / "lance").exists()

        manifest = coord.create_backup_set()
        coord.restore_from_manifest(manifest)  # must not raise

        assert not (base_dir / "lance").exists()

    def test_lance_backup_dir_captured_under_backup_set(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        """After create_backup_set, the backup set dir must contain lance/."""
        base_dir, backup_dir, coord = live_env

        lance_dir = base_dir / "lance"
        lance_dir.mkdir()
        (lance_dir / "index.bin").write_bytes(b"\x00\x01\x02")

        manifest = coord.create_backup_set()

        # The backup set dir is parent of any stored file_path
        backup_set_dir = Path(manifest.stores[0].file_path).parent
        lance_in_backup = backup_set_dir / "lance"
        assert lance_in_backup.is_dir(), (
            f"lance/ was not captured in backup set at {backup_set_dir}"
        )
        assert (lance_in_backup / "index.bin").exists(), (
            "index.bin not found inside captured lance/ directory"
        )

    def test_lance_restore_uses_pre_restore_snapshot_on_failure(
        self,
        live_env: tuple[Path, Path, BackupCoordinator],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If restore fails after lance/ is removed but before rename, rollback restores it."""
        base_dir, backup_dir, coord = live_env

        # Set up live lance/ with known content
        lance_dir = base_dir / "lance"
        lance_dir.mkdir()
        (lance_dir / "live.bin").write_bytes(b"live-vector-epoch")

        manifest = coord.create_backup_set()

        # Replace live lance/ with "later" content
        import shutil
        shutil.rmtree(str(lance_dir))
        lance_dir.mkdir()
        (lance_dir / "live.bin").write_bytes(b"later-epoch-content")

        # Inject failure during the lance rename step: copytree succeeds but
        # rename raises. We do this by patching Path.rename so it fails after
        # all SQLite stores are written.
        real_rename = Path.rename
        rename_calls = [0]

        def _failing_rename(self_path: Path, target: Path) -> Path:
            rename_calls[0] += 1
            if "lance.restore_staging" in str(self_path):
                raise OSError("simulated rename failure for lance staging")
            return real_rename(self_path, target)

        monkeypatch.setattr(Path, "rename", _failing_rename)

        with pytest.raises(BackupRestoreError):
            coord.restore_from_manifest(manifest)

        # Rollback must have restored live lance/ to the "later epoch" state
        assert lance_dir.exists(), "lance/ directory disappeared after rollback"
        assert (lance_dir / "live.bin").read_bytes() == b"later-epoch-content", (
            "lance/ was not rolled back to pre-restore state"
        )

        # No leftover snapshot directories
        assert not (base_dir / "lance.pre_restore").exists(), (
            "lance.pre_restore was not cleaned up after rollback"
        )


# ---------------------------------------------------------------------------
# Guard: unverified manifest and hash mismatch still rejected
# ---------------------------------------------------------------------------

class TestVerificationGuards:
    """Existing verification guards must still work after refactor."""

    def test_unverified_manifest_rejected(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        base_dir, backup_dir, coord = live_env
        manifest = coord.create_backup_set()
        from dataclasses import replace
        bad = replace(manifest, verified=False)
        with pytest.raises(BackupRestoreError, match="unverified"):
            coord.restore_from_manifest(bad)

    def test_tampered_manifest_hash_rejected(
        self, live_env: tuple[Path, Path, BackupCoordinator]
    ) -> None:
        base_dir, backup_dir, coord = live_env
        manifest = coord.create_backup_set()
        from dataclasses import replace
        bad = replace(manifest, manifest_hash="deadbeef" * 8)
        with pytest.raises(BackupRestoreError, match="Manifest hash mismatch"):
            coord.restore_from_manifest(bad)
