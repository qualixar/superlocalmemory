# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The restore that runs at start-up: replaces the store, loses nothing it should keep.

A non-technical user presses "Restore memories to before the update". What they
must get: the old store back, a copy of what it replaced, every memory they
added since put back, every memory they deleted or erased still gone -- and
never a restore performed while something else is writing to the store.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from superlocalmemory.core.file_lock import exclusive_lock
from superlocalmemory.storage import backup
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import (
    add_memory,
    add_profile,
    as_4118,
    columns,
    confirm_kind,
    current_store,
    delete_fact,
    delete_memory,
    fact_ids,
    file_digest,
    integrity,
    memory_ids,
    record_erasure,
)


def _scenario(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Use signed wheels"],
               fact_ids=["f1", "f2"])
    add_memory(memory_db, "m2", ["The office moved to Pune"], fact_ids=["f3"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _restore(tmp_path, learning_db, memory_db, point, **kwargs):
    ur.request_restore(point.point_id, requested_by="test", data_root=tmp_path,
                       memory_db=memory_db)
    return ur.perform_pending_restore(tmp_path, memory_db, learning_db, **kwargs)


def _content(db: Path, fact_id: str) -> str | None:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        row = conn.execute("SELECT content FROM atomic_facts WHERE fact_id=?",
                           (fact_id,)).fetchone()
    return row[0] if row else None


def test_restore_replaces_store_and_keeps_safety_copy(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("UPDATE atomic_facts SET content='changed after the update' "
                     "WHERE fact_id='f1'")
        conn.commit()

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert _content(memory_db, "f1") == "Deploys go out on Tuesdays"
    assert integrity(memory_db) == "ok"
    memory_safety = [Path(p) for p in outcome.safety_copies if "memory" in Path(p).name]
    assert memory_safety and memory_safety[0].parent == tmp_path / "pre-restore"
    assert _content(memory_safety[0], "f1") == "changed after the update"


def test_refuses_while_another_daemon_is_alive(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "daemon.pid").write_text(str(other.pid), encoding="utf-8")
        before = file_digest(memory_db)

        outcome = _restore(tmp_path, learning_db, memory_db, point)

        assert outcome.status == "refused" and str(other.pid) in outcome.message
        assert file_digest(memory_db) == before
        assert (tmp_path / "restore-intent.json").exists(), "kept for the next start"
    finally:
        other.kill()
        other.wait()


def test_refuses_while_the_store_is_open_for_writing(tmp_path) -> None:
    """An engine that owns the writer lease -- in this process or another."""
    learning_db, memory_db, point = _scenario(tmp_path)
    delete_fact(memory_db, "f3")
    before = file_digest(memory_db)
    lease = memory_db.resolve().with_name(memory_db.name + ".writer.lock")
    with exclusive_lock(lease):
        outcome = _restore(tmp_path, learning_db, memory_db, point)
    assert outcome.status == "refused"
    assert file_digest(memory_db) == before and "f3" not in fact_ids(memory_db)
    assert (tmp_path / "restore-intent.json").exists()


def test_refuses_when_the_disk_is_too_full(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    delete_fact(memory_db, "f3")
    before = file_digest(memory_db)
    monkeypatch.setattr(ur, "free_bytes", lambda _root: 1024)

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "refused" and "disk" in outcome.message.lower()
    assert file_digest(memory_db) == before
    assert (tmp_path / "restore-intent.json").exists()


def test_erased_profile_stays_erased(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays", "Ravi's phone is 98450"],
               fact_ids=["f1", "f2"])
    add_profile(memory_db, "alice")
    add_memory(memory_db, "ma", ["Alice's address is 12 Elm St", "Alice likes tea"],
               profile="alice", fact_ids=["fa1", "fa2"])
    with closing(sqlite3.connect(memory_db)) as conn:
        # Rows only a whole-profile erasure removes, and a correction that
        # protects f2 from an ordinary delete.
        conn.execute("INSERT INTO canonical_entities (entity_id, profile_id, canonical_name) "
                     "VALUES ('e-alice', 'alice', 'Alice')")
        conn.execute(
            "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
            "successor_fact_id, reason_code, status, version, idempotency_key, "
            "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
            "created_at, updated_at) VALUES ('c1', 'default', 'personal', 'f2', 'f1', "
            "'user', 'applied', 1, 'k', 'a', 'user', 'high', 't', 't')")
        conn.execute(
            "INSERT INTO correction_events (event_id, case_id, profile_id, scope, event_type, "
            "operation_id, actor_id, actor_kind, actor_trust_tier, resulting_version, "
            "system_occurred_at) VALUES ('ev1', 'c1', 'default', 'personal', 'applied', "
            "'op', 'a', 'user', 'high', 1, 't')")
        conn.commit()
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    # After the copy: alice is erased entirely, and one of the owner's facts too.
    delete_memory(memory_db, "ma")
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("DELETE FROM canonical_entities WHERE profile_id='alice'")
        conn.execute("DELETE FROM correction_events WHERE case_id='c1'")
        conn.execute("DELETE FROM correction_cases WHERE case_id='c1'")
        conn.commit()
    record_erasure(memory_db, profile_id="alice", subject_type="profile", subject_id="alice",
                   fact_ids=["fa1", "fa2"], memory_ids={"fa1": "ma", "fa2": "ma"})
    delete_fact(memory_db, "f2")
    record_erasure(memory_db, profile_id="default", subject_type="fact", subject_id="f2",
                   fact_ids=["f2"], memory_ids={"f2": "m1"})

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "restored", outcome.message
    assert fact_ids(memory_db, "alice") == set() and "ma" not in memory_ids(memory_db)
    assert "f2" not in fact_ids(memory_db) and "f1" in fact_ids(memory_db)
    with closing(sqlite3.connect(memory_db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM erasure_receipts").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM projection_tombstones").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM memories WHERE profile_id='alice'"
                            ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM canonical_entities WHERE "
                            "profile_id='alice'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM correction_cases").fetchone()[0] == 0
    assert integrity(memory_db) == "ok"


def test_intent_is_renamed_before_return(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    outcome = _restore(tmp_path, learning_db, memory_db, point)
    assert outcome.status == "restored"
    assert not (tmp_path / "restore-intent.json").exists()
    assert len(list(tmp_path.glob("restore-intent.done-*.json"))) == 1
    recorded = json.loads((tmp_path / "restore-outcome.json").read_text(encoding="utf-8"))
    assert recorded["status"] == "restored" and recorded["point_id"] == point.point_id
    assert ur.perform_pending_restore(tmp_path, memory_db, learning_db) is None


def test_deleted_facts_are_deleted_again(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    delete_memory(memory_db, "m2")                       # takes f3 with it
    outcome = _restore(tmp_path, learning_db, memory_db, point)
    assert outcome.status == "restored"
    assert "f3" not in fact_ids(memory_db) and "m2" not in memory_ids(memory_db)
    assert {"f1", "f2"} <= fact_ids(memory_db)
    # A forget leaves a tombstone (as the product's delete path does), so the
    # copy drops f3 through the tombstone: it is counted with the erasures.
    assert outcome.applied["facts_deleted"] + outcome.applied["facts_erased"] == 1


def test_confirmed_kinds_are_reapplied(tmp_path) -> None:
    learning_db, memory_db, point = _scenario(tmp_path)
    confirm_kind(memory_db, "f1", "rule")
    outcome = _restore(tmp_path, learning_db, memory_db, point)
    assert outcome.reimport_pending

    engine = SimpleNamespace(_db=SimpleNamespace(db_path=memory_db))
    report = ur.reimport_delta(engine, Path(outcome.delta_dir), data_root=tmp_path)

    assert report.kinds_reapplied == 1
    with closing(sqlite3.connect(memory_db)) as conn:
        row = conn.execute("SELECT memory_kind, memory_kind_source, fact_type FROM "
                           "atomic_facts WHERE fact_id='f1'").fetchone()
    assert row == ("rule", "user", "semantic")
    assert json.loads((tmp_path / "restore-outcome.json").read_text(encoding="utf-8"))[
        "reimport_pending"] is False


def test_m052_reapplies_on_the_restored_store(tmp_path) -> None:
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    as_4118(learning_db, memory_db)
    assert mr.apply_all(learning_db, memory_db)["failed"] == []   # the 4.1.19 upgrade
    assert "memory_kind" in columns(memory_db)
    [point] = ur.list_restore_points(tmp_path)

    outcome = _restore(tmp_path, learning_db, memory_db, point)
    assert outcome.status == "restored"
    assert "memory_kind" not in columns(memory_db), "the copy predates the update"

    result = mr.apply_all(learning_db, memory_db)                # the next start
    assert "M052_memory_kinds" in result["applied"]
    assert "memory_kind" in columns(memory_db) and fact_ids(memory_db) == {"f1"}
    assert len(ur.list_restore_points(tmp_path)) == 2, "a fresh copy before re-applying"


# ---------------------------------------------------------------------------
# Re-import through the real engine
# ---------------------------------------------------------------------------

def _engine(config, embedder):
    from superlocalmemory.core.engine import MemoryEngine

    engine = MemoryEngine(config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=embedder):
        engine.initialize()
        engine._embedder = embedder
    return engine


def _store(engine, text: str) -> None:
    from superlocalmemory.core.engine_ingestion import canonical_store, local_trusted_actor_id

    canonical_store(engine, text, source_type="python-api",
                    trusted_actor_id=local_trusted_actor_id("python-api"),
                    require_complete=False)


def _memory_count(db: Path) -> int:
    with closing(sqlite3.connect(f"file:{db}?mode=ro", uri=True)) as conn:
        return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


def test_reimport_is_idempotent(tmp_path, mode_a_config, mock_embedder) -> None:
    memory_db = Path(mode_a_config.db_path)
    learning_db = memory_db.parent / "learning.db"
    engine = _engine(mode_a_config, mock_embedder)
    _store(engine, "The deployment pipeline signs every wheel before upload")
    _store(engine, "Quarterly planning happens in the first week of the quarter")
    engine.close()
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=memory_db.parent / "pre-migration-snapshots")
    point = ur.list_restore_points(memory_db.parent)[0]
    engine = _engine(mode_a_config, mock_embedder)
    _store(engine, "Varun prefers architecture reviews on Thursday mornings")
    engine.close()
    assert _memory_count(memory_db) == 3

    outcome = _restore(memory_db.parent, learning_db, memory_db, point)
    assert outcome.status == "restored" and _memory_count(memory_db) == 2

    engine = _engine(mode_a_config, mock_embedder)
    try:
        first = ur.reimport_delta(engine, Path(outcome.delta_dir))
        second = ur.reimport_delta(engine, Path(outcome.delta_dir))
    finally:
        engine.close()
    assert (first.added, first.failed) == (1, 0), first
    assert (second.added, second.already_present) == (0, 1), second
    assert _memory_count(memory_db) == 3


def test_stale_vector_and_graph_copies_stop_serving_after_a_restore(tmp_path) -> None:
    """LLD §13: the projections describe the replaced store; SQLite serves until a
    fresh stage of the RESTORED store is prepared, verified and promoted."""
    learning_db, memory_db, point = _scenario(tmp_path)
    stage = tmp_path / "scale-staging" / "20261001T000000Z-abcd1234"
    stage.mkdir(parents=True)
    (stage / "scale-engine.json").write_text(json.dumps({
        "schema_version": 1, "stage_id": stage.name, "state": "verified",
        "created_at": "2026-10-01T00:00:00Z", "profile_id": "default"}), encoding="utf-8")
    saved: list[str] = []
    config = SimpleNamespace(base_dir=tmp_path, data_dir=tmp_path, db_path=memory_db,
                             active_profile="default", scale_engine_state="promoted",
                             graph_backend="cozo", vector_backend="lancedb",
                             save=lambda: saved.append(config.scale_engine_state))

    outcome = _restore(tmp_path, learning_db, memory_db, point, config=config)

    assert outcome.projections == "rebuild_scheduled"
    assert (config.scale_engine_state, config.graph_backend, config.vector_backend) == (
        "local_core", "auto", "auto") and saved == ["local_core"]
    assert json.loads((stage / "scale-engine.json").read_text(encoding="utf-8"))["state"] == "superseded"


def test_a_full_disk_part_way_through_changes_nothing(tmp_path, monkeypatch) -> None:
    """CRIT: the disk fills while the staging copy is written (ENOSPC)."""
    import errno

    from superlocalmemory.storage import _restore_boot

    learning_db, memory_db, point = _scenario(tmp_path)
    delete_fact(memory_db, "f3")
    before = file_digest(memory_db)

    def disk_full(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(_restore_boot, "_copy_quiescent", disk_full)

    outcome = _restore(tmp_path, learning_db, memory_db, point)

    assert outcome.status == "failed" and "not changed" in outcome.message
    assert file_digest(memory_db) == before
    assert not (tmp_path / "pre-restore" / "staging").exists()
    assert not (tmp_path / "restore-intent.json").exists(), "set aside, not retried forever"
