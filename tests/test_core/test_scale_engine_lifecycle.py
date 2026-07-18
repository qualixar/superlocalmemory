"""Contract tests for the staged Cozo/Lance projection lifecycle."""
from __future__ import annotations

import json
import os
import sqlite3
import struct
from pathlib import Path

import pytest
import sqlite_vec

from superlocalmemory.core.scale_engine import ScaleEngineError, ScaleEngineManager
from superlocalmemory.storage.sqlite_vectors import (
    CanonicalVectorError,
    count_canonical_vectors,
)


class _Config:
    def __init__(self, root: Path) -> None:
        self.base_dir = root
        self.data_dir = root
        self.db_path = root / "memory.db"
        self.active_profile = "default"
        self.scale_engine_state = "local_core"
        self.graph_backend = "auto"
        self.vector_backend = "auto"
        self.saved = 0

    def save(self) -> None:
        self.saved += 1


class _Graph:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.counts_path = self.path / "counts.json"

    def bulk_import_from_sqlite(self, conn, profile_id):
        nodes = conn.execute(
            "SELECT COUNT(*) FROM canonical_entities WHERE profile_id=?", (profile_id,)
        ).fetchone()[0]
        edges = conn.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT 1 FROM graph_edges WHERE profile_id=? "
            "GROUP BY source_id, target_id, COALESCE(edge_type, 'related')"
            ")",
            (profile_id,),
        ).fetchone()[0]
        self.counts_path.write_text(json.dumps({"entities": nodes, "edges": edges}))

    def health_check(self):
        counts = json.loads(self.counts_path.read_text())
        return {"status": "active", **counts}

    def close(self):
        pass


class _Vectors:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.count_path = self.path / "count.txt"

    def bulk_import_from_sqlite(self, conn, profile_id="default"):
        count = conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings fe "
            "JOIN embedding_metadata em ON em.vec_rowid = fe.rowid "
            "JOIN atomic_facts af ON af.fact_id = em.fact_id "
            "AND af.profile_id = em.profile_id "
            "WHERE af.profile_id = ? AND fe.profile_id = af.profile_id",
            (profile_id,),
        ).fetchone()[0]
        self.count_path.write_text(str(count))

    def health_check(self):
        return {"status": "active", "vectors": int(self.count_path.read_text())}

    def close(self):
        pass


def _factory(cozo: Path, lance: Path):
    return _Graph(cozo), _Vectors(lance)


def _make_legacy_projection(lifecycle: ScaleEngineManager) -> tuple[Path, Path]:
    """Create the structural signature produced by the pre-v3.7 backends."""
    cozo, lance = lifecycle.active_paths
    (cozo / "graph").mkdir(parents=True)
    (lance / "embeddings.lance").mkdir(parents=True)
    return cozo, lance


@pytest.fixture
def manager(tmp_path):
    db = sqlite3.connect(tmp_path / "memory.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript(
        """
        CREATE TABLE graph_edges (
            source_id TEXT, target_id TEXT, edge_type TEXT, weight REAL, profile_id TEXT
        );
        CREATE TABLE canonical_entities (
            entity_id TEXT, canonical_name TEXT, entity_type TEXT,
            first_seen TEXT, last_seen TEXT, fact_count INTEGER, profile_id TEXT
        );
        CREATE TABLE atomic_facts (
            fact_id TEXT, canonical_entities_json TEXT, lifecycle TEXT, profile_id TEXT
        );
        CREATE VIRTUAL TABLE fact_embeddings USING vec0(
            profile_id TEXT PARTITION KEY,
            embedding float[768] distance_metric=cosine
        );
        CREATE TABLE embedding_metadata (
            vec_rowid INTEGER PRIMARY KEY,
            fact_id TEXT NOT NULL UNIQUE,
            profile_id TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT '',
            dimension INTEGER NOT NULL DEFAULT 768
        );
        INSERT INTO graph_edges VALUES
          ('a', 'b', 'related', 1.0, 'default'), ('b', 'c', 'related', 1.0, 'default');
        INSERT INTO canonical_entities VALUES
          ('entity-a', 'Entity A', 'concept', '2026-01-01', '2026-01-01', 1, 'default'),
          ('entity-b', 'Entity B', 'concept', '2026-01-01', '2026-01-01', 1, 'default');
        INSERT INTO atomic_facts VALUES
          ('a', '["entity-a"]', 'active', 'default'),
          ('b', '["entity-b"]', 'active', 'default'),
          ('foreign', '[]', 'active', 'other');
        """
    )
    vector = struct.pack("<768f", *([0.25] * 768))
    for fact_id, profile_id in (("a", "default"), ("b", "default"), ("foreign", "other")):
        cursor = db.execute(
            "INSERT INTO fact_embeddings(profile_id, embedding) VALUES (?, ?)",
            (profile_id, vector),
        )
        db.execute(
            "INSERT INTO embedding_metadata(vec_rowid, fact_id, profile_id) "
            "VALUES (?, ?, ?)",
            (cursor.lastrowid, fact_id, profile_id),
        )
    db.commit()
    db.close()
    cfg = _Config(tmp_path)
    return ScaleEngineManager(cfg, backend_factory=_factory), cfg


def test_prepare_verify_promote_preserves_rollback_copy(manager):
    lifecycle, cfg = manager
    # Existing unsafe paths are retained as a rollback point, not overwritten.
    old_cozo, old_lance = lifecycle.active_paths
    old_cozo.mkdir(parents=True)
    old_lance.mkdir(parents=True)
    (old_cozo / "old").write_text("cozo")
    (old_lance / "old").write_text("lance")

    prepared = lifecycle.prepare()
    assert prepared["state"] == "prepared"
    verified = lifecycle.verify(prepared["stage_id"])
    assert verified["state"] == "verified"
    promoted = lifecycle.promote(prepared["stage_id"])

    assert promoted["state"] == "promoted"
    assert cfg.scale_engine_state == "promoted"
    assert (old_cozo / "counts.json").exists()
    assert (old_lance / "count.txt").exists()
    assert (lifecycle.backup_root / promoted["backup_id"] / "cozo" / "old").read_text() == "cozo"
    assert cfg.saved == 3


def test_verify_rejects_a_stage_when_canonical_data_changed(manager):
    lifecycle, _ = manager
    prepared = lifecycle.prepare()
    db = sqlite3.connect(lifecycle.db_path)
    db.execute("INSERT INTO graph_edges VALUES ('c', 'd', 'related', 1.0, 'default')")
    db.commit()
    db.close()
    with pytest.raises(ScaleEngineError, match="changed after preparation"):
        lifecycle.verify(prepared["stage_id"])


def test_projection_contract_normalizes_legacy_duplicate_edges_by_max_weight(manager):
    """Raw legacy duplicates are one logical edge and retain strongest weight."""
    lifecycle, _ = manager
    with sqlite3.connect(lifecycle.db_path) as db:
        initial_counts = lifecycle._canonical_counts(db)
        initial_fingerprint = lifecycle._projection_fingerprint(db, initial_counts)
        db.execute(
            "INSERT INTO graph_edges VALUES ('a', 'b', 'related', 0.25, 'default')"
        )
        weaker_counts = lifecycle._canonical_counts(db)
        weaker_fingerprint = lifecycle._projection_fingerprint(db, weaker_counts)
        db.execute(
            "INSERT INTO graph_edges VALUES ('a', 'b', 'related', 1.5, 'default')"
        )
        stronger_counts = lifecycle._canonical_counts(db)
        stronger_fingerprint = lifecycle._projection_fingerprint(db, stronger_counts)

    assert initial_counts["edges"] == 2
    assert weaker_counts == initial_counts
    assert weaker_fingerprint == initial_fingerprint
    assert stronger_counts == initial_counts
    assert stronger_fingerprint != initial_fingerprint


def test_prepare_excludes_foreign_profile_vectors_from_parity(manager):
    lifecycle, _ = manager
    prepared = lifecycle.prepare()
    # The fixture contains one vector owned by ``other``.  Promotion is
    # default-profile-only, so both manifest and staged Lance count stay at 2.
    assert prepared["canonical"]["vectors"] == 2
    assert prepared["observed"]["vectors"] == 2
    assert lifecycle.verify(prepared["stage_id"])["state"] == "verified"


def test_prepare_rejects_metadata_that_points_to_a_missing_vector(manager):
    lifecycle, _ = manager
    with sqlite3.connect(lifecycle.db_path) as db:
        db.execute(
            "INSERT INTO atomic_facts VALUES (?, ?, ?, ?)",
            ("missing-vector", "[]", "active", "default"),
        )
        db.execute(
            "INSERT INTO embedding_metadata(vec_rowid, fact_id, profile_id) "
            "VALUES (?, ?, ?)",
            (9999, "missing-vector", "default"),
        )

    with pytest.raises(ScaleEngineError, match="canonical vector"):
        lifecycle.prepare()


def test_verify_translates_vector_corruption_to_scale_engine_error(manager):
    lifecycle, _ = manager
    prepared = lifecycle.prepare()
    with sqlite3.connect(lifecycle.db_path) as db:
        db.execute(
            "INSERT INTO atomic_facts VALUES (?, ?, ?, ?)",
            ("late-missing-vector", "[]", "active", "default"),
        )
        db.execute(
            "INSERT INTO embedding_metadata(vec_rowid, fact_id, profile_id) "
            "VALUES (?, ?, ?)",
            (9998, "late-missing-vector", "default"),
        )

    with pytest.raises(ScaleEngineError, match="canonical vector"):
        lifecycle.verify(prepared["stage_id"])


def test_prepare_rejects_vector_partition_that_disagrees_with_fact_profile(manager):
    lifecycle, _ = manager
    with sqlite3.connect(lifecycle.db_path) as db:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        rowid = db.execute(
            "SELECT vec_rowid FROM embedding_metadata WHERE fact_id='a'"
        ).fetchone()[0]
        blob = db.execute(
            "SELECT embedding FROM fact_embeddings WHERE rowid=?", (rowid,)
        ).fetchone()[0]
        db.execute("DELETE FROM fact_embeddings WHERE rowid=?", (rowid,))
        db.execute(
            "INSERT INTO fact_embeddings(rowid, profile_id, embedding) VALUES (?, ?, ?)",
            (rowid, "other", blob),
        )

    with pytest.raises(ScaleEngineError, match="canonical vector"):
        lifecycle.prepare()


def test_shadow_only_vector_payload_is_not_treated_as_empty(tmp_path):
    db = sqlite3.connect(tmp_path / "shadow-only.db")
    db.executescript(
        """
        CREATE TABLE fact_embeddings_rowids (
            rowid INTEGER PRIMARY KEY, id INTEGER, chunk_id INTEGER, chunk_offset INTEGER
        );
        CREATE TABLE fact_embeddings_vector_chunks00 (
            rowid INTEGER PRIMARY KEY, vectors BLOB NOT NULL
        );
        INSERT INTO fact_embeddings_rowids VALUES (1, 1, 1, 0);
        INSERT INTO fact_embeddings_vector_chunks00 VALUES (1, X'00000000');
        """
    )

    with pytest.raises(CanonicalVectorError, match="shadow"):
        count_canonical_vectors(db, "default")


def test_rollback_requires_explicit_backup_and_returns_to_local_core(manager):
    lifecycle, cfg = manager
    prepared = lifecycle.prepare()
    lifecycle.verify(prepared["stage_id"])
    promoted = lifecycle.promote(prepared["stage_id"])
    result = lifecycle.rollback(promoted["backup_id"])
    assert result["state"] == "local_core"
    assert cfg.scale_engine_state == "local_core"


def test_status_distinguishes_legacy_paths_from_live_scale_routing(manager):
    """Existing projection folders are not proof that retrieval uses them."""
    lifecycle, _ = manager
    cozo, lance = _make_legacy_projection(lifecycle)

    status = lifecycle.status()

    assert status["paths_present"] == {"cozo": True, "lance": True}
    assert status["active"] == {"cozo": False, "lance": False}
    assert status["retrieval_routing"] == "canonical_sqlite"
    assert status["legacy_projection_candidate"] is True
    assert status["legacy_candidate_requires_confirmation"] is True


def test_status_does_not_treat_bare_directory_names_as_a_legacy_projection(manager):
    lifecycle, _ = manager
    for path in lifecycle.active_paths:
        path.mkdir(parents=True)

    status = lifecycle.status()

    assert status["paths_present"] == {"cozo": True, "lance": True}
    assert status["legacy_projection_candidate"] is False
    assert status["legacy_candidate_requires_confirmation"] is False


def test_status_requires_daemon_backend_health_after_a_completed_promotion(manager):
    lifecycle, cfg = manager
    _make_legacy_projection(lifecycle)
    lifecycle.adopt_legacy_projection()
    db = sqlite3.connect(lifecycle.db_path)
    db.execute("CREATE TABLE backend_status (backend_name TEXT, status TEXT)")
    db.execute("INSERT INTO backend_status VALUES ('cozo', 'active')")
    db.execute("INSERT INTO backend_status VALUES ('lancedb', 'failed')")
    db.commit()
    db.close()

    status = lifecycle.status()

    assert cfg.scale_engine_state == "promoted"
    assert status["last_daemon_observation"] == {"cozo": "active", "lance": "failed"}
    assert status["active"] == {"cozo": False, "lance": False}
    assert status["retrieval_routing"] == "daemon_runtime_check_required"


def test_adopt_legacy_rebuilds_verifies_and_promotes_current_projection(manager):
    """A legacy v3.5 projection is replaced only after fresh parity proof."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    (old_cozo / "legacy-marker").write_text("cozo")
    (old_lance / "legacy-marker").write_text("lance")

    result = lifecycle.adopt_legacy_projection()

    assert result is not None
    assert result["state"] == "promoted"
    assert cfg.scale_engine_state == "promoted"
    assert lifecycle.status()["retrieval_routing"] == "daemon_runtime_check_required"
    assert lifecycle.status()["active"] == {"cozo": False, "lance": False}
    backup = lifecycle.backup_root / result["backup_id"]
    assert (backup / "cozo" / "legacy-marker").read_text() == "cozo"
    assert (backup / "lance" / "legacy-marker").read_text() == "lance"


def test_adoption_fsyncs_each_renamed_projection_directory(manager, monkeypatch):
    lifecycle, _ = manager
    _make_legacy_projection(lifecycle)
    seen: list[Path] = []
    monkeypatch.setattr(
        ScaleEngineManager,
        "_fsync_directory",
        staticmethod(lambda path: seen.append(path)),
    )

    result = lifecycle.adopt_legacy_projection()

    stage_dir = lifecycle.staging_root / result["stage_id"]
    backup_dir = lifecycle.backup_root / result["backup_id"]
    assert lifecycle.data_dir in seen
    assert stage_dir in seen
    assert lifecycle.backup_root in seen
    assert backup_dir in seen


def test_adopt_legacy_is_a_noop_without_legacy_projection(manager):
    lifecycle, cfg = manager

    assert lifecycle.adopt_legacy_projection() is None
    assert cfg.scale_engine_state == "local_core"


def test_failed_legacy_adoption_preserves_paths_and_allows_corrected_retry(manager, monkeypatch):
    """A rejected stage stays inspectable without permanently blocking adoption."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    (old_cozo / "legacy-marker").write_text("cozo")
    (old_lance / "legacy-marker").write_text("lance")
    verify = lifecycle._verify

    def reject_parity(_: str):
        raise ScaleEngineError("simulated parity failure")

    monkeypatch.setattr(lifecycle, "_verify", reject_parity)

    with pytest.raises(ScaleEngineError, match="simulated parity failure"):
        lifecycle.adopt_legacy_projection()

    assert cfg.scale_engine_state == "local_core"
    assert (old_cozo / "legacy-marker").read_text() == "cozo"
    assert (old_lance / "legacy-marker").read_text() == "lance"
    rejected = lifecycle.status()
    assert rejected["legacy_projection_candidate"] is True
    assert rejected["migration_repair_required"] is False
    assert len(rejected["stages"]) == 1
    rejected_manifest = rejected["stages"][0]
    assert rejected_manifest["state"] == "rejected"
    rejected_dir = lifecycle.staging_root / rejected_manifest["stage_id"]
    assert (rejected_dir / lifecycle.MANIFEST_NAME).exists()
    assert not (rejected_dir / "cozo").exists()
    assert not (rejected_dir / "lance").exists()

    monkeypatch.setattr(lifecycle, "_verify", verify)
    promoted = lifecycle.adopt_legacy_projection()

    assert promoted is not None
    assert promoted["state"] == "promoted"
    assert cfg.scale_engine_state == "promoted"


def test_successful_adoption_retires_a_prior_prepared_retry_payload(manager):
    """A 3.7.3-style rejected prepared stage cannot leak disk indefinitely."""
    lifecycle, cfg = manager
    _make_legacy_projection(lifecycle)
    prior = lifecycle._prepare()
    prior_dir = lifecycle.staging_root / prior["stage_id"]
    assert (prior_dir / "cozo").exists()
    assert (prior_dir / "lance").exists()
    cfg.scale_engine_state = "local_core"

    promoted = lifecycle.adopt_legacy_projection()

    assert promoted is not None
    assert promoted["retired_stages"] == [prior["stage_id"]]
    prior_manifest = json.loads((prior_dir / lifecycle.MANIFEST_NAME).read_text())
    assert prior_manifest["state"] == "superseded"
    assert not (prior_dir / "cozo").exists()
    assert not (prior_dir / "lance").exists()


def test_legacy_adoption_refuses_a_second_writer(manager):
    """Concurrent daemon starts cannot both move active projection paths."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    lock_path = lifecycle.lifecycle_lock_path
    lock_path.write_text(json.dumps({"pid": os.getpid()}))

    with pytest.raises(ScaleEngineError, match="already in progress"):
        lifecycle.adopt_legacy_projection()

    assert cfg.scale_engine_state == "local_core"
    assert not lifecycle.staging_root.exists()


def test_legacy_adoption_recovers_a_lock_left_by_a_dead_process(manager):
    """A crashed upgrader cannot block every future daemon start forever."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    lock_path = lifecycle.lifecycle_lock_path
    lock_path.write_text(json.dumps({"pid": 999_999_999, "started_at": "2026-01-01T00:00:00Z"}))

    result = lifecycle.adopt_legacy_projection()

    assert result is not None
    assert cfg.scale_engine_state == "promoted"
    assert not lock_path.exists()


def test_recovery_reverses_an_interrupted_half_promotion(manager):
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    (old_cozo / "legacy-marker").write_text("cozo")
    (old_lance / "legacy-marker").write_text("lance")
    prepared = lifecycle.prepare()
    lifecycle.verify(prepared["stage_id"])
    stage_dir = lifecycle.staging_root / prepared["stage_id"]
    backup_id = "interrupted-backup"
    backup_dir = lifecycle.backup_root / backup_id
    lifecycle.backup_root.mkdir(parents=True)
    backup_dir.mkdir()
    journal = {
        "schema_version": lifecycle.SCHEMA_VERSION,
        "state": "intent",
        "stage_id": prepared["stage_id"],
        "backup_id": backup_id,
        "moves": [],
    }
    journal["moves"].append({"name": "cozo", "kind": "active_to_backup", "state": "intent"})
    lifecycle._write_promotion_journal(journal)
    os.replace(old_cozo, backup_dir / "cozo")
    # Simulate a crash after the rename but before the completion journal write.
    journal["moves"].append({"name": "cozo", "kind": "stage_to_active", "state": "intent"})
    lifecycle._write_promotion_journal(journal)
    os.replace(stage_dir / "cozo", old_cozo)

    result = lifecycle.recover_interrupted_promotion()

    assert result == "reversed_interrupted_promotion"
    assert cfg.scale_engine_state == "local_core"
    assert (old_cozo / "legacy-marker").read_text() == "cozo"
    assert (old_lance / "legacy-marker").read_text() == "lance"
    assert (stage_dir / "cozo").exists()
    assert not lifecycle.promotion_journal_path.exists()


def test_adoption_preserves_stage_when_promotion_recovery_is_unresolved(manager, monkeypatch):
    """Never retire bytes that an unresolved promotion journal still needs."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    (old_cozo / "legacy-marker").write_text("cozo")
    (old_lance / "legacy-marker").write_text("lance")
    replace = ScaleEngineManager._replace_durable
    calls = 0

    def fail_forward_and_first_reverse(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated rename failure")
        if calls == 5:
            raise ScaleEngineError("simulated reverse recovery failure")
        replace(source, target)

    monkeypatch.setattr(
        ScaleEngineManager,
        "_replace_durable",
        staticmethod(fail_forward_and_first_reverse),
    )

    with pytest.raises(ScaleEngineError, match="automatic recovery needs repair"):
        lifecycle.adopt_legacy_projection()

    status = lifecycle.status()
    assert lifecycle.promotion_journal_path.exists()
    assert status["migration_repair_required"] is True
    assert cfg.scale_engine_state == "verified"
    assert status["stages"][0]["state"] == "verified"
    stage_dir = lifecycle.staging_root / status["stages"][0]["stage_id"]
    assert (stage_dir / "lance").exists()

    monkeypatch.setattr(
        ScaleEngineManager,
        "_replace_durable",
        staticmethod(replace),
    )
    assert lifecycle.recover_interrupted_promotion() == "reversed_interrupted_promotion"
    assert (old_cozo / "legacy-marker").read_text() == "cozo"
    assert (old_lance / "legacy-marker").read_text() == "lance"
    assert not lifecycle.promotion_journal_path.exists()


def test_adoption_succeeds_when_committed_journal_cleanup_recovers(manager, monkeypatch):
    """A transient committed-journal unlink failure is still a promotion."""
    lifecycle, cfg = manager
    old_cozo, old_lance = _make_legacy_projection(lifecycle)
    (old_cozo / "legacy-marker").write_text("cozo")
    (old_lance / "legacy-marker").write_text("lance")
    unlink = Path.unlink
    journal_unlinks = 0

    def fail_first_journal_unlink(path: Path, *args, **kwargs) -> None:
        nonlocal journal_unlinks
        if path == lifecycle.promotion_journal_path:
            journal_unlinks += 1
            if journal_unlinks == 1:
                raise OSError("simulated first journal unlink failure")
        unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_journal_unlink)

    promoted = lifecycle.adopt_legacy_projection()

    assert promoted is not None
    assert promoted["state"] == "promoted"
    assert cfg.scale_engine_state == "promoted"
    assert journal_unlinks == 2
    assert not lifecycle.promotion_journal_path.exists()
    assert lifecycle.status()["migration_repair_required"] is False
    backup = lifecycle.backup_root / promoted["backup_id"]
    assert (backup / "cozo" / "legacy-marker").read_text() == "cozo"
    assert (backup / "lance" / "legacy-marker").read_text() == "lance"


def test_recovery_reverses_an_interrupted_rollback(manager):
    lifecycle, cfg = manager
    _make_legacy_projection(lifecycle)
    promoted = lifecycle.adopt_legacy_projection()
    backup_id = promoted["backup_id"]
    active_cozo, _ = lifecycle.active_paths
    displaced = lifecycle.backup_root / "rollback-displaced-interrupted"
    displaced.mkdir()
    journal = {
        "schema_version": lifecycle.SCHEMA_VERSION,
        "operation": "rollback",
        "state": "intent",
        "backup_id": backup_id,
        "displaced_id": displaced.name,
        "moves": [{"name": "cozo", "kind": "active_to_displaced", "state": "intent"}],
    }
    lifecycle._write_promotion_journal(journal)
    os.replace(active_cozo, displaced / "cozo")

    result = lifecycle.recover_interrupted_promotion()

    assert result == "reversed_interrupted_rollback"
    assert cfg.scale_engine_state == "promoted"
    assert active_cozo.exists()
    assert (lifecycle.backup_root / backup_id / "cozo").exists()
    assert not lifecycle.promotion_journal_path.exists()


def test_legacy_adoption_uses_default_projection_without_changing_user_profile(manager):
    _, cfg = manager
    cfg.active_profile = "client-acme"
    lifecycle = ScaleEngineManager(cfg, backend_factory=_factory, profile_id="default")
    old_cozo, old_lance = _make_legacy_projection(lifecycle)

    result = lifecycle.adopt_legacy_projection()

    assert result is not None
    assert result["state"] == "promoted"
    assert cfg.active_profile == "client-acme"
