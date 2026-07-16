"""Contract tests for the staged Cozo/Lance projection lifecycle."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.scale_engine import ScaleEngineError, ScaleEngineManager


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
        edges = conn.execute("SELECT COUNT(*) FROM graph_edges WHERE profile_id=?", (profile_id,)).fetchone()[0]
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
            "SELECT COUNT(*) FROM fact_embeddings_rowids fer "
            "JOIN atomic_facts af ON af.fact_id = fer.fact_id WHERE af.profile_id = ?",
            (profile_id,),
        ).fetchone()[0]
        self.count_path.write_text(str(count))

    def health_check(self):
        return {"status": "active", "vectors": int(self.count_path.read_text())}

    def close(self):
        pass


def _factory(cozo: Path, lance: Path):
    return _Graph(cozo), _Vectors(lance)


@pytest.fixture
def manager(tmp_path):
    db = sqlite3.connect(tmp_path / "memory.db")
    db.executescript(
        """
        CREATE TABLE graph_edges (source_id TEXT, target_id TEXT, profile_id TEXT);
        CREATE TABLE canonical_entities (entity_id TEXT, profile_id TEXT);
        CREATE TABLE atomic_facts (fact_id TEXT, profile_id TEXT);
        CREATE TABLE fact_embeddings_rowids (fact_id TEXT);
        INSERT INTO graph_edges VALUES ('a', 'b', 'default'), ('b', 'c', 'default');
        INSERT INTO canonical_entities VALUES ('entity-a', 'default'), ('entity-b', 'default');
        INSERT INTO atomic_facts VALUES ('a', 'default'), ('b', 'default'), ('foreign', 'other');
        INSERT INTO fact_embeddings_rowids VALUES ('a'), ('b'), ('foreign');
        """
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
    db.execute("INSERT INTO graph_edges VALUES ('c', 'd', 'default')")
    db.commit()
    db.close()
    with pytest.raises(ScaleEngineError, match="changed after preparation"):
        lifecycle.verify(prepared["stage_id"])


def test_prepare_excludes_foreign_profile_vectors_from_parity(manager):
    lifecycle, _ = manager
    prepared = lifecycle.prepare()
    # The fixture contains one vector owned by ``other``.  Promotion is
    # default-profile-only, so both manifest and staged Lance count stay at 2.
    assert prepared["canonical"]["vectors"] == 2
    assert prepared["observed"]["vectors"] == 2
    assert lifecycle.verify(prepared["stage_id"])["state"] == "verified"


def test_rollback_requires_explicit_backup_and_returns_to_local_core(manager):
    lifecycle, cfg = manager
    prepared = lifecycle.prepare()
    lifecycle.verify(prepared["stage_id"])
    promoted = lifecycle.promote(prepared["stage_id"])
    result = lifecycle.rollback(promoted["backup_id"])
    assert result["state"] == "local_core"
    assert cfg.scale_engine_state == "local_core"
