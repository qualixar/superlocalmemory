# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

from __future__ import annotations

import json
from pathlib import Path

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M018_ingestion_operations
from superlocalmemory.storage.migrations import M019_derivation_lineage


def _db(path: Path) -> DatabaseManager:
    db = DatabaseManager(path)
    db.initialize(schema)
    with db.raw_connection() as conn:
        M018_ingestion_operations.apply(conn)
        M019_derivation_lineage.apply(conn)
    return db


def _seed(db: DatabaseManager, *, content: str = "Alpha uses SQLite") -> None:
    db.execute(
        "INSERT INTO memories "
        "(memory_id, profile_id, content, session_id, speaker, role) "
        "VALUES ('m1','default',?,'s1','user','user')",
        (content,),
    )
    db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id,profile_id,canonical_name,entity_type,fact_count) "
        "VALUES ('e1','default','Alpha','person',1)",
    )
    db.execute(
        "INSERT INTO entity_profiles "
        "(profile_entry_id,entity_id,profile_id,knowledge_summary,fact_ids_json) "
        "VALUES ('ep1','e1','default',?,'[\"f1\"]')",
        (f"Entity summary: {content}",),
    )
    db.execute(
        "INSERT INTO memory_scenes "
        "(scene_id,profile_id,theme,fact_ids_json,entity_ids_json) "
        "VALUES ('scene1','default','database','[\"f1\"]','[\"e1\"]')",
    )
    db.execute(
        "INSERT INTO graph_edges "
        "(edge_id,profile_id,source_id,target_id,edge_type,weight) "
        "VALUES ('edge1','default','f1','e1','entity',1.0)",
    )
    db.execute(
        "INSERT INTO atomic_facts "
        "(fact_id,memory_id,profile_id,content,fact_type,confidence,"
        "source_turn_ids_json,embedding,fisher_mean,fisher_variance) "
        "VALUES ('f1','m1','default',?,'semantic',0.9,'[\"m1\"]',"
        "'[0.1,0.2]','[0.1,0.2]','[1.0,1.0]')",
        (content,),
    )
    db.execute(
        "INSERT INTO ingestion_operations "
        "(operation_id,profile_id,source_type,idempotency_key,source_hash,"
        "raw_content,state,queryable_fact_ids_json,final_fact_ids_json,"
        "derivation_version,derivation_state_json) "
        "VALUES ('op1','default','cli','idem1','sha',?,'complete',"
        "'[\"f1\"]','[\"f1\"]','v3.7-ingestion-1',"
        "'{\"bm25\":true,\"vector\":true}')",
        (content,),
    )


def test_export_is_deterministic_git_friendly_and_span_linked(tmp_path: Path) -> None:
    from superlocalmemory.core.evidence_bundle import export_evidence_bundle

    db = _db(tmp_path / "source.db")
    _seed(db)
    first = export_evidence_bundle(db, "default", tmp_path / "first")
    second = export_evidence_bundle(db, "default", tmp_path / "second")

    assert first["bundle_id"] == second["bundle_id"]
    assert first["files"] == second["files"]
    facts = [json.loads(line) for line in (tmp_path / "first/facts.jsonl").read_text().splitlines()]
    assert facts[0]["fact_id"] == "f1"
    assert "embedding" not in facts[0]
    spans = [json.loads(line) for line in (tmp_path / "first/source_spans.jsonl").read_text().splitlines()]
    assert spans == [{
        "end": len("Alpha uses SQLite"),
        "fact_id": "f1",
        "operation_id": "op1",
        "start": 0,
        "text_sha256": first["source_spans"]["f1"]["text_sha256"],
    }]
    assert first["derivation_versions"] == ["v3.7-ingestion-1"]
    assert first["files"]["entity_profiles.jsonl"]["count"] == 1
    assert first["files"]["memory_scenes.jsonl"]["count"] == 1
    assert first["lineage_coverage"]["fact"]["legacy_exact_inference"] == 1
    assert first["lineage_coverage"]["profile"]["unresolved"] == 1


def test_verify_rejects_tampering_and_reports_unresolved_links(tmp_path: Path) -> None:
    from superlocalmemory.core.evidence_bundle import (
        export_evidence_bundle,
        verify_evidence_bundle,
    )

    db = _db(tmp_path / "source.db")
    _seed(db)
    bundle = tmp_path / "bundle"
    export_evidence_bundle(db, "default", bundle)
    assert verify_evidence_bundle(bundle).valid is True

    with (bundle / "facts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"fact_id":"forged"}\n')
    report = verify_evidence_bundle(bundle)
    assert report.valid is False
    assert any("sha256" in error for error in report.errors)


def test_round_trip_import_and_rebuild_derived_state(tmp_path: Path) -> None:
    from superlocalmemory.core.evidence_bundle import (
        export_evidence_bundle,
        import_evidence_bundle,
        rebuild_derived_state,
        verify_evidence_bundle,
        verify_rebuild_equivalence,
    )

    source = _db(tmp_path / "source.db")
    _seed(source)
    bundle = tmp_path / "bundle"
    export_evidence_bundle(source, "default", bundle)

    target = _db(tmp_path / "target.db")
    result = import_evidence_bundle(target, bundle, replace=True, rollback_dir=tmp_path / "rollback")
    assert result.valid is True
    assert target.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'")[0]["content"] == "Alpha uses SQLite"
    assert target.execute("SELECT embedding FROM atomic_facts WHERE fact_id='f1'")[0]["embedding"] is None

    class Embedder:
        def embed(self, text: str) -> list[float]:
            assert text == "Alpha uses SQLite"
            return [0.25, 0.75]

    rebuilt = rebuild_derived_state(target, "default", embedder=Embedder())
    assert rebuilt["bm25_rows"] == 1
    assert rebuilt["embeddings"] == 1
    assert target.execute("SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id='f1'")[0]["c"] == 1
    assert verify_evidence_bundle(bundle).valid is True
    equivalence = verify_rebuild_equivalence(target, bundle, profile_id="default")
    assert equivalence["equivalent_on_verified_surfaces"] is True
    assert equivalence["status"] == "partial"
    assert equivalence["verified_surfaces"] == [
        "relational_truth",
        "fts_membership",
        "bm25_tokens",
    ]
    assert "semantic_vector_retrieval" in equivalence["unverified_surfaces"]

    target.execute("UPDATE atomic_facts SET content='tampered' WHERE fact_id='f1'")
    broken = verify_rebuild_equivalence(target, bundle, profile_id="default")
    assert broken["equivalent_on_verified_surfaces"] is False
    assert "atomic_facts" in broken["mismatches"]


def test_replace_requires_rollback_and_preserves_preimage(tmp_path: Path) -> None:
    from superlocalmemory.core.evidence_bundle import (
        export_evidence_bundle,
        import_evidence_bundle,
        verify_evidence_bundle,
    )

    source = _db(tmp_path / "source.db")
    _seed(source, content="new value")
    bundle = tmp_path / "bundle"
    export_evidence_bundle(source, "default", bundle)

    target = _db(tmp_path / "target.db")
    _seed(target, content="old value")
    try:
        import_evidence_bundle(target, bundle, replace=True)
    except ValueError as exc:
        assert "rollback_dir" in str(exc)
    else:
        raise AssertionError("replace without rollback evidence must fail closed")

    rollback = tmp_path / "rollback"
    import_evidence_bundle(target, bundle, replace=True, rollback_dir=rollback)
    assert verify_evidence_bundle(rollback).valid is True
    old = [json.loads(line) for line in (rollback / "facts.jsonl").read_text().splitlines()]
    assert old[0]["content"] == "old value"

    # Rollback is executable, not merely an exported file: restore the preimage.
    import_evidence_bundle(
        target,
        rollback,
        replace=True,
        rollback_dir=tmp_path / "pre-rollback-new-state",
    )
    restored = target.execute(
        "SELECT content FROM atomic_facts WHERE fact_id='f1'"
    )
    assert restored[0]["content"] == "old value"
    restored_summary = target.execute(
        "SELECT knowledge_summary FROM entity_profiles WHERE profile_entry_id='ep1'"
    )
    assert restored_summary[0]["knowledge_summary"] == "Entity summary: old value"


def test_legacy_paraphrase_is_unresolved_not_given_a_fabricated_span(
    tmp_path: Path,
) -> None:
    from superlocalmemory.core.evidence_bundle import export_evidence_bundle

    db = _db(tmp_path / "legacy.db")
    _seed(db, content="Paraphrased derived fact")
    db.execute(
        "UPDATE ingestion_operations SET raw_content='Original wording only' "
        "WHERE operation_id='op1'"
    )
    manifest = export_evidence_bundle(db, "default", tmp_path / "legacy-bundle")

    assert manifest["source_spans"] == {}
    assert manifest["lineage_coverage"]["fact"]["unresolved"] == 1
    assert manifest["unresolved_source_links"] == [
        "fact f1 has no exact span in operation op1"
    ]
