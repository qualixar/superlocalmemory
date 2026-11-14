# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #322: merge an evidence bundle into a profile that already has memories."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.core.evidence_bundle import (
    export_evidence_bundle,
    verify_evidence_bundle,
)
from superlocalmemory.core.evidence_merge import (
    MergeConflictError,
    merge_evidence_bundle,
)
from tests.test_core.test_evidence_bundle import _db, _seed


def _own_memory(db, *, fact_id="f9", content="Beta runs on Postgres", profile="default",
                memory_id=None):
    memory_id = memory_id or f"m-{fact_id}"
    db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, speaker, role) "
        "VALUES (?,?,?,'s9','user','user')",
        (memory_id, profile, content),
    )
    db.execute(
        "INSERT INTO atomic_facts (fact_id,memory_id,profile_id,content,fact_type,confidence,"
        "source_turn_ids_json) VALUES (?,?,?,?,'semantic',0.8,'[]')",
        (fact_id, memory_id, profile, content),
    )


def _bundle(tmp_path: Path, *, content: str = "Alpha uses SQLite") -> Path:
    source = _db(tmp_path / "source.db")
    _seed(source, content=content)
    bundle = tmp_path / "bundle"
    export_evidence_bundle(source, "default", bundle)
    return bundle


def _facts(db, profile="default") -> dict[str, str]:
    rows = db.execute(
        "SELECT fact_id, content FROM atomic_facts WHERE profile_id=? ORDER BY fact_id",
        (profile,),
    )
    return {row["fact_id"]: row["content"] for row in rows}


def test_merge_adds_the_bundle_and_keeps_what_the_profile_had(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)

    report = merge_evidence_bundle(
        target, bundle, target_profile_id="default",
        execute=True, rollback_dir=tmp_path / "rollback",
    )

    assert report.executed is True
    assert _facts(target) == {"f1": "Alpha uses SQLite", "f9": "Beta runs on Postgres"}
    assert report.added["atomic_facts"] == 1
    assert report.added["memories"] == 1
    assert report.added["canonical_entities"] == 1
    assert report.kept["atomic_facts"] == 0
    # The full-text table follows through its triggers.
    fts = {row["fact_id"] for row in target.execute("SELECT fact_id FROM atomic_facts_fts")}
    assert {"f1", "f9"} <= fts


def test_preview_writes_nothing_and_predicts_the_real_counts(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)

    preview = merge_evidence_bundle(target, bundle, target_profile_id="default")

    assert preview.executed is False
    assert _facts(target) == {"f9": "Beta runs on Postgres"}
    assert not target.execute("SELECT 1 FROM bm25_tokens WHERE fact_id='f1'")
    real = merge_evidence_bundle(
        target, bundle, target_profile_id="default",
        execute=True, rollback_dir=tmp_path / "rollback",
    )
    assert real.added == preview.added
    assert real.kept == preview.kept
    assert real.keyword_rows == preview.keyword_rows


def test_merging_the_same_bundle_twice_changes_nothing_the_second_time(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)
    merge_evidence_bundle(target, bundle, target_profile_id="default",
                          execute=True, rollback_dir=tmp_path / "rb1")
    first = _facts(target)

    again = merge_evidence_bundle(target, bundle, target_profile_id="default",
                                  execute=True, rollback_dir=tmp_path / "rb2")

    assert _facts(target) == first
    assert sum(again.added.values()) == 0
    assert again.kept["atomic_facts"] == 1
    assert again.keyword_rows == 0


def test_an_existing_row_is_never_overwritten(tmp_path):
    bundle = _bundle(tmp_path, content="Alpha uses SQLite")
    target = _db(tmp_path / "target.db")
    _own_memory(target, fact_id="f1", content="Alpha moved to DuckDB", memory_id="m1")

    report = merge_evidence_bundle(target, bundle, target_profile_id="default",
                                   execute=True, rollback_dir=tmp_path / "rb")

    assert _facts(target)["f1"] == "Alpha moved to DuckDB"
    assert report.kept["atomic_facts"] == 1
    assert report.added["atomic_facts"] == 0


def test_an_id_owned_by_another_profile_stops_the_merge_before_writing(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    target.execute("INSERT INTO profiles (profile_id, name) VALUES ('work','work')")
    _own_memory(target, fact_id="f1", content="someone else's", profile="work")
    _own_memory(target, fact_id="f9")

    with pytest.raises(MergeConflictError, match="atomic_facts.f1"):
        merge_evidence_bundle(target, bundle, target_profile_id="default",
                              execute=True, rollback_dir=tmp_path / "rb")

    assert _facts(target) == {"f9": "Beta runs on Postgres"}
    assert _facts(target, "work") == {"f1": "someone else's"}
    assert not (tmp_path / "rb").exists()  # refused before the rollback export


def test_a_real_merge_needs_a_rollback_bundle_and_writes_a_valid_one(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)

    with pytest.raises(ValueError, match="rollback_dir"):
        merge_evidence_bundle(target, bundle, target_profile_id="default", execute=True)
    assert _facts(target) == {"f9": "Beta runs on Postgres"}

    merge_evidence_bundle(target, bundle, target_profile_id="default",
                          execute=True, rollback_dir=tmp_path / "rb")
    rollback = verify_evidence_bundle(tmp_path / "rb")
    assert rollback.valid is True
    saved = [json.loads(line) for line in
             (tmp_path / "rb/facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["fact_id"] for row in saved] == ["f9"]  # the profile before the merge


def test_merge_refuses_a_missing_profile_and_an_invalid_bundle(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")

    with pytest.raises(ValueError, match="does not exist"):
        merge_evidence_bundle(target, bundle, target_profile_id="nobody")

    with (bundle / "facts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"fact_id":"forged"}\n')
    with pytest.raises(ValueError, match="invalid evidence bundle"):
        merge_evidence_bundle(target, bundle, target_profile_id="default")


def test_keyword_rows_match_the_live_index_and_skip_wordless_facts(tmp_path):
    from superlocalmemory.retrieval.bm25_channel import tokenize

    bundle = _bundle(tmp_path, content="The engine uses SQLite")
    target = _db(tmp_path / "target.db")
    _own_memory(target)

    merge_evidence_bundle(target, bundle, target_profile_id="default",
                          execute=True, rollback_dir=tmp_path / "rb")

    stored = target.execute("SELECT tokens FROM bm25_tokens WHERE fact_id='f1'")
    assert json.loads(stored[0]["tokens"]) == tokenize("The engine uses SQLite")
    assert "the" not in json.loads(stored[0]["tokens"])

    wordless = _bundle(tmp_path / "w", content="-" * 99)
    other = _db(tmp_path / "other.db")
    _own_memory(other)
    report = merge_evidence_bundle(other, wordless, target_profile_id="default",
                                   execute=True, rollback_dir=tmp_path / "rb-w")
    assert report.added["atomic_facts"] == 1
    assert report.keyword_rows == 0


def test_an_operation_already_recorded_under_its_idempotency_key_is_kept(tmp_path):
    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)
    target.execute(
        "INSERT INTO ingestion_operations (operation_id,profile_id,source_type,"
        "idempotency_key,source_hash,raw_content,state) "
        "VALUES ('op-local','default','cli','idem1','sha','x','complete')",
    )

    report = merge_evidence_bundle(target, bundle, target_profile_id="default",
                                   execute=True, rollback_dir=tmp_path / "rb")

    assert report.kept["ingestion_operations"] == 1
    assert report.added["ingestion_operations"] == 0
    assert report.added["atomic_facts"] == 1


def test_cli_merge_previews_by_default_and_rejects_replace(tmp_path, monkeypatch, capsys):
    from superlocalmemory.cli import evidence_cmd

    bundle = _bundle(tmp_path)
    target = _db(tmp_path / "target.db")
    _own_memory(target)
    monkeypatch.setattr(evidence_cmd, "_database", lambda: target)

    args = Namespace(evidence_command="import", bundle=str(bundle), profile="default",
                     merge=True, replace=False, execute=False, rollback_dir=None, json=True)
    evidence_cmd.cmd_evidence(args)
    shown = json.loads(capsys.readouterr().out.strip())
    assert shown["mode"].startswith("merge preview")
    assert shown["added"]["atomic_facts"] == 1
    assert _facts(target) == {"f9": "Beta runs on Postgres"}

    with pytest.raises(SystemExit, match="not both"):
        evidence_cmd.cmd_evidence(Namespace(**{**vars(args), "replace": True}))

    with pytest.raises(SystemExit, match="merge refused: .*rollback_dir"):
        evidence_cmd.cmd_evidence(Namespace(**{**vars(args), "execute": True}))


def test_import_help_lists_merge():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", "evidence", "import", "--help"],
        text=True, capture_output=True, timeout=30, check=False,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "--merge" in result.stdout


def test_rebuild_writes_the_rows_the_live_index_would_write(tmp_path):
    from superlocalmemory.core.evidence_bundle import (
        import_evidence_bundle,
        rebuild_derived_state,
        verify_rebuild_equivalence,
    )
    from superlocalmemory.retrieval.bm25_channel import tokenize

    bundle = _bundle(tmp_path, content="The engine uses SQLite")
    target = _db(tmp_path / "target.db")
    import_evidence_bundle(target, bundle, replace=True, rollback_dir=tmp_path / "rb")
    _own_memory(target, fact_id="f-dash", content="-" * 99)

    result = rebuild_derived_state(target, "default")

    rows = {row["fact_id"]: json.loads(row["tokens"])
            for row in target.execute("SELECT fact_id, tokens FROM bm25_tokens")}
    assert rows == {"f1": tokenize("The engine uses SQLite")}  # no stopwords, no wordless row
    assert result["bm25_rows"] == 1
    assert "bm25_tokens" not in verify_rebuild_equivalence(
        target, bundle, profile_id="default").get("mismatches", [])


def test_export_of_an_edited_fact_stays_verifiable(tmp_path):
    """A fact edited after it was saved keeps "exact" lineage pointing at its old
    text. Export used to write that stale span, so verify refused the bundle
    (24 such facts on a real 43k-fact store)."""
    source = _db(tmp_path / "source.db")
    _seed(source, content="Alpha uses SQLite")
    source.execute(
        "INSERT INTO derivation_lineage (lineage_id,profile_id,object_type,object_id,"
        "operation_id,source_status,source_start,source_end) "
        "VALUES ('l1','default','fact','f1','op1','exact',0,17)",
    )
    source.execute("UPDATE atomic_facts SET content='Alpha moved to DuckDB' WHERE fact_id='f1'")

    manifest = export_evidence_bundle(source, "default", tmp_path / "bundle")
    report = verify_evidence_bundle(tmp_path / "bundle")

    assert report.valid is True, report.errors
    assert any("f1" in item for item in manifest["unresolved_source_links"])
