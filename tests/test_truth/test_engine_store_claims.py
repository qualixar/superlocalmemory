# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Python API states a slot and a source file the way the other doors do (D-36, D-38)."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.truth import leases, slot_claims


def _meta(engine, fact_id: str) -> dict:
    row = engine._db.execute("SELECT m.metadata_json FROM atomic_facts f JOIN memories m "
                             "ON m.memory_id = f.memory_id WHERE f.fact_id = ?", (fact_id,))[0]
    return json.loads(row["metadata_json"] or "{}")


def test_store_records_the_slot_claim_and_the_lease(engine_with_mock_deps, tmp_path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    ids = engine_with_mock_deps.store("The daemon listens on port 8765", slot="SLM/daemon port",
                                      slot_value="8765", source_file=str(src))
    assert ids
    meta = _meta(engine_with_mock_deps, ids[0])
    assert meta[slot_claims.METADATA_KEY]["value"] == "8765"
    assert meta[leases.METADATA_KEY]["path"] == str(src.resolve())


def test_a_caller_cannot_forge_the_reserved_keys(engine_with_mock_deps) -> None:
    ids = engine_with_mock_deps.store("Plain memory", metadata={
        slot_claims.METADATA_KEY: {"spec": "x/y", "value": "z"},
        leases.METADATA_KEY: {"path": "/etc/hosts"}})
    meta = _meta(engine_with_mock_deps, ids[0])
    assert slot_claims.METADATA_KEY not in meta and leases.METADATA_KEY not in meta


@pytest.mark.parametrize("kwargs", [{"slot": "no-relation-here"}, {"source_file": "relative.txt"}])
def test_a_bad_claim_is_refused_before_anything_is_stored(engine_with_mock_deps, kwargs) -> None:
    before = engine_with_mock_deps._db.execute("SELECT COUNT(*) AS n FROM memories")[0]["n"]
    with pytest.raises(ValueError):
        engine_with_mock_deps.store("Should not be saved", **kwargs)
    after = engine_with_mock_deps._db.execute("SELECT COUNT(*) AS n FROM memories")[0]["n"]
    assert after == before
