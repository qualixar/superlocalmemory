"""Consolidation steps 1 and 3 survive a canonical_entities_json that holds non-UTF-8 bytes."""

from __future__ import annotations

from superlocalmemory.core.config import CCQConfig
from superlocalmemory.encoding.cognitive_consolidator import (
    CognitiveConsolidator,
    ConsolidationCluster,
)

BAD = b"\xff\xfe\x00not text"


class _Db:
    def __init__(self, rows):
        self._rows = rows

    def current_fact_clause(self, alias):
        return ""

    def execute(self, sql, params=()):
        return self._rows


def _consolidator(rows):
    return CognitiveConsolidator(_Db(rows), config=CCQConfig())


def test_step1_reads_a_blob_as_no_entities() -> None:
    rows = [{"fact_id": "f1", "content": "c", "canonical_entities_json": BAD},
            {"fact_id": "f2", "content": "c", "canonical_entities_json": b'["a"]'}]
    out = _consolidator(rows)._step1_identify("p")
    assert [d["canonical_entities"] for d in out] == [[], ["a"]]


def test_step3_reads_a_blob_as_no_entities() -> None:
    rows = [{"fact_id": "f1", "content": "alpha", "importance": 0.5, "confidence": 0.9,
             "canonical_entities_json": BAD}]
    cluster = ConsolidationCluster("c1", ("f1",), (), "2026-01-01", 0.1, 1)
    gist = _consolidator(rows)._step3_extract_gist(cluster, "p")
    assert gist.gist_text
