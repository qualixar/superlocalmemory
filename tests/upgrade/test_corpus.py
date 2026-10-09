# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fast checks for the upgrade corpus, the recall verdict and the manifest schema."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_fixture as bf  # noqa: E402
import corpus  # noqa: E402
import upgrade_check as uc  # noqa: E402


def test_corpus_shape():
    assert len(corpus.MEMORIES) == 40
    assert len(corpus.QUERIES) == 12
    ids = [m.id for m in corpus.MEMORIES]
    assert len(set(ids)) == 40
    assert len({m.text for m in corpus.MEMORIES}) == 40


def test_every_memory_carries_its_reference():
    for m in corpus.MEMORIES:
        assert corpus.ref_of(m.text) == m.id


def test_expected_ids_exist_and_match_query_words():
    by_id = {m.id: m for m in corpus.MEMORIES}
    for q in corpus.QUERIES:
        assert q.expected
        for exp in q.expected:
            assert exp in by_id
            words = [w.lower() for w in q.text.split() if len(w) > 5]
            assert any(w in by_id[exp].text.lower() for w in words), q.text


def test_ref_of_handles_missing():
    assert corpus.ref_of("no reference here") is None
    assert corpus.ref_of("") is None


def test_corpus_has_no_real_looking_contacts():
    blob = " ".join(m.text for m in corpus.MEMORIES)
    assert "@" not in blob and "http" not in blob


def test_overlap_counts_shared_top5():
    assert uc.overlap(["a", "b", "c"], ["c", "b", "z"]) == 2
    assert uc.overlap([], ["a"]) == 0
    assert uc.overlap(list("abcdefg"), list("abcdexx")) == 5


def test_verdict_identical():
    base = [["a", "b"], ["c"]]
    v = uc.recall_verdict(base_a=base, base_b=base, new=base)
    assert v["verdict"] == "identical"


def test_verdict_within_baseline_noise():
    a = [["a", "b", "c"]]
    b = [["a", "b", "x"]]  # baseline disagrees with itself by one id
    new = [["a", "b", "x"]]  # new matches reference as well as the baseline pair
    v = uc.recall_verdict(base_a=a, base_b=b, new=new)
    assert v["verdict"] in {"identical", "within_noise"}
    assert v["new_overlap"] >= v["baseline_overlap"]


def test_verdict_worse_than_baseline():
    a = [["a", "b", "c"]]
    new = [["x", "y", "z"]]
    v = uc.recall_verdict(base_a=a, base_b=a, new=new)
    assert v["verdict"] == "worse"


def test_manifest_validation_accepts_good_and_rejects_bad():
    good = {
        "version": "4.1.24", "python": "3.12.1", "cli": "slm remember",
        "counts": {"memories": 40}, "sha256": {"memory.db": "0" * 64},
        "embedding_mode": "keyword-only", "built_at": "2026-10-09T00:00:00Z",
    }
    assert bf.validate_manifest(good) == []
    bad = dict(good, counts={"memories": "40"}, sha256={"memory.db": "xyz"})
    del bad["python"]
    problems = bf.validate_manifest(bad)
    assert any("python" in p for p in problems)
    assert any("counts" in p for p in problems)
    assert any("sha256" in p for p in problems)


@pytest.mark.parametrize("n", [0, 39])
def test_memory_ids_are_sequential_refs(n):
    assert corpus.MEMORIES[n].id == f"QX-{1001 + n}"


def test_committed_manifests_are_well_formed_and_hold_no_secrets():
    import json

    files = sorted((Path(__file__).resolve().parent / "manifests").glob("*.json"))
    assert files
    for f in files:
        m = json.loads(f.read_text())
        assert bf.validate_manifest(m) == [], f.name
        assert m["version"] == f.stem
        assert not m.get("errors")
        assert "/home/" not in f.read_text() and "-key" not in f.read_text()
