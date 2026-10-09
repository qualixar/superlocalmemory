"""Dataset building on the invented fixture; synthetic images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import build_dataset as bd
import synth_images

FIXTURE = Path(__file__).parent / "fixtures" / "fake_locomo.json"


def _lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_build_doc_ids_prefix_and_qrels(tiny_home):
    ds = tiny_home / "dataset"
    corpus = {d["doc_id"]: d for d in _lines(ds / "corpus.jsonl")}
    assert "t:fake-1:D1:1" in corpus
    assert corpus["t:fake-1:D1:1"]["text"].startswith("[8 May 2023] Mira:")
    assert corpus["t:fake-2:D2:3"]["kind"] == "text"
    queries = _lines(ds / "queries" / "dev.jsonl") + _lines(ds / "queries" / "test.jsonl")
    qrels = {}
    for split in ("dev", "test"):
        qrels.update(json.loads((ds / "qrels" / f"{split}.json").read_text()))
    assert len(queries) == 11
    by_stratum = {}
    for q in queries:
        by_stratum.setdefault(q["stratum"], []).append(q)
        assert q.get("provisional") is True
        if q["answerable"]:
            assert qrels[q["id"]] and all(d in corpus for d in qrels[q["id"]])
    for q in by_stratum["unanswerable"]:
        assert q["answerable"] is False and qrels[q["id"]] == {}


def test_qrels_equal_evidence(tiny_home):
    data = json.loads(FIXTURE.read_text())
    ds = tiny_home / "dataset"
    allq = _lines(ds / "queries" / "dev.jsonl") + _lines(ds / "queries" / "test.jsonl")
    qrels = {}
    for split in ("dev", "test"):
        qrels.update(json.loads((ds / "qrels" / f"{split}.json").read_text()))
    for q in allq:
        if not q["answerable"]:
            continue
        _, sid, idx = q["id"].split(":")
        ev = data[int(sid[-1]) - 1]["qa"][int(idx)]["evidence"]
        assert set(qrels[q["id"]]) == {f"t:{sid}:{e}" for e in ev}


def test_missing_evidence_fails_loudly(tmp_path):
    data = json.loads(FIXTURE.read_text())
    data[0]["qa"][0]["evidence"] = ["D9:9"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    sel = [{"sample_id": "fake-1", "qa_index": 0, "stratum": "temporal"}]
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(json.dumps(sel))
    with pytest.raises(ValueError, match="D9:9"):
        bd.build(bad, sel_path, tmp_path / "ds", media=False)


def test_selection_is_deterministic_and_ids_only():
    data = json.loads(FIXTURE.read_text())
    quotas = {"text_single_hop": 3, "text_multi_hop": 2, "entity": 2,
              "temporal": 2, "unanswerable": 2}
    a = bd.select_locomo(data, quotas, n_convs=2)
    assert a == bd.select_locomo(data, quotas, n_convs=2)
    assert all(set(x) == {"sample_id", "qa_index", "stratum"} for x in a)


def test_synth_images_deterministic(tmp_path):
    from PIL import Image
    import imagehash

    a = synth_images.generate(tmp_path / "a")
    b = synth_images.generate(tmp_path / "b")
    assert len(a) == 24 and len(set(a)) == 24
    for name in a:
        pa, pb = tmp_path / "a" / f"{name}.png", tmp_path / "b" / f"{name}.png"
        assert hashlib.sha256(pa.read_bytes()).hexdigest() == hashlib.sha256(pb.read_bytes()).hexdigest()
        assert imagehash.phash(Image.open(pa)) == imagehash.phash(Image.open(pb))
        w, h = Image.open(pa).size
        assert (w, h) == synth_images.SIZE
