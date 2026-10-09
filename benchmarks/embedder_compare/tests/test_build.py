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
    assert len(a) == 30 and len(set(a)) == 30
    for name in a:
        pa, pb = tmp_path / "a" / f"{name}.png", tmp_path / "b" / f"{name}.png"
        assert hashlib.sha256(pa.read_bytes()).hexdigest() == hashlib.sha256(pb.read_bytes()).hexdigest()
        assert imagehash.phash(Image.open(pa)) == imagehash.phash(Image.open(pb))
        w, h = Image.open(pa).size
        assert (w, h) == synth_images.SIZE


def test_visual_images_have_queries_and_no_text_content():
    names = [n for n in synth_images.CONTENT if n.startswith("vis_")]
    assert len(names) == 6
    assert all(synth_images.CONTENT[n][0] == "visual" for n in names)
    golden = Path(__file__).parents[1] / "golden" / "media_queries.jsonl"
    rel = {d for line in golden.read_text().splitlines() for d in json.loads(line)["relevant"]}
    assert {f"img:syn_{n}" for n in names} <= rel


def test_font_dir_override_and_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(synth_images, "_font_dir_cache", None)
    monkeypatch.setattr(synth_images, "FONT_SEARCH", ())
    monkeypatch.setenv("SLM_BENCH_FONT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="SLM_BENCH_FONT_DIR"):
        synth_images.font_dir()
    for f in synth_images.FONT_FILES:
        (tmp_path / f).write_bytes(b"")
    assert synth_images.font_dir() == tmp_path
    monkeypatch.setattr(synth_images, "_font_dir_cache", None)


def test_build_refuses_changed_test_set_without_refreeze(tmp_path):
    data = json.loads(FIXTURE.read_text())
    quotas = {"text_single_hop": 3, "text_multi_hop": 2, "entity": 2, "temporal": 2, "unanswerable": 2}
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps(bd.select_locomo(data, quotas, n_convs=2)))
    frozen, ds = tmp_path / "frozen.sha256", tmp_path / "ds"
    with pytest.raises(ValueError, match="refreeze"):
        bd.build(FIXTURE, sel, ds, media=False, frozen_path=frozen)
    assert not ds.exists()
    bd.build(FIXTURE, sel, ds, media=False, frozen_path=frozen, refreeze=True)
    first = frozen.read_text()
    bd.build(FIXTURE, sel, ds, media=False, frozen_path=frozen)  # same hash: fine
    frozen.write_text("0" * 64 + "\n")
    before = (ds / "manifest.lock").read_text()
    with pytest.raises(ValueError, match="frozen"):
        bd.build(FIXTURE, sel, ds, media=False, frozen_path=frozen)
    assert (ds / "manifest.lock").read_text() == before and first != frozen.read_text()


def test_synth_pdfs_generate(tmp_path):
    import synth_pdfs

    names = synth_pdfs.generate(tmp_path, Path(__file__).resolve().parents[3])
    assert names and all((tmp_path / f"{n}.pdf").stat().st_size > 0 for n in names)
