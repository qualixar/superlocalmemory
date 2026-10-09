"""RSS sampler, manifest guard, embedder prompt handling, smoke run."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

import embedders
import report
import rss
import run_eval


def test_rss_sampler_positive():
    with rss.RssSampler(os.getpid(), interval=0.02) as s:
        _ = bytearray(5_000_000)
        time.sleep(0.1)
    assert s.peak_mb > 0


class FakeModel:
    def __init__(self, prompts=None):
        self.calls = []
        self.prompts = prompts or {}

    def encode(self, x, **kw):
        self.calls.append((x, kw))
        n = len(x) if isinstance(x, list) else 1
        out = np.ones((n, 4), dtype=np.float32) * 2.0
        return out if isinstance(x, list) else out[0]


def test_nomic_ships_without_prefixes_and_prefixed_variant_adds_them():
    e = embedders.Nomic()
    e._model = FakeModel()
    e.embed_queries(["hello"])
    e.embed_docs(["world"])
    assert e._model.calls[0][0] == ["hello"]
    assert e._model.calls[1][0] == ["world"]
    p = embedders.NomicPrefixed()
    p._model = FakeModel()
    p.embed_queries(["hello"])
    p.embed_docs(["world"])
    assert p._model.calls[0][0] == ["search_query: hello"]
    assert p._model.calls[1][0] == ["search_document: world"]


def test_eg2_prompt_names_and_images():
    e = embedders.EG2Full()
    e._model = FakeModel({"SearchQuery": "q", "Document": "d"})
    e.embed_queries(["a"])
    e.embed_docs(["b"])
    v = e.embed_images(["/tmp/x.png"])
    assert e._model.calls[0][1]["prompt_name"] == "SearchQuery"
    assert e._model.calls[1][1]["prompt_name"] == "Document"
    assert e._model.calls[2][0] == {"image": "/tmp/x.png"}
    assert v.shape == (1, 4) and abs(np.linalg.norm(v[0]) - 1) < 1e-5
    t = embedders.EG2Text()
    assert t.config_kwargs == {"vision_config": None, "audio_config": None}
    assert embedders.EG2Full.config_kwargs == {"audio_config": None}


def test_eg2_missing_prompt_names_raise():
    e = embedders.EG2Full()
    e._model = FakeModel({"Document": "d"})
    with pytest.raises(KeyError, match="SearchQuery"):
        e.embed_queries(["a"])


def _run_smoke(home):
    rc = run_eval.main(["--home", str(home), "--systems", "bm25,s1"])
    assert rc == 0
    return home / "runs"


def test_smoke_bm25_end_to_end(tiny_home, tmp_path, capsys):
    runs = _run_smoke(tiny_home)
    assert (runs / "bm25.json").exists()
    status = json.loads((runs / "status.json").read_text())
    assert status["bm25"]["ran"] is True
    assert status["s1"]["ran"] is False and status["s1"]["reason"]
    out = tmp_path / "out"
    rc = report.main(["--home", str(tiny_home), "--out", str(out), "--frozen", str(tiny_home / "frozen.sha256")])
    assert rc == 0
    md = (out / "RESULTS.md").read_text()
    for section in ("## Systems", "## Recall@5", "## Unanswerable",
                    "## Latency and memory", "## One-model rule", "not run"):
        assert section in md
    assert (out / "RESULTS.json").exists()


def test_manifest_mismatch_refuses(tiny_home, tmp_path, capsys):
    import shutil

    home = tmp_path / "copy"
    shutil.copytree(tiny_home, home)
    test_file = home / "dataset" / "queries" / "test.jsonl"
    test_file.write_text(test_file.read_text() + "\n")
    rc = report.main(["--home", str(home), "--out", str(tmp_path / "o"), "--frozen", str(home / "frozen.sha256")])
    assert rc != 0
    assert "manifest" in capsys.readouterr().err.lower()


def test_frozen_hash_mismatch_refuses_even_if_lock_is_consistent(tiny_home, tmp_path, capsys):
    frozen = tmp_path / "other.sha256"
    frozen.write_text("0" * 64 + "\n")
    _run_smoke(tiny_home)
    rc = report.main(["--home", str(tiny_home), "--out", str(tmp_path / "o"), "--frozen", str(frozen)])
    assert rc == 2
    assert "frozen" in capsys.readouterr().err


def test_run_on_different_dataset_or_missing_query_refuses(tiny_home, tmp_path, capsys):
    import shutil

    home = tmp_path / "copy"
    shutil.copytree(tiny_home, home)
    run_eval.main(["--home", str(home), "--systems", "bm25"])
    status_path = home / "runs" / "status.json"
    status = json.loads(status_path.read_text())
    args = ["--home", str(home), "--out", str(tmp_path / "o"), "--frozen", str(home / "frozen.sha256")]
    status["bm25"]["dataset"] = "stale"
    status_path.write_text(json.dumps(status))
    assert report.main(args) == 2
    assert "different dataset" in capsys.readouterr().err
    run_eval.main(["--home", str(home), "--systems", "bm25"])
    run_path = home / "runs" / "bm25.json"
    run = json.loads(run_path.read_text())
    run.pop(sorted(run)[0])
    run_path.write_text(json.dumps(run))
    assert report.main(args) == 2
    assert "lacks 1 queries" in capsys.readouterr().err


def test_run_scores_are_strictly_decreasing_ranks():
    run, abst = run_eval.run_dicts(["q"], [[("a", 0.5), ("b", 0.5), ("c", 0.1)]])
    assert list(run["q"].values()) == [3.0, 2.0, 1.0]
    assert abst["q"] == 0.5


def test_abstention_tau_ignores_test_labels(tiny_home):
    qs = report.load_queries(tiny_home / "dataset")
    scores = {q["id"]: (0.1 if not q["answerable"] else 0.9) for q in qs["dev"] + qs["test"]}
    tau1 = report.abstention(qs, scores)["all"]["tau"]
    flipped = {"dev": qs["dev"], "test": [dict(q, answerable=not q["answerable"]) for q in qs["test"]]}
    shifted = {**scores, **{q["id"]: 0.5 for q in qs["test"]}}
    assert report.abstention(flipped, scores)["all"]["tau"] == tau1
    assert report.abstention(qs, shifted)["all"]["tau"] == tau1
    assert report.tune_tau(qs["dev"], scores) == tau1


def _fake_python(tmp_path):
    """An executable that runs worker.py with hash-based fake embedders."""
    import stat
    import sys

    harness = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = tmp_path / "fakepy"
    script.write_text(f"""#!{sys.executable}
import sys, zlib
sys.path.insert(0, {harness!r})
import numpy as np
import embedders, worker

def vec(text):
    v = np.zeros(64, np.float32)
    for w in text.lower().split():
        v[zlib.crc32(w.encode()) % 64] += 1.0
    return v / max(np.linalg.norm(v), 1e-9)

class Fake:
    def __init__(self): self.name = "fake"
    def load(self): pass
    def embed_queries(self, t): return np.stack([vec(x) for x in t])
    embed_docs = embed_queries
    def embed_images(self, p): return np.stack([vec(x) for x in p])

embedders.REGISTRY = {{k: Fake for k in ("nomic", "nomic_prefixed", "eg2_text", "eg2_full")}}
sys.exit(worker.main())
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_model_systems_plumbing_with_fake_embedders(tiny_home, tmp_path):
    import shutil

    home = tmp_path / "h"
    shutil.copytree(tiny_home, home)
    fake = _fake_python(tmp_path)
    rc = run_eval.main(["--home", str(home), "--systems", "s1,s1p,s2,s3",
                        "--python-slm", fake, "--python-eg2", fake])
    assert rc == 0
    status = json.loads((home / "runs" / "status.json").read_text())
    assert all(status[s]["ran"] for s in ("s1", "s1p", "s2", "s3")), status
    assert report.main(["--home", str(home), "--out", str(tmp_path / "o"),
                        "--frozen", str(home / "frozen.sha256")]) == 0
    res = json.loads((tmp_path / "o" / "RESULTS.json").read_text())
    assert res["one_model"]["verdict"] == "SAMPLE TOO SMALL TO DECIDE"
    assert "s1 -> s2" in res["paired"]
    s3 = json.loads((home / "runs" / "s3.timings.json").read_text())
    assert s3["fusion"]["media_weight"] in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    assert set(s3["processes"]) == {"nomic loadout", "eg2_full loadout (media channel)"}


def test_worker_warms_up_before_timing(tmp_path):
    import worker

    class Counting:
        name = "c"
        calls = 0

        def embed_queries(self, texts):
            Counting.calls += 1
            return np.ones((len(texts), 2), np.float32)

        embed_docs = embed_images = embed_queries

    info = worker.run_stage(Counting(), {"queries": ["a", "b", "c"]}, 0, tmp_path)
    assert Counting.calls == 4 and info["queries"]["n"] == 3


def test_eg2_full_media_vectors_cached_across_systems(monkeypatch, tmp_path):
    calls = []

    def fake_spawn(job, python):
        calls.append(job)
        np.save(tmp_path / "0_queries.npy", np.ones((1, 2), np.float32))
        np.save(tmp_path / "0_images.npy", np.ones((1, 2), np.float32))
        return {"stages": [{"load_s": 1.0}]}, 5.0, tmp_path

    monkeypatch.setattr(run_eval, "spawn", fake_spawn)
    run_eval._EG2_FULL_CACHE.clear()
    qs = [{"text": "q"}]
    assert run_eval.eg2_full_media(qs, ["/x.png"], "py")[3] is False
    assert run_eval.eg2_full_media(qs, ["/x.png"], "py")[3] is True
    assert len(calls) == 1
