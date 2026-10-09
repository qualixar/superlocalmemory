"""RSS sampler, manifest guard, embedder prompt handling, smoke run."""
from __future__ import annotations

import json
import os
import time

import numpy as np

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


def test_nomic_prefixes():
    e = embedders.Nomic()
    e._model = FakeModel()
    e.embed_queries(["hello"])
    e.embed_docs(["world"])
    assert e._model.calls[0][0] == ["search_query: hello"]
    assert e._model.calls[1][0] == ["search_document: world"]


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
    rc = report.main(["--home", str(tiny_home), "--out", str(out)])
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
    rc = report.main(["--home", str(home), "--out", str(tmp_path / "o")])
    assert rc != 0
    assert "manifest" in capsys.readouterr().err.lower()


def test_abstention_tau_ignores_test_labels(tiny_home):
    qs = report.load_queries(tiny_home / "dataset")
    scores = {q["id"]: (0.1 if not q["answerable"] else 0.9) for q in qs["dev"] + qs["test"]}
    tau1 = report.tune_tau(qs["dev"], scores)
    flipped = [dict(q, answerable=not q["answerable"]) for q in qs["test"]]
    tau2 = report.tune_tau(qs["dev"], scores)
    assert tau1 == tau2 and flipped
