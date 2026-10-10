"""Model-decision candidates (c1/c1f/c2/c2f/c3/c3f), worker media_queries, _layer_norm."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import candidates
import embedders
import ranking
import run_eval
import worker

N_TEXT, DIM = 8, 12
NQ = 3


def _corpus() -> list[dict]:
    text = [{"doc_id": f"t:{i}", "kind": "text", "text": f"turn {i}"} for i in range(N_TEXT)]
    media = [{"doc_id": "img:a", "kind": "image", "path": "media/a.png"},
             {"doc_id": "pdf:b#p1", "kind": "pdf_page", "path": "media/b.png"}]
    return text + media


def _queries() -> list[dict]:
    return [{"id": f"q{i}", "text": f"query {i}"} for i in range(NQ)]


def _eye(rows: list[int]) -> np.ndarray:
    return np.eye(DIM, dtype=np.float32)[rows]


def _query_vecs() -> np.ndarray:
    q = np.zeros((NQ, DIM), np.float32)
    for i in range(N_TEXT):  # q0 prefers t:0, then t:1, ... and has zero text sim to media
        q[0, i] = 1.0 - 0.1 * i
    q[1, 3] = q[2, 5] = 1.0
    return q


def _stage_vectors(kinds: dict) -> dict:
    """Deterministic vectors: media items sit on axes 10/11 of the image space."""
    vecs = {"queries": _query_vecs(), "docs": _eye(list(range(N_TEXT)) + [8, 9]),
            "images": _eye([10, 11])}
    mq = np.zeros((NQ, DIM), np.float32)
    mq[0, 10] = mq[1, 11] = mq[2, 10] = 1.0
    vecs["media_queries"] = mq
    return {k: vecs[k] for k in kinds}


class FakeRt:
    """Stand-in for run_eval: spawn writes fake vectors; the rest is the real thing."""

    def __init__(self, tmp: Path):
        self.tmp, self.spawns, self.jobs, self.cos_calls = tmp, 0, [], []
        self.eg2_calls = 0
        self.doc_text, self.summarise = run_eval.doc_text, lambda t, p: {"peak": p}

    def spawn(self, job: dict, python: str):
        self.spawns += 1
        self.jobs.append((job, python))
        out = self.tmp / f"out{self.spawns}"
        out.mkdir()
        stage = job["stages"][0]
        kinds = {k for k in ("queries", "docs", "images", "media_queries") if k in stage}
        for k, v in _stage_vectors(kinds).items():
            np.save(out / f"0_{k}.npy", v)
        return {"stages": []}, 100.0, out

    def cosine_tops(self, qv, dv, ids):
        self.cos_calls.append((qv, dv, ids))
        return run_eval.cosine_tops(qv, dv, ids)

    def eg2_full_media(self, queries, paths, python):
        self.eg2_calls += 1
        v = _stage_vectors({"queries", "images"})
        return v["queries"], v["images"], {"peak": 1.0}, False


@pytest.fixture(autouse=True)
def clear_cache():
    candidates._CACHE.clear()
    yield
    candidates._CACHE.clear()


@pytest.fixture
def rt(tmp_path):
    return FakeRt(tmp_path)


PY = {"slm": "py-slm", "eg2": "py-eg2"}


def _run(system, rt, dev_qrels=None):
    return candidates.run(system, _corpus(), _queries(), PY, Path("/ds"), dev_qrels or {}, rt)


def test_merged_c1_ranks_text_plus_media_only(rt):
    tops, abst, timings = _run("c1", rt)
    want = {f"t:{i}" for i in range(N_TEXT)} | {"img:a", "pdf:b#p1"}
    assert abst is None and len(tops) == NQ
    assert all({d for d, _ in t} == want and len(t) == len(want) for t in tops)
    qv, dv, ids = rt.cos_calls[0]
    assert ids == [f"t:{i}" for i in range(N_TEXT)] + ["img:a", "pdf:b#p1"]
    assert dv.shape == (N_TEXT + 2, DIM)
    assert timings["vectors_reused"] is False
    assert list(timings["processes"]) == ["nomic text + nomic vision loadout"]


def test_merged_c1_media_competes_on_image_vectors(rt):
    tops, _, _ = _run("c1", rt)
    assert tops[0][0][0] == "t:0"
    # q1 text vec hits t:3; media rows are image vectors, not OCR text vectors
    assert tops[1][0][0] == "t:3"
    assert dict(tops[0])["img:a"] == 0.0


def test_fused_c1f_tunes_media_weight_on_dev(rt):
    dev = {"q0": {"img:a": 1}}
    tops, abst, timings = _run("c1f", rt, dev)
    fus = timings["fusion"]
    table = fus["dev_recall_by_weight"]
    assert table["0.0"] == 0.0 and fus["media_weight"] > 0
    assert table[str(fus["media_weight"])] == 1.0
    assert fus["media_weight"] in ranking.MEDIA_WEIGHTS
    assert len(abst) == NQ == len(tops)
    assert "img:a" in [d for d, _ in tops[0]][:5]


def test_fused_text_channel_covers_all_items_media_channel_only_media(rt):
    _run("c1f", rt, {"q0": {"img:a": 1}})
    (_, dv_t, ids_t), (mq, dv_m, ids_m) = rt.cos_calls[:2]
    assert len(ids_t) == N_TEXT + 2 and dv_t.shape[0] == N_TEXT + 2
    assert ids_m == ["img:a", "pdf:b#p1"]
    assert np.array_equal(mq, _stage_vectors({"media_queries"})["media_queries"])


def test_fused_without_dev_signal_prefers_text_channel(rt):
    _, _, timings = _run("c1f", rt, {})
    assert timings["fusion"]["media_weight"] == ranking.MEDIA_WEIGHTS[0]


def test_c1_then_c1f_spawns_once(rt):
    _, _, t1 = _run("c1", rt)
    _, _, t2 = _run("c1f", rt)
    assert rt.spawns == 1
    assert t1["vectors_reused"] is False and t2["vectors_reused"] is True
    job, python = rt.jobs[0]
    assert python == "py-slm" and job["stages"][0]["embedder"] == "nomic_c1"
    assert job["stages"][0]["media_queries"] == [q["text"] for q in _queries()]


def test_c3_uses_same_vectors_for_q_and_mq(rt):
    _run("c3f", rt, {"q0": {"img:a": 1}})
    stage = rt.jobs[0][0]["stages"][0]
    assert stage["embedder"] == "qwen3vl" and "media_queries" not in stage
    assert rt.jobs[0][1] == "py-eg2"
    (q_text, _, _), (q_media, _, _) = rt.cos_calls[:2]
    assert np.array_equal(q_text, q_media)
    assert np.array_equal(q_text, _query_vecs())


def test_c2_uses_text_loadout_and_full_media(rt):
    _, _, t = _run("c2f", rt, {"q0": {"img:a": 1}})
    assert rt.spawns == 1 and rt.eg2_calls == 1
    assert rt.jobs[0][0]["stages"][0]["embedder"] == "eg2_text"
    assert set(t["processes"]) == {"eg2_text loadout", "eg2_full loadout"}
    _, _, merged = _run("c2", rt)
    assert merged["vectors_reused"] is False  # fake eg2_full_media reports not-reused


def test_unknown_candidate_raises(rt):
    with pytest.raises(RuntimeError, match="unknown candidate"):
        _run("c9", rt)
    with pytest.raises(RuntimeError, match="unknown candidate"):
        _run("c9f", rt)


def test_systems_and_labels_agree():
    assert set(candidates.SYSTEMS) == set(candidates.LABELS)


class _NoMq:
    name = "nomq"

    def embed_queries(self, t):
        return np.ones((len(t), 2), np.float32)

    embed_docs = embed_images = embed_queries


def test_worker_media_queries_requires_embedder_support(tmp_path):
    with pytest.raises(SystemExit, match="cannot embed media_queries"):
        worker.run_stage(_NoMq(), {"media_queries": ["a"]}, 0, tmp_path)


def test_worker_media_queries_written_when_supported(tmp_path):
    class WithMq(_NoMq):
        def embed_media_queries(self, t):
            return np.full((len(t), 2), 3.0, np.float32)

    info = worker.run_stage(WithMq(), {"media_queries": ["a", "b"], "queries": ["a"]}, 0, tmp_path)
    assert np.load(tmp_path / "0_media_queries.npy").shape == (2, 2)
    assert info["media_queries"]["n"] == 2 and (tmp_path / "0_queries.npy").exists()


def test_worker_skips_empty_media_queries_for_unsupporting_embedder(tmp_path):
    info = worker.run_stage(_NoMq(), {"queries": ["a"]}, 0, tmp_path)
    assert "media_queries" not in info


def test_layer_norm_rows_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    out = embedders._layer_norm(rng.normal(5, 3, size=(6, 32)))
    assert np.allclose(out.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(out.var(axis=1), 1, atol=1e-3)
    assert out.dtype == np.float32


def test_layer_norm_constant_row_is_finite_zero():
    out = embedders._layer_norm(np.full((1, 8), 4.0))
    assert np.all(np.isfinite(out)) and np.allclose(out, 0)


def test_registry_has_candidate_embedders():
    assert embedders.REGISTRY["nomic_c1"] is embedders.NomicMultimodal
    assert "qwen3vl" in embedders.REGISTRY
    assert hasattr(embedders.NomicMultimodal, "embed_media_queries")
