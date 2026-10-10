"""The real-mode loader, with a fake SentenceTransformer in place of the library."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "runtimes" / "multimodal_worker.py"


class FakeST:
    calls: list = []

    def __init__(self, model, **kwargs):
        FakeST.calls.append((model, kwargs))
        self.prompts = {"SearchQuery": "q: ", "Document": "d: "}
        self.encoded: list = []

    def get_sentence_embedding_dimension(self):
        return 768

    def encode(self, items, **kw):
        self.encoded.append((items, kw))
        return [[1.0] + [0.0] * 767 for _ in items]


@pytest.fixture
def worker(monkeypatch):
    FakeST.calls = []
    torch = types.SimpleNamespace(float32="F32", backends=types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.delenv("SLM_MEDIA_WORKER_FAKE", raising=False)
    spec = importlib.util.spec_from_file_location("mm_worker_under_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REV = "914f7f89142e33e77833254d9c9b90c3cef7303b"


def test_full_loadout_kwargs(worker, tmp_path):
    reply = worker.handle({"cmd": "load", "model": "google/embeddinggemma-2", "revision": REV,
                           "hf_home": str(tmp_path), "device": "cpu"})
    assert reply == {"ok": True, "dim": 768}
    model, kw = FakeST.calls[-1]
    assert model == "google/embeddinggemma-2" and kw["revision"] == REV and kw["device"] == "cpu"
    assert kw["model_kwargs"] == {"attn_implementation": "sdpa", "dtype": "F32"}
    assert kw["config_kwargs"] == {"audio_config": None}  # the sound tower is never loaded


def test_text_only_loadout_drops_the_picture_tower_too(worker, tmp_path):
    worker.handle({"cmd": "load", "model": "google/embeddinggemma-2", "revision": REV,
                   "hf_home": str(tmp_path), "device": "cpu", "role": "text"})
    assert FakeST.calls[-1][1]["config_kwargs"] == {"vision_config": None, "audio_config": None}
    reply = worker.handle({"cmd": "embed_image", "paths": [str(tmp_path)]})
    assert reply["ok"] is False


def test_prompts_for_queries_and_documents(worker, tmp_path):
    worker.handle({"cmd": "load", "model": "m", "revision": "", "hf_home": str(tmp_path), "device": "cpu"})
    worker.handle({"cmd": "embed_text", "texts": ["a"], "prompt": "SearchQuery"})
    worker.handle({"cmd": "embed_text", "texts": ["a"], "prompt": "Document"})
    enc = worker._STATE["model"].encoded
    assert [kw["prompt_name"] for _, kw in enc] == ["SearchQuery", "Document"]


def test_images_over_the_pixel_cap_are_shrunk(worker, tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    big = tmp_path / "big.png"
    PIL.new("RGB", (2000, 1500), "red").save(big)
    worker.handle({"cmd": "load", "model": "m", "revision": "", "hf_home": str(tmp_path), "device": "cpu",
                   "max_pixels": 262144})
    reply = worker.handle({"cmd": "embed_image", "paths": [str(big)]})
    assert reply["ok"]
    sent = worker._STATE["model"].encoded[-1][0][0]
    assert sent.size[0] * sent.size[1] <= 262144
    assert abs(sent.size[0] / sent.size[1] - 4 / 3) < 0.02


def test_shrink_keeps_aspect_and_leaves_small_pictures_alone(worker):
    class Pic:
        def __init__(self, size):
            self.size = size

        def resize(self, size):
            return Pic(size)

    assert worker._shrunk(Pic((2000, 1500)), 262144).size[0] * worker._shrunk(Pic((2000, 1500)), 262144).size[1] <= 262144
    small = Pic((100, 100))
    assert worker._shrunk(small, 262144) is small and worker._shrunk(Pic((4000, 4000)), 0).size == (4000, 4000)
