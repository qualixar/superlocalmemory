"""The text evidence floor follows the live text model, and falls back to the configured one."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.text_floor import text_semantic_floor
from superlocalmemory.runtimes import media_models


def _with_profile(monkeypatch, model: str, text_floor: float | None) -> None:
    profile = media_models.ModelProfile(model, "", 768, 1600, 0.3, 0, 1500, text_min_semantic=text_floor)
    monkeypatch.setattr(media_models, "MODEL_PROFILES", MappingProxyType({model: profile}))


def test_a_model_with_its_own_floor_uses_it(monkeypatch):
    _with_profile(monkeypatch, "acme/text-model", 0.41)
    embedder = SimpleNamespace(model_name="acme/text-model")
    assert text_semantic_floor(embedder, 0.60) == 0.41


def test_every_other_embedder_keeps_the_configured_floor(monkeypatch):
    _with_profile(monkeypatch, "acme/text-model", 0.41)
    assert text_semantic_floor(None, 0.60) == 0.60
    assert text_semantic_floor(SimpleNamespace(), 0.60) == 0.60            # no model_name (EmbeddingService)
    assert text_semantic_floor(SimpleNamespace(model_name="nomic-ai/nomic-embed-text-v1.5"), 0.55) == 0.55
    assert text_semantic_floor(SimpleNamespace(model_name=object()), 0.60) == 0.60  # a mock's attribute
    _with_profile(monkeypatch, "acme/no-floor", None)
    assert text_semantic_floor(SimpleNamespace(model_name="acme/no-floor"), 0.60) == 0.60


def test_the_retrieval_engine_reads_it_from_its_embedder(monkeypatch):
    _with_profile(monkeypatch, "acme/text-model", 0.41)
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine._config = SimpleNamespace(min_semantic_evidence=0.60)
    engine._embedder = SimpleNamespace(model_name="acme/text-model")
    assert engine._text_floor() == 0.41
    engine._embedder = SimpleNamespace()
    assert engine._text_floor() == 0.60
