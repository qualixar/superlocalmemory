"""A recall whose text vectors need the SLM service says so, instead of "warming"."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.cli.recall_text import incomplete_line
from superlocalmemory.core import daemon_text_embedder as dte
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval.query_embedding import QueryEmbedder

EG2 = "google/embeddinggemma-2"


def _embedder(monkeypatch, ping):
    monkeypatch.setattr(dte, "_owned_daemon_request",
                        lambda method, path, body, timeout: ping if path.endswith("/ping") else None)
    return dte.DaemonTextEmbedder(SimpleNamespace(model_name=EG2, dimension=768))


def test_a_down_daemon_is_the_needs_service_status_without_waiting(monkeypatch):
    emb = _embedder(monkeypatch, None)
    assert emb.needs_service is True
    vector, status = QueryEmbedder(lambda: emb).embed("what did I say", 0.0)
    assert vector is None and status == chstat.NEEDS_SERVICE
    assert chstat.is_fault(status)


def test_a_reachable_daemon_that_is_loading_is_still_warming_not_needs_service(monkeypatch):
    ping = {"ok": True, "embedder": {"available": True, "warm": False, "model": EG2, "dimension": 768}}
    emb = _embedder(monkeypatch, ping)
    assert emb.needs_service is False


def test_the_plain_recall_line_uses_the_existing_sentence():
    result = {"incomplete_channels": ["semantic"], "channel_status": {"semantic": chstat.NEEDS_SERVICE}}
    assert dte.NEEDS_SERVICE in incomplete_line(result)
    warming = {"incomplete_channels": ["semantic"], "channel_status": {"semantic": "warming"}}
    assert "still loading" in incomplete_line(warming)
