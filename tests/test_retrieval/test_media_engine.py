"""Recall with the picture channel: untouched when off, found by what it shows when on."""

from __future__ import annotations

import pytest

from superlocalmemory.core.config import ChannelWeights, RetrievalConfig
from superlocalmemory.retrieval import channel_status as chstat
from superlocalmemory.retrieval.engine import RetrievalEngine, apply_channel_weights
from superlocalmemory.retrieval.media_channel import MediaChannel
from superlocalmemory.storage.models import Mode

from ._media_support import FakeClient, MemoryDb, blend, make_store, mid, unit
from .test_evidence_floor import _build_engine, _make_fact

pytestmark = pytest.mark.usefixtures("closes_retrieval_engines")

TEXT = [_make_fact("t1", "the red bicycle"), _make_fact("t2", "a blue door")]


def _text_engine(**kw):
    return _build_engine(facts=TEXT, semantic_results=[("t1", 0.9), ("t2", 0.7)],
                         bm25_results=[("t1", 2.0)], **kw)


def _snapshot(resp):
    return [(r.fact.fact_id, r.score, dict(r.channel_scores)) for r in resp.results]


# -- G-UPG: pictures off, nothing changes -------------------------------------------

def test_with_pictures_off_the_recall_is_exactly_what_it_was():
    plain = _text_engine(config=RetrievalConfig(disabled_channels=["media"]))
    off = _text_engine()  # media on the code path, but no media.db: not active
    a = plain.recall("red bicycle", "default", Mode.A, limit=10)
    b = off.recall("red bicycle", "default", Mode.A, limit=10)
    assert _snapshot(a) == _snapshot(b) and _snapshot(a)
    assert a.channel_weights == b.channel_weights
    assert "media" not in b.channel_weights
    assert all("media" not in cs for _, _, cs in _snapshot(b))


def test_an_inactive_channel_leaves_no_trace_in_channel_status():
    eng = _text_engine()
    status: dict[str, str] = {}
    from superlocalmemory.retrieval.strategy import QueryStrategy
    eng._run_channels("q", "default", QueryStrategy(query_type="factual", weights={}),
                      channel_status=status)
    assert "media" not in status


def test_channel_weights_as_dict_is_unchanged_and_media_defaults_to_one():
    assert "media" not in ChannelWeights().as_dict()
    assert len(ChannelWeights().as_dict()) == 6
    assert ChannelWeights().media == 1.0


def test_apply_channel_weights_adds_no_media_key_when_absent():
    from superlocalmemory.storage.models import RetrievalResult
    r = RetrievalResult(fact=TEXT[0], score=0.5, channel_scores={"semantic": 0.8})
    out = apply_channel_weights([r], {"semantic": 2.0})
    assert "media" not in out[0].channel_scores
    with_media = RetrievalResult(fact=TEXT[0], score=0.5, channel_scores={"media": 0.5})
    assert apply_channel_weights([with_media], {"media": 2.0})[0].channel_scores["media"] == 1.0


# -- active -------------------------------------------------------------------------

def _active_engine(tmp_path, rows, facts, vector, client_vector, **kw):
    memdb = MemoryDb()
    for memory_id, fact_id in facts:
        memdb.add(memory_id, fact_id)
    store = make_store(tmp_path, rows)
    eng = _build_engine(facts=[_make_fact(f, c) for f, c in kw.pop("contents")],
                        semantic_results=kw.pop("semantic", []), bm25_results=kw.pop("bm25", []),
                        **kw)
    client = FakeClient(client_vector)
    eng._media_channel = MediaChannel(lambda: store, lambda: client, memdb)
    return eng, store, client


def _recall_for_score(tmp_path, cosine):
    """A pure-visual anchor whose match with the question is exactly ``cosine``."""
    image = [cosine, (1.0 - cosine * cosine) ** 0.5, 0.0, 0.0]
    eng, store, _ = _active_engine(
        tmp_path, [(mid(1), "image", "m1", image)], [("m1", "img")], None, unit(0),
        contents=[("img", "[Image without text]")])
    try:
        return eng.recall("a photo of a harbour", "default", Mode.A, limit=10)
    finally:
        store.close()


def test_a_picture_with_no_words_is_found_by_what_it_shows(tmp_path):
    resp = _recall_for_score(tmp_path, 0.8)
    assert [r.fact.fact_id for r in resp.results] == ["img"]
    assert resp.results[0].channel_scores["media"] == pytest.approx(0.8, abs=1e-3)
    assert resp.channel_weights["media"] == 1.0


def test_a_weak_picture_match_is_floored(tmp_path):
    assert _recall_for_score(tmp_path, 0.1).results == []


def test_a_cold_worker_is_reported_and_the_text_answer_stands(tmp_path):
    eng, store, client = _active_engine(
        tmp_path, [(mid(1), "image", "m1", unit(0))], [("m1", "img")], None, None,
        contents=[("img", "[Image without text]"), ("t1", "the red bicycle")],
        semantic=[("t1", 0.9)], bm25=[("t1", 2.0)])
    status: dict[str, str] = {}
    from superlocalmemory.retrieval.strategy import QueryStrategy
    try:
        out = eng._run_channels("red bicycle", "default",
                                QueryStrategy(query_type="factual", weights={}),
                                channel_status=status)
    finally:
        store.close()
    assert status["media"] == chstat.WARMING
    assert "media" not in out and "bm25" in out
    assert client.warmups == 1


def test_ids_and_scores_match_what_recall_returned_before_pictures_existed():
    """Pinned from the release before the picture channel existed."""
    resp = _text_engine().recall("red bicycle", "default", Mode.A, limit=10)
    assert _snapshot(resp) == [
        ("t1", 0.515, {"semantic": 0.9, "bm25": 2.0}),
        ("t2", 0.5086, {"semantic": 0.7, "bm25": 0.0}),
    ]
    assert resp.channel_weights == {"semantic": 1.5, "bm25": 1.0, "entity_graph": 1.0,
                                    "temporal": 1.0, "spreading_activation": 1.0, "hopfield": 0.8}
