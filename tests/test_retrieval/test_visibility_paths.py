"""A hidden fact never gets in through any later path, and never reaches the judge's input."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from superlocalmemory.retrieval import project_search, visibility
from superlocalmemory.retrieval.fusion import FusionResult
from superlocalmemory.storage.models import Mode

from .test_evidence_floor import _build_engine, _make_fact

pytestmark = pytest.mark.usefixtures("closes_retrieval_engines")

HIDDEN = "hid"
FACTS = [_make_fact("t1", "the red bicycle"), _make_fact("t2", "a blue door"), _make_fact(HIDDEN, "secret")]
HIDE = visibility.VisibilityContext(hidden_fact_ids=frozenset({HIDDEN}))


class Seen:
    """What the later stages were handed: the pool, and what the result builder got."""

    def __init__(self, eng):
        self.loaded: set[str] = set()
        self.built: set[str] = set()
        load, build = eng._load_facts, eng._build_results

        def spy_load(*a, **k):
            out = load(*a, **k)
            self.loaded |= set(out)
            return out

        def spy_build(final_top, facts, *a, **k):
            self.built |= {fr.fact_id for fr in final_top} | set(facts)
            return build(final_top, facts, *a, **k)

        eng._load_facts, eng._build_results = spy_load, spy_build


def _engine(path, limit_semantic=True):
    sem = [("t1", 0.9), ("t2", 0.7)]
    eng = _build_engine(facts=list(FACTS), semantic_results=sem, bm25_results=[("t1", 2.0)])
    if path == "profile":
        eng._profile_channel = MagicMock()
        eng._profile_channel.search.return_value = [(HIDDEN, 0.95)]
    elif path == "bridge":
        eng._bridge = MagicMock()
        eng._bridge.discover.return_value = [(HIDDEN, 0.95)]
    elif path == "scene":
        scene = SimpleNamespace(fact_ids=["t1", HIDDEN])
        eng._db.get_scenes_for_facts_batch.return_value = {"t1": [scene]}
    elif path == "entity_boost":
        eng._entity.score_candidates.return_value = {HIDDEN: 1.0}
        eng._semantic.search.return_value = sem + [(HIDDEN, 0.8)]
    elif path == "diversity":
        eng._semantic.search.return_value = sem + [(HIDDEN, 0.65)]
        eng._bm25.search.return_value = [("t1", 2.0), (HIDDEN, 9.0)]
    return eng


def _recall(eng, path, **kw):
    facets = None
    if path == "supplement":
        facets = SimpleNamespace(project="p", kind=None, tags=None, agent=None, about=None)
    return eng.recall("red bicycle", "default", Mode.A, limit=kw.pop("limit", 10), facets=facets, **kw)


PATHS = ["profile", "supplement", "bridge", "scene", "entity_boost", "diversity"]


@pytest.fixture(autouse=True)
def _supplement(monkeypatch):
    def inject(engine, ch_results, **kw):
        return {**ch_results, "bm25": list(ch_results.get("bm25", [])) + [(HIDDEN, 3.0)]}

    monkeypatch.setattr(project_search, "supplement", inject)


@pytest.mark.parametrize("path", PATHS)
def test_each_path_really_carries_the_fact_when_nothing_is_hidden(path):
    eng = _engine(path)
    seen = Seen(eng)
    _recall(eng, path)
    assert HIDDEN in seen.loaded, path  # the evidence floor may still drop it later


@pytest.mark.parametrize("path", PATHS)
def test_a_hidden_fact_never_appears_and_never_reaches_later_stages(path):
    eng = _engine(path)
    seen = Seen(eng)
    with visibility.use(HIDE):
        resp = _recall(eng, path)
    assert HIDDEN not in {r.fact.fact_id for r in resp.results}
    assert HIDDEN not in seen.loaded and HIDDEN not in seen.built
    assert {"t1"} <= {r.fact.fact_id for r in resp.results}


def test_a_hidden_promotion_in_the_final_cut_is_dropped(monkeypatch):
    eng = _engine("none")
    seen = Seen(eng)
    promote = eng._enforce_channel_diversity
    monkeypatch.setattr(type(eng), "_enforce_channel_diversity",
                        staticmethod(lambda top, *a, **k: list(promote(top, *a, **k))
                                     + [FusionResult(HIDDEN, 0.4, {}, {"semantic": 0.9})]))
    with visibility.use(HIDE):
        resp = _recall(eng, "none")
    assert HIDDEN not in {r.fact.fact_id for r in resp.results}
    assert HIDDEN not in seen.built


def test_the_default_context_gives_identical_ids_and_scores():
    def snap(resp):
        return [(r.fact.fact_id, r.score, dict(r.channel_scores)) for r in resp.results]

    before = snap(_recall(_engine("none"), "none"))
    with visibility.use(visibility.VisibilityContext()):
        after = snap(_recall(_engine("none"), "none"))
    assert before == after and before


def test_the_default_context_asks_the_database_nothing_extra():
    eng = _engine("none")
    _recall(eng, "none")
    sql = [str(c) for c in eng._db.execute.call_args_list]
    assert not any("metadata_json" in s for s in sql)


# -- hide_media ---------------------------------------------------------------------

def _media_engine(tmp_path, monkeypatch):
    from superlocalmemory.retrieval.media_channel import MediaChannel

    from ._media_support import FakeClient, MemoryDb, make_store, mid, unit

    memdb = MemoryDb()
    for memory_id, fact_id, src in [
            ("m_img", "img", {"type": "media", "media_id": "a" * 32}),
            ("m_page", "page", {"type": "document", "document_id": "d1", "page": 2}),
            ("m_note", "t1", None)]:
        memdb.add(memory_id, fact_id, src)
    (tmp_path / "media.db").write_bytes(b"")
    monkeypatch.setattr("superlocalmemory.runtimes.features.media_enabled", lambda root=None: True)
    facts = [_make_fact("t1", "the red bicycle"), _make_fact("img", "a bicycle photo"),
             _make_fact("page", "a bicycle page")]
    for f, m in zip(facts, ["m_note", "m_img", "m_page"]):
        f.memory_id = m
    eng = _build_engine(facts=facts, semantic_results=[("t1", 0.9), ("img", 0.8), ("page", 0.7)],
                        bm25_results=[("t1", 2.0)])
    sql_log: list[str] = []

    def run(sql, params=()):
        sql_log.append(sql)
        return memdb.execute(sql, params) if "FROM memories" in sql or "FROM atomic_facts" in sql else []

    eng._db.db_path = tmp_path / "memory.db"
    eng._db.execute.side_effect = run
    store = make_store(tmp_path, [(mid(1), "image", "m_img", unit(0))])
    client = FakeClient(unit(0))
    eng._media_channel = MediaChannel(lambda: store, lambda: client, memdb)
    return eng, store, client, sql_log


def test_hide_media_drops_picture_and_page_facts_and_skips_the_media_channel(tmp_path, monkeypatch):
    eng, store, client, _ = _media_engine(tmp_path, monkeypatch)
    seen = Seen(eng)
    try:
        with visibility.use(visibility.VisibilityContext(hide_media=True)):
            resp = eng.recall("red bicycle", "default", Mode.A, limit=10)
    finally:
        store.close()
    assert [r.fact.fact_id for r in resp.results] == ["t1"]
    assert not ({"img", "page"} & (seen.loaded | seen.built))
    assert client.calls == [] and "media" not in resp.channel_weights


def test_without_hide_media_the_same_recall_still_returns_them(tmp_path, monkeypatch):
    eng, store, client, _ = _media_engine(tmp_path, monkeypatch)
    try:
        resp = eng.recall("red bicycle", "default", Mode.A, limit=10)
    finally:
        store.close()
    assert {"t1", "img", "page"} <= {r.fact.fact_id for r in resp.results}
    assert client.calls
