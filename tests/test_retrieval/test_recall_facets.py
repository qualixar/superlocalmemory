# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Narrowing a recall by project, by the agent that saved it, or by what it is about."""

from __future__ import annotations

from argparse import Namespace

import pytest

from superlocalmemory.retrieval.facets import Facets, list_facets, matching_fact_ids
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    FactType,
    MemoryRecord,
    RecallResponse,
    RetrievalResult,
)


@pytest.fixture()
def db(tmp_path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "memory.db")
    mgr.initialize(schema)
    mgr.store_entity(CanonicalEntity(entity_id="e_pallavi", profile_id="default",
                                     canonical_name="Pallavi"))
    return mgr


def _save(db, content, *, project=None, agent=None, entities=(), tags=None) -> str:
    meta = {k: v for k, v in (("project", project), ("agent_id", agent), ("tags", tags))
           if v is not None}
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content, metadata=meta))
    return db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id, content=content,
                                    fact_type=FactType.SEMANTIC,
                                    canonical_entities=list(entities)))


def test_project_filter_ignores_case_and_spaces(db) -> None:
    slm = _save(db, "Ship 4.1.19 with memory kinds.", project="SuperLocalMemory")
    other = _save(db, "Bounded loops need a gate.", project="bounded-loops")
    none = _save(db, "Untagged memory.")
    keep = matching_fact_ids(db, [slm, other, none], "default", Facets.of(project=" superlocalmemory "))
    assert keep == {slm}


def test_facets_combine_as_and(db) -> None:
    a = _save(db, "A", project="slm", agent="claude-desktop")
    b = _save(db, "B", project="slm", agent="grok")
    assert matching_fact_ids(db, [a, b], "default", Facets.of(project="slm", agent="grok")) == {b}


def test_about_finds_memories_that_mention_the_name(db) -> None:
    about = _save(db, "Pallavi reviews the release notes.", entities=("e_pallavi",))
    other = _save(db, "Nothing about people.")
    assert matching_fact_ids(db, [about, other], "default", Facets.of(about="pallavi")) == {about}
    assert matching_fact_ids(db, [about, other], "default", Facets.of(about="Nobody Known")) == set()


def test_no_facets_keeps_everything_and_blank_values_are_no_facet(db) -> None:
    ids = [_save(db, "x"), _save(db, "y", project="p")]
    assert matching_fact_ids(db, ids, "default", Facets.of()) == set(ids)
    assert Facets.of(project="  ", agent="", about=None).empty


def test_tags_filter_matches_exact_label_csv_stored(db) -> None:
    tagged = _save(db, "Ship with token optimization.", tags="token-optimization,v4.1.21")
    other = _save(db, "Nothing tagged this way.", tags="unrelated")
    none = _save(db, "No tags at all.")
    keep = matching_fact_ids(
        db, [tagged, other, none], "default", Facets.of(tags="token-optimization"))
    assert keep == {tagged}


def test_tags_filter_is_canonical_not_a_raw_compare(db) -> None:
    tagged = _save(db, "x", tags="Token-Optimization")
    keep = matching_fact_ids(db, [tagged], "default",
                             Facets.of(tags=" token-optimization "))
    assert keep == {tagged}


def test_tags_filter_reads_a_json_array_string(db) -> None:
    # Stored as a real list -> json.dumps makes metadata_json hold a JSON
    # array; json_extract then hands that back as the string '["a", "b"]'.
    tagged = _save(db, "x", tags=["token-optimization", "v4.1.21"])
    keep = matching_fact_ids(db, [tagged], "default", Facets.of(tags="v4.1.21"))
    assert keep == {tagged}


def test_tags_filter_all_requires_every_label(db) -> None:
    both = _save(db, "x", tags="a,b")
    one = _save(db, "y", tags="a")
    keep = matching_fact_ids(db, [both, one], "default",
                             Facets.of(tags=["a", "b"], tags_match="all"))
    assert keep == {both}


def test_tags_filter_any_requires_one_label(db) -> None:
    both = _save(db, "x", tags="a,b")
    one = _save(db, "y", tags="a")
    neither = _save(db, "z", tags="c")
    keep = matching_fact_ids(db, [both, one, neither], "default",
                             Facets.of(tags=["a", "b"], tags_match="any"))
    assert keep == {both, one}


def test_tags_filter_default_match_is_all(db) -> None:
    assert Facets.of(tags=["a", "b"]).tags_match == "all"
    assert Facets.of(tags=["a"], tags_match="bogus").tags_match == "all"
    assert Facets.of(tags=["a"], tags_match="ANY").tags_match == "any"


def test_tags_combine_with_project_as_and(db) -> None:
    both = _save(db, "x", project="slm", tags="decision")
    project_only = _save(db, "y", project="slm", tags="status")
    tag_only = _save(db, "z", project="loops", tags="decision")
    keep = matching_fact_ids(
        db, [both, project_only, tag_only], "default",
        Facets.of(project="slm", tags="decision"))
    assert keep == {both}


def test_a_comma_bearing_label_needs_the_list_form(db) -> None:
    tagged = _save(db, "x", tags=["release, 4.1.22"])
    # The CSV *string* form "release, 4.1.22" can only ever name two labels
    # ("release" and "4.1.22") - it cannot ask for the one comma-bearing tag.
    assert matching_fact_ids(db, [tagged], "default",
                             Facets.of(tags="release, 4.1.22")) == set()
    assert matching_fact_ids(db, [tagged], "default",
                             Facets.of(tags=["release, 4.1.22"])) == {tagged}


def test_tags_filter_duplicates_in_the_request_collapse(db) -> None:
    tagged = _save(db, "x", tags="a")
    facets = Facets.of(tags="a,A, a ")
    assert facets.tags == ("a",)
    assert matching_fact_ids(db, [tagged], "default", facets) == {tagged}


def test_an_edited_tag_is_found_by_the_new_label_not_the_old(db) -> None:
    memory_id = db.store_memory(MemoryRecord(
        profile_id="default", content="x", metadata={"tags": "old-label"}))
    fact_id = db.store_fact(AtomicFact(profile_id="default", memory_id=memory_id,
                                      content="x", fact_type=FactType.SEMANTIC))
    assert matching_fact_ids(db, [fact_id], "default", Facets.of(tags="old-label")) == {fact_id}
    db.store_memory(MemoryRecord(profile_id="default", memory_id=memory_id, content="x",
                                 metadata={"tags": "new-label"}))
    assert matching_fact_ids(db, [fact_id], "default", Facets.of(tags="old-label")) == set()
    assert matching_fact_ids(db, [fact_id], "default", Facets.of(tags="new-label")) == {fact_id}


def test_list_facets_counts_memories_per_project_and_agent(db) -> None:
    _save(db, "1", project="SLM", agent="claude")
    _save(db, "2", project="slm", agent="claude")
    _save(db, "3", project="loops", agent="grok")
    _save(db, "4", project="")
    out = list_facets(db, "default")
    assert out["projects"] == [{"name": "slm", "memories": 2}, {"name": "loops", "memories": 1}]
    assert out["agents"] == [{"name": "claude", "memories": 2}, {"name": "grok", "memories": 1}]


def _save_kind(db, content, *, kind=None, source=None, confidence=None) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id="default", content=content))
    return db.store_fact(AtomicFact(
        profile_id="default", memory_id=memory_id, content=content,
        fact_type=FactType.SEMANTIC, memory_kind=kind, memory_kind_source=source,
        memory_kind_confidence=confidence,
    ))


def test_real_retrieval_engine_is_wired_with_the_configured_threshold(
    mode_a_config, mock_embedder,
) -> None:
    """4.1.19 M3 regression: ``RetrievalEngine.__init__``'s own ``config``
    is a ``RetrievalConfig`` (no ``memory_kinds`` field — that section lives
    on the top-level ``SLMConfig``). A first attempt at this fix read
    ``self._config.memory_kinds.display_min_confidence`` straight off
    ``RetrievalEngine`` and raised ``AttributeError`` on every real recall —
    caught only by running the real engine construction path
    (``core.engine_wiring.init_retrieval``), not by any mock-based unit
    test. This pins the wiring with a value that is NOT the class's own
    default (0.20) — using the default on both sides would pass whether or
    not the value was actually threaded through.
    """
    import dataclasses
    from unittest.mock import patch

    from superlocalmemory.core.engine import MemoryEngine

    mode_a_config.memory_kinds = dataclasses.replace(
        mode_a_config.memory_kinds, display_min_confidence=0.37,
    )
    engine = MemoryEngine(mode_a_config)
    with patch("superlocalmemory.core.engine_wiring.init_embedder", return_value=mock_embedder):
        engine.initialize()
        engine._embedder = mock_embedder
    try:
        assert engine._retrieval_engine is not None
        assert engine._retrieval_engine._display_min_confidence == 0.37
    finally:
        engine.close()


def test_kind_facet_uses_the_configured_display_threshold(db) -> None:
    """4.1.19 M3: the kind facet must honour the CONFIGURED
    ``memory_kinds.display_min_confidence`` everywhere — not the 0.20
    hard-coded default ``storage.memory_kinds.kind_fields`` falls back to
    when nobody passes one.
    """
    low_conf = _save_kind(db, "maybe a decision", kind="decision",
                          source="model:llm", confidence=0.15)
    other = _save_kind(db, "unrelated note")
    ids = [low_conf, other]

    # Default threshold (0.20): a 0.15-confidence suggestion is not shown.
    assert matching_fact_ids(db, ids, "default", Facets.of(kind="decision")) == set()

    # A caller-configured LOWER threshold (0.10) surfaces the same row.
    assert matching_fact_ids(
        db, ids, "default", Facets.of(kind="decision"),
        display_min_confidence=0.10,
    ) == {low_conf}


def _response():
    fact = AtomicFact(fact_id="f1", content="x", fact_type=FactType.SEMANTIC)
    return RecallResponse(query="q", results=[RetrievalResult(fact=fact, score=0.5)])


@pytest.mark.parametrize("facets,expect_key", [(None, False), (Facets.of(), False),
                                               (Facets.of(project="slm"), True)])
def test_facets_reach_retrieval_only_when_set(engine_with_mock_deps, monkeypatch,
                                              facets, expect_key) -> None:
    seen = {}

    def fake_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr(engine_with_mock_deps._retrieval_engine, "recall", fake_recall)
    engine_with_mock_deps.recall("what did we ship", fast=True, facets=facets)
    assert ("facets" in seen) is expect_key
    if expect_key:
        assert seen["facets"].project == "slm"


def test_http_recall_passes_facets_to_the_engine(engine_with_mock_deps, monkeypatch) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    seen = {}

    def fake_engine_recall(*args, **kwargs):
        seen.update(kwargs)
        return _response()

    monkeypatch.setattr(engine_with_mock_deps, "recall", fake_engine_recall)
    with _client(engine_with_mock_deps) as client:
        r1 = client.get("/recall", params={"q": "decisions", "project": "SLM",
                                           "saved_by": "claude-desktop", "about": "Pallavi"})
        plain = dict(seen)
        seen.clear()
        r2 = client.get("/recall", params={"q": "decisions"})
        facets_page = client.get("/api/v3/facets")
    assert r1.status_code == 200, r1.text and r2.status_code == 200
    assert plain["facets"].as_dict() == {"project": "SLM", "agent": "claude-desktop",
                                         "about": "Pallavi"}
    assert "facets" not in seen
    assert facets_page.status_code == 200 and "projects" in facets_page.json()


def test_mcp_proxy_sends_only_the_facets_that_were_set(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q", project="slm", saved_by="", about="Pallavi")
    assert paths, "the proxy did not call the daemon"
    assert "project=slm" in paths[0] and "about=Pallavi" in paths[0]
    assert "saved_by" not in paths[0]


def test_mcp_proxy_sends_a_csv_tags_string_as_is(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q", tags="decision,status")
    assert "tags=decision%2Cstatus" in paths[0]
    assert "tags_match" not in paths[0]


def test_mcp_proxy_sends_a_tags_list_as_repeated_params(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q", tags=["release, 4.1.22", "decision"], tags_match="any")
    assert paths[0].count("tags=") == 2
    assert "tags_match=any" in paths[0]


def test_mcp_proxy_sends_no_tags_param_when_unset(monkeypatch) -> None:
    from superlocalmemory.cli import daemon
    from superlocalmemory.mcp import _daemon_proxy

    paths = []
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    proxy = _daemon_proxy.DaemonPoolProxy(port=8765)
    proxy.recall("q")
    assert "tags=" not in paths[0]
    assert "tags_match" not in paths[0]


def test_cli_recall_adds_the_facets_to_the_request(monkeypatch) -> None:
    from superlocalmemory.cli import commands, daemon

    paths = []
    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request",
                        lambda method, path, *a, **k: paths.append(path) or {"results": []})
    args = Namespace(query="what did we decide", limit=5, json=True, project="SLM",
                     saved_by="claude-desktop", about="", fast=None, window="", as_of="",
                     known_as_of="", valid_at="", include_unknown=False, include_global=None,
                     include_shared=None, session_id="", profile_id="")
    try:
        commands.cmd_recall(args)
    except SystemExit:
        pass
    assert paths, "the CLI did not call the daemon"
    assert "&project=SLM" in paths[0] and "&saved_by=claude-desktop" in paths[0]
    assert "&about=" not in paths[0]
