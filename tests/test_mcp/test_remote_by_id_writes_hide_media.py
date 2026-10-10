# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A remote app cannot change, by id, a memory it is not allowed to see (audit F-11, widened).

``test_remote_delete_hides_media`` covers delete. The same rule holds for every other
tool that takes the id of a memory and writes: update_memory, set_memory_kind,
confirm_memory_kinds, remember(replaces=...), review_correction, core_memory pin/unpin,
report_feedback and report_outcome. A memory the app's recall hides (a connected-folder
file, a picture not cleared for remote apps) is answered like an id that does not exist,
and nothing is written. A caller on this computer is unchanged.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests.test_mcp.test_remote_delete_hides_media import _daemon, _exists
from tests.test_mcp.test_remote_more_reads_hide_media import (  # noqa: F401  (fixture and helpers)
    _as, _Server, world)

HIDDEN = {"remote": ("zebra picture good", "zebra picture dirty", "zebra folder note"),
          "remote_media": ("zebra picture dirty", "zebra folder note"),
          "local": ()}
VIEWS = ["remote", "remote_media"]
PLAIN = "zebra plain note"
NEW_TEXT = "The release train for the platform team leaves on Thursday at 09:00 UTC."


def _kind_of(engine, fact_id: str):
    rows = engine._db.execute("SELECT memory_kind FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    return dict(rows[0])["memory_kind"] if rows else None


def _content_of(engine, fact_id: str):
    rows = engine._db.execute("SELECT content FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    return dict(rows[0])["content"] if rows else None


# -- daemon routes ------------------------------------------------------------------------------


@pytest.mark.parametrize("view", VIEWS)
def test_the_update_route_refuses_a_hidden_memory(world, view) -> None:
    engine, ids = world
    for text in HIDDEN[view]:
        reply = _daemon(engine).patch(f"/api/memories/{ids[text]}", params={"caller_view": view},
                                      json={"content": "changed by a remote app"})
        assert reply.status_code == 404, (text, reply.status_code, reply.text)
        assert reply.json()["detail"] == "Memory not found"
        assert _content_of(engine, ids[text]) == text


@pytest.mark.parametrize("view", VIEWS)
def test_the_update_route_still_reaches_a_visible_memory(world, view) -> None:
    engine, ids = world
    reply = _daemon(engine).patch(f"/api/memories/{ids[PLAIN]}", params={"caller_view": view},
                                  json={"content": "changed"})
    assert reply.status_code != 404, reply.text   # past the view check


@pytest.mark.parametrize("view", VIEWS)
def test_the_kind_route_refuses_a_hidden_memory(world, view) -> None:
    engine, ids = world
    for text in HIDDEN[view]:
        before = _kind_of(engine, ids[text])
        reply = _daemon(engine).patch(f"/api/memory-kinds/fact/{ids[text]}", params={"caller_view": view},
                                      json={"kind": "decision"})
        assert reply.status_code == 404, (text, reply.status_code, reply.text)
        assert reply.json()["detail"] == "Memory not found"
        assert _kind_of(engine, ids[text]) == before


@pytest.mark.parametrize("view", VIEWS)
def test_the_confirm_route_answers_a_hidden_memory_as_not_found(world, view) -> None:
    engine, ids = world
    items = [{"fact_id": ids[t], "kind": "decision"} for t in HIDDEN[view]]
    reply = _daemon(engine).post("/api/memory-kinds/confirm", params={"caller_view": view},
                                 json={"items": items})
    assert reply.status_code == 200, reply.text
    assert [(i["ok"], i["error"]) for i in reply.json()["items"]] == [(False, "Memory not found.")] * len(items)
    assert all(_kind_of(engine, ids[t]) != "decision" for t in HIDDEN[view])


@pytest.mark.parametrize("view", VIEWS)
def test_remember_cannot_replace_a_hidden_memory(world, view) -> None:
    from superlocalmemory.core.remember_replaces import not_found
    from tests.test_server.test_canonical_remember_route import _client

    engine, ids = world
    with _client(engine) as client:
        for n, text in enumerate(HIDDEN[view]):
            reply = client.post("/remember", params={"caller_view": view},
                                json={"content": f"{NEW_TEXT} {n}", "replaces": ids[text],
                                      "idempotency_key": f"hidden-{view}-{n}"})
            assert reply.status_code == 422, (text, reply.status_code, reply.text)
            # word for word the refusal for an id that does not exist
            assert reply.json()["detail"] == not_found(ids[text]).as_error()
            assert _exists(engine, ids[text])


@pytest.mark.parametrize("view", VIEWS)
def test_remember_may_still_replace_a_visible_memory(world, view) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    engine, ids = world
    with _client(engine) as client:
        reply = client.post("/remember", params={"caller_view": view},
                            json={"content": NEW_TEXT, "replaces": ids[PLAIN],
                                  "idempotency_key": f"visible-{view}"})
    assert reply.status_code == 200, reply.text


def test_remember_from_this_computer_may_replace_a_folder_memory(world) -> None:
    from tests.test_server.test_canonical_remember_route import _client

    engine, ids = world
    with _client(engine) as client:
        reply = client.post("/remember", json={"content": NEW_TEXT, "replaces": ids["zebra folder note"],
                                               "idempotency_key": "local-folder"})
    assert reply.status_code == 200, reply.text


@pytest.mark.parametrize("view", VIEWS)
def test_a_correction_case_naming_a_hidden_memory_cannot_be_reviewed(world, monkeypatch, view) -> None:
    from superlocalmemory.server.routes import memories

    engine, ids = world

    class Store:
        def get_case(self, case_id):
            return SimpleNamespace(predecessor_fact_id=ids["zebra folder note"],
                                   successor_fact_id=ids[PLAIN])

    monkeypatch.setattr(memories, "_correction_store_for", lambda eng, profile: Store())
    reply = _daemon(engine).post("/api/corrections/case1/apply", params={"caller_view": view},
                                 json={"expected_version": 1})
    assert reply.status_code == 404, reply.text
    assert reply.json()["detail"] == "Correction case not found"


# -- the tools tell the daemon who is asking ----------------------------------------------------

CALLERS = [("local", False, False), ("remote", True, False), ("remote_media", True, True)]


def _record_paths(monkeypatch, reply) -> list[str]:
    paths: list[str] = []

    def request(method, path, body=None, **kw):
        paths.append(path)
        return reply

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda *a, **k: True)
    return paths


def _check_marked(paths: list[str], who: str) -> None:
    (path,) = paths
    if who == "local":
        assert "caller_view" not in path
    else:
        assert f"caller_view={who}" in path


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_update_memory_marks_its_request(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine, ids = world
    paths = _record_paths(monkeypatch, {"success": True})
    server = _Server()
    register_core_tools(server, lambda: engine)
    with _as(key, media):
        asyncio.run(server.tools["update_memory"](fact_id=ids[PLAIN], content="x"))
    _check_marked(paths, who)


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_set_memory_kind_marks_its_request(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_kinds import register_kind_tools

    engine, ids = world
    paths = _record_paths(monkeypatch, {"success": True, "ok": True})
    server = _Server()
    register_kind_tools(server, lambda: engine)
    with _as(key, media):
        asyncio.run(server.tools["set_memory_kind"](fact_id=ids[PLAIN], kind="decision"))
    _check_marked(paths, who)


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_confirm_memory_kinds_marks_its_request(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_kinds import register_kind_tools

    engine, ids = world
    paths = _record_paths(monkeypatch, {"items": []})
    server = _Server()
    register_kind_tools(server, lambda: engine)
    with _as(key, media):
        asyncio.run(server.tools["confirm_memory_kinds"](items=[{"fact_id": ids[PLAIN], "kind": "decision"}]))
    _check_marked(paths, who)


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_review_correction_marks_its_request(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine, _ = world
    paths = _record_paths(monkeypatch, {"success": True})
    server = _Server()
    register_core_tools(server, lambda: engine)
    with _as(key, media):
        asyncio.run(server.tools["review_correction"](case_id="c1", action="apply", expected_version=1))
    _check_marked(paths, who)


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_remember_marks_its_request_when_it_replaces_something(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine, ids = world
    paths = _record_paths(monkeypatch, {"success": True, "fact_ids": [], "memory_id": "m"})
    server = _Server()
    register_core_tools(server, lambda: engine)
    with _as(key, media):
        asyncio.run(server.tools["remember"](content="fresh words zz", replaces=ids[PLAIN]))
    remember_paths = [p for p in paths if p.startswith("/remember")]
    _check_marked(remember_paths, who)


# -- tools that write inside the daemon ---------------------------------------------------------


@pytest.mark.parametrize("action", ["pin", "unpin"])
@pytest.mark.parametrize("view", VIEWS)
def test_core_memory_cannot_pin_or_unpin_a_hidden_memory(world, view, action) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    server = _Server()
    register_active_tools(server, lambda: engine)
    key, media = True, view == "remote_media"
    for text in HIDDEN[view]:
        with _as(key, media):
            out = asyncio.run(server.tools["core_memory"](action=action, fact_id=ids[text]))
        assert out == {"success": False, "error": f"Memory {ids[text]} not found"}, out
        assert not engine._db.get_pinned(engine.profile_id) or ids[text] not in {
            f.fact_id for f in engine._db.get_pinned(engine.profile_id)}


def test_core_memory_pins_a_visible_memory_and_a_local_caller_pins_anything(world) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    server = _Server()
    register_active_tools(server, lambda: engine)
    with _as(True, False):
        assert asyncio.run(server.tools["core_memory"](action="pin", fact_id=ids[PLAIN]))["success"]
    with _as(False, False):
        assert asyncio.run(server.tools["core_memory"](action="pin", fact_id=ids["zebra folder note"]))["success"]


@pytest.mark.parametrize("view", VIEWS)
def test_report_feedback_refuses_a_hidden_or_missing_memory_alike(world, view) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    server = _Server()
    register_active_tools(server, lambda: engine)
    key, media = True, view == "remote_media"
    outs = []
    for fact_id in [ids[t] for t in HIDDEN[view]] + ["feedfacefeedface"]:
        with _as(key, media):
            outs.append(asyncio.run(server.tools["report_feedback"](fact_id=fact_id, feedback="relevant")))
    assert all(o == {"success": False, "error": "Memory not found."} for o in outs), outs


@pytest.mark.parametrize("view", VIEWS)
def test_report_feedback_still_records_for_a_visible_memory(world, view) -> None:
    from superlocalmemory.mcp.tools_active import register_active_tools

    engine, ids = world
    server = _Server()
    register_active_tools(server, lambda: engine)
    with _as(True, view == "remote_media"):
        out = asyncio.run(server.tools["report_feedback"](fact_id=ids[PLAIN], feedback="relevant"))
    assert out.get("error") != "Memory not found.", out


@pytest.mark.parametrize("view", VIEWS)
def test_report_outcome_drops_the_ids_a_remote_app_may_not_act_on(world, monkeypatch, view) -> None:
    from superlocalmemory.learning import outcomes
    from superlocalmemory.mcp.tools_v28 import register_v28_tools

    engine, ids = world
    seen: list[list[str]] = []

    def record(self, **kw):
        seen.append(list(kw["fact_ids"]))
        return SimpleNamespace(outcome_id="o1")

    monkeypatch.setattr(outcomes.OutcomeTracker, "record_outcome", record)
    server = _Server()
    register_v28_tools(server, lambda: engine)
    key, media = True, view == "remote_media"
    hidden = [ids[t] for t in HIDDEN[view]]
    with _as(key, media):
        mixed = asyncio.run(server.tools["report_outcome"](
            memory_ids=",".join([*hidden, ids[PLAIN]]), outcome="success"))
        only_hidden = asyncio.run(server.tools["report_outcome"](
            memory_ids=",".join(hidden), outcome="success"))
    assert mixed["success"], mixed
    assert seen == [[ids[PLAIN]]], seen
    assert only_hidden == {"success": False, "error": "Memory not found."}


def test_report_outcome_for_a_local_caller_keeps_every_id(world, monkeypatch) -> None:
    from superlocalmemory.learning import outcomes
    from superlocalmemory.mcp.tools_v28 import register_v28_tools

    engine, ids = world
    seen: list[list[str]] = []
    monkeypatch.setattr(outcomes.OutcomeTracker, "record_outcome",
                        lambda self, **kw: seen.append(list(kw["fact_ids"])) or SimpleNamespace(outcome_id="o1"))
    server = _Server()
    register_v28_tools(server, lambda: engine)
    wanted = [ids["zebra folder note"], "feedfacefeedface"]
    with _as(False, False):
        asyncio.run(server.tools["report_outcome"](memory_ids=",".join(wanted), outcome="success"))
    assert seen == [wanted]
