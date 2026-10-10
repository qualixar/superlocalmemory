# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A remote app cannot delete, by id, a memory it is not allowed to see (audit F-11).

Ids are not guessable, but a web app that learned the id of a folder file or an
unvetted picture must not be able to remove it: the delete follows the same view
as recall. A caller on this computer deletes anything in its profile, as before.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tests.test_mcp.test_remote_more_reads_hide_media import (  # noqa: F401  (fixture and helpers)
    _as, _Server, world)

LOOPBACK = ("127.0.0.1", 50000)
CALLERS = [("local", False, False), ("remote", True, False), ("remote_media", True, True)]
HIDDEN_FROM = {  # text -> the callers that must not be able to delete it
    "zebra picture good": {"remote"},
    "zebra picture dirty": {"remote", "remote_media"},
    "zebra folder note": {"remote", "remote_media"},
    "zebra plain note": set(),
}


def _daemon(engine) -> TestClient:
    from superlocalmemory.server.unified_daemon import create_app

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    descriptor = app.state.daemon_descriptor
    # What the tools send: the daemon's own capability, so the write is allowed.
    headers = {"X-SLM-Daemon-Capability": descriptor.capability,
               "X-SLM-Target-Instance": descriptor.instance_id}
    return TestClient(app, base_url="http://127.0.0.1:8765", client=LOOPBACK,
                      raise_server_exceptions=False, headers=headers)


def _exists(engine, fact_id: str) -> bool:
    return bool(engine._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,)))


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
@pytest.mark.parametrize("text", sorted(HIDDEN_FROM))
def test_the_daemon_route_refuses_a_hidden_memory_to_a_remote_view(world, who, key, media, text) -> None:
    engine, ids = world
    view = {"local": "", "remote": "remote", "remote_media": "remote_media"}[who]
    params = {"caller_view": view} if view else {}

    reply = _daemon(engine).delete(f"/api/memories/{ids[text]}", params=params)

    if who in HIDDEN_FROM[text]:
        assert reply.status_code == 404, (reply.status_code, reply.text)
        assert reply.json()["detail"] == "Memory not found"   # the same words as a missing id
        assert _exists(engine, ids[text])
    else:
        assert reply.status_code != 404, reply.text           # past the view check: the delete ran


@pytest.mark.parametrize("who,key,media", CALLERS, ids=[c[0] for c in CALLERS])
def test_the_tool_tells_the_daemon_who_is_asking(world, monkeypatch, who, key, media) -> None:
    from superlocalmemory.mcp.tools_core import register_core_tools

    engine, ids = world
    sent: list[str] = []

    def request(method, path, body=None, **kw):
        sent.append(path)
        return {"success": True}

    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request", request)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda *a, **k: True)
    server = _Server()
    register_core_tools(server, lambda: engine)
    with _as(key, media):
        out = asyncio.run(server.tools["delete_memory"](fact_id=ids["zebra plain note"]))

    assert out["success"], out
    (path,) = sent
    if who == "local":
        assert "caller_view" not in path
    else:
        assert f"caller_view={who}" in path
