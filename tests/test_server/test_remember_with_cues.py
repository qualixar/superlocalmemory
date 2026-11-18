# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saving a memory with cues (W3-8): ``cues`` on HTTP, MCP and the CLI. A bad cue is
refused before anything is saved; a caller's metadata can never set the reserved key;
the next maintenance pass binds the cues to the memory's first fact."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.cue_binding import METADATA_KEY
from tests.test_server.test_canonical_remember_route import _client
from tests.test_server.test_remember_with_a_slot import _cli, _daemon_body

CONTENT = "Run the schema migration check before pushing."
CUES = ["path:src/**/schema.py", "tool:Bash:git push"]


def _stored(engine) -> list:
    return [json.loads(r["metadata_json"] or "{}").get(METADATA_KEY)
            for r in engine._db.execute("SELECT metadata_json FROM memories")]


def test_http_remember_records_the_cues_and_the_pass_binds_them(engine_with_mock_deps) -> None:
    from superlocalmemory.core import cue_binding

    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={"content": CONTENT, "cues": CUES + ["PATH:x/y.py"],
                                                  "idempotency_key": "cues-http-1"})
    assert response.status_code == 200, response.text
    assert _stored(engine_with_mock_deps) == [CUES + ["path:x/y.py"]]
    assert cue_binding.run(engine_with_mock_deps._db, engine_with_mock_deps._db.db_path.parent) == 3
    rows = engine_with_mock_deps._db.execute("SELECT kind, pattern, origin FROM memory_cues")
    assert {(r["kind"], r["pattern"], r["origin"]) for r in rows} == {
        ("path", "src/**/schema.py", "explicit"), ("tool", "Bash:git push", "explicit"),
        ("path", "x/y.py", "explicit")}


def test_a_memory_saved_without_cues_has_no_cue_key(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        assert client.post("/remember", json={"content": CONTENT, "cues": [],
                                              "idempotency_key": "cues-http-2"}).status_code == 200
    assert _stored(engine_with_mock_deps) == [None]


def test_a_bad_cue_is_refused_and_nothing_is_saved(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        for cues in (["nope"], ["date:soon"], ["path:ok.py", "repo:a/b"], ["tool:Bash:"]):
            response = client.post("/remember", json={"content": CONTENT, "cues": cues})
            assert response.status_code == 422, cues
            assert response.json()["detail"]["code"] == "INVALID_CUE"
        too_many = [f"repo:r{i}" for i in range(9)]
        assert client.post("/remember", json={"content": CONTENT, "cues": too_many}
                           ).status_code == 422
    assert engine_with_mock_deps._db.execute("SELECT * FROM memories") == []


def test_metadata_cannot_smuggle_cues(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "idempotency_key": "cues-smuggle-1",
            "metadata": {METADATA_KEY: ["repo:anything"]}})
    assert response.status_code == 200, response.text
    assert _stored(engine_with_mock_deps) == [None]


# -- the tool interface ------------------------------------------------------------------------

def test_mcp_remember_sends_the_cues_as_a_request_field(monkeypatch) -> None:
    _out, body = _daemon_body(monkeypatch, cues=CUES)
    assert body["cues"] == CUES and METADATA_KEY not in body["metadata"]
    _out, plain = _daemon_body(monkeypatch)
    assert "cues" not in plain and plain["idempotency_key"] != body["idempotency_key"]


def test_mcp_remember_refuses_a_bad_cue_before_sending(monkeypatch) -> None:
    out, body = _daemon_body(monkeypatch, cues=["path:ok.py", "wrong"])
    assert out["success"] is False and out["code"] == "INVALID_CUE" and body is None
    assert "KIND:PATTERN" in out["error"] or "kind" in out["error"]


def test_the_fallback_proxy_lifts_the_cues_into_a_request_field(monkeypatch) -> None:
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon_mod, "daemon_request", fake_request)
    DaemonPoolProxy(port=1).store(CONTENT, {"cues": CUES, "tags": ""})
    assert sent["body"]["cues"] == CUES and "cues" not in sent["body"]["metadata"]


# -- the command line --------------------------------------------------------------------------

def test_cli_remember_sends_each_cue(monkeypatch) -> None:
    body = _cli(monkeypatch, slot=None, slot_value=None, cues=["Path:a/b.py", "repo:x"])
    assert body["cues"] == ["path:a/b.py", "repo:x"]
    plain = _cli(monkeypatch, slot=None, slot_value=None)
    assert "cues" not in plain and plain["idempotency_key"] != body["idempotency_key"]


def test_cli_remember_refuses_a_bad_cue_before_contacting_slm(monkeypatch, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _cli(monkeypatch, slot=None, slot_value=None, cues=["whoops"])
    assert exit_info.value.code == 2 and "KIND:PATTERN" in capsys.readouterr().err


def test_slm_remember_offers_a_repeatable_cue_flag(tmp_path) -> None:
    import os
    import subprocess
    import sys

    env = {**os.environ, "COLUMNS": "200", "SLM_DATA_DIR": str(tmp_path)}
    out = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "remember", "--help"],
                         capture_output=True, text=True, timeout=60, check=True, env=env).stdout
    assert "--cue KIND:PATTERN" in out
    lines = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "cues", "list",
                            "--json"], capture_output=True, text=True, timeout=60, check=True,
                           env=env).stdout
    assert json.loads(lines)["data"] == {"cues": [], "count": 0}
