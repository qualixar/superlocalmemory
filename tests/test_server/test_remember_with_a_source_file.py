# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saving a memory with the file it came from (D-38): ``source_file`` on HTTP, MCP and
the CLI. The daemon fingerprints the file at the door; a bad path or a remote
caller is refused before anything is saved; metadata can never forge a lease."""

from __future__ import annotations

import json

from superlocalmemory.truth.leases import METADATA_KEY
from tests.test_server.test_canonical_remember_route import _client
from tests.test_server.test_remember_with_a_slot import _cli, _daemon_body

CONTENT = "The daemon listens on port 8765."


def _leases(engine) -> list:
    return [json.loads(r["metadata_json"] or "{}").get(METADATA_KEY)
            for r in engine._db.execute("SELECT metadata_json FROM memories")]


def test_http_remember_fingerprints_the_source_file(engine_with_mock_deps, tmp_path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={"content": CONTENT, "source_file": str(src),
                                                  "idempotency_key": "lease-http-1"})
    assert response.status_code == 200, response.text
    (lease,) = _leases(engine_with_mock_deps)
    assert lease["path"] == str(src.resolve()) and len(lease["sha256"]) == 64
    assert lease["size"] == src.stat().st_size


def test_a_bad_source_file_is_refused_and_nothing_is_saved(engine_with_mock_deps, tmp_path) -> None:
    with _client(engine_with_mock_deps) as client:
        for path in ("relative.txt", str(tmp_path / "missing.txt"), str(tmp_path)):
            response = client.post("/remember", json={"content": CONTENT, "source_file": path})
            assert response.status_code == 422, path
            assert response.json()["detail"]["code"] == "INVALID_SOURCE_FILE"
    assert engine_with_mock_deps._db.execute("SELECT * FROM memories") == []


def test_a_remote_caller_cannot_name_a_source_file(engine_with_mock_deps, tmp_path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("port: 8765\n")
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", params={"caller_view": "remote"},
                               json={"content": CONTENT, "source_file": str(src)})
    assert response.status_code == 422
    assert engine_with_mock_deps._db.execute("SELECT * FROM memories") == []


def test_metadata_cannot_forge_a_lease(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "idempotency_key": "lease-smuggle-1",
            "metadata": {METADATA_KEY: {"path": "/etc/hosts"}}})
    assert response.status_code == 200, response.text
    assert _leases(engine_with_mock_deps) == [None]


def test_mcp_remember_sends_the_source_file_as_a_request_field(monkeypatch) -> None:
    _out, body = _daemon_body(monkeypatch, source_file="/Users/someone/project/config.yaml")
    assert body["source_file"] == "/Users/someone/project/config.yaml"
    assert "source_file" not in body["metadata"]
    _out, plain = _daemon_body(monkeypatch)
    assert "source_file" not in plain and plain["idempotency_key"] != body["idempotency_key"]


def test_mcp_remember_refuses_a_relative_source_file_before_sending(monkeypatch) -> None:
    out, body = _daemon_body(monkeypatch, source_file="config.yaml")
    assert out["success"] is False and out["code"] == "INVALID_SOURCE_FILE" and body is None


def test_the_fallback_proxy_lifts_the_source_file_into_a_request_field(monkeypatch) -> None:
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon_mod, "daemon_request", fake_request)
    DaemonPoolProxy(port=1).store(CONTENT, {"source_file": "/a/b.txt", "tags": ""})
    assert sent["body"]["source_file"] == "/a/b.txt"
    assert "source_file" not in sent["body"]["metadata"]


def test_cli_remember_sends_the_source_file_as_an_absolute_path(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    body = _cli(monkeypatch, slot=None, slot_value=None, source_file="config.yaml")
    assert body["source_file"] == str((tmp_path / "config.yaml").resolve())
