# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Saving a memory with a slot (D-36): ``POST /remember`` with ``slot`` and
``slot_value`` records the claim on the memory; a malformed slot is refused
before anything is saved; a caller's metadata can never set one."""

from __future__ import annotations

import json

from superlocalmemory.truth.slot_claims import METADATA_KEY
from tests.test_server.test_canonical_remember_route import _client

CONTENT = "From now on SLM ships one big release a month."


def _claims(engine) -> list:
    return [json.loads(r["metadata_json"] or "{}").get(METADATA_KEY)
            for r in engine._db.execute("SELECT metadata_json FROM memories")]


def test_http_remember_with_a_slot_records_the_claim(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "slot": "SLM/release cadence", "slot_value": "monthly",
            "idempotency_key": "slot-http-1"})
    assert response.status_code == 200, response.text
    assert _claims(engine_with_mock_deps) == [
        {"spec": "SLM/release cadence", "value": "monthly"}]


def test_a_slot_with_no_value_is_recorded_without_one(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "slot": "SLM/release cadence", "idempotency_key": "slot-http-2"})
    assert response.status_code == 200, response.text
    assert _claims(engine_with_mock_deps) == [{"spec": "SLM/release cadence"}]


def test_a_malformed_slot_is_refused_and_nothing_is_saved(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        for body in ({"slot": "only-one-part"}, {"slot_value": "monthly"},
                     {"slot": "a/b", "slot_value": "v" * 241}):
            response = client.post("/remember", json={"content": CONTENT, **body})
            assert response.status_code == 422, body
            assert response.json()["detail"]["code"] == "INVALID_SLOT"
    assert engine_with_mock_deps._db.execute("SELECT * FROM memories") == []


def test_metadata_cannot_smuggle_a_slot_claim(engine_with_mock_deps) -> None:
    with _client(engine_with_mock_deps) as client:
        response = client.post("/remember", json={
            "content": CONTENT, "idempotency_key": "slot-smuggle-1",
            "metadata": {METADATA_KEY: {"spec": "SLM/release cadence", "value": "monthly"}}})
    assert response.status_code == 200, response.text
    assert _claims(engine_with_mock_deps) == [None]


# -- the tool interface ------------------------------------------------------------------------

def _mcp_remember():
    from tests.test_server.test_remember_with_a_kind import _mcp_remember as build

    return build()


def _daemon_body(monkeypatch, **kwargs):
    import asyncio

    from superlocalmemory.cli import daemon

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    out = asyncio.run(_mcp_remember()(CONTENT, agent_id="agent-x", **kwargs))
    return out, sent.get("body")


def test_mcp_remember_sends_the_slot_as_request_fields(monkeypatch) -> None:
    _out, body = _daemon_body(monkeypatch, slot="SLM/release cadence", slot_value="monthly")
    assert (body["slot"], body["slot_value"]) == ("SLM/release cadence", "monthly")
    assert METADATA_KEY not in body["metadata"]


def test_mcp_remember_without_a_slot_sends_no_slot_field(monkeypatch) -> None:
    _out, body = _daemon_body(monkeypatch)
    assert "slot" not in body and "slot_value" not in body


def test_mcp_remember_refuses_a_malformed_slot_before_sending(monkeypatch) -> None:
    out, body = _daemon_body(monkeypatch, slot="nope")
    assert out["success"] is False and out["code"] == "INVALID_SLOT" and body is None


def test_mcp_derived_key_includes_the_slot_only_when_one_is_named(monkeypatch) -> None:
    keys = [_daemon_body(monkeypatch, **kw)[1]["idempotency_key"] for kw in (
        {}, {}, {"slot": "a/b", "slot_value": "x"}, {"slot": "a/b", "slot_value": "x"},
        {"slot": "a/b", "slot_value": "y"})]
    plain, plain_again, x, x_again, y = keys
    assert plain == plain_again and x == x_again and len({plain, x, y}) == 3


def test_the_fallback_proxy_lifts_the_slot_into_request_fields(monkeypatch) -> None:
    from superlocalmemory.cli import daemon as daemon_mod
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon_mod, "daemon_request", fake_request)
    DaemonPoolProxy(port=1).store(CONTENT, {"slot": "a/b", "slot_value": "x", "tags": ""})
    body = sent["body"]
    assert (body["slot"], body["slot_value"]) == ("a/b", "x")
    assert "slot" not in body["metadata"] and "slot_value" not in body["metadata"]


# -- the command line --------------------------------------------------------------------------

def _cli(monkeypatch, **kwargs):
    from argparse import Namespace

    from superlocalmemory.cli import commands, daemon

    sent = {}

    def fake_request(method, path, body=None, **flags):
        sent.update(body=body)
        return {"ok": True, "fact_ids": ["f1"], "count": 1, "status": "stored"}

    monkeypatch.setattr(daemon, "is_daemon_running", lambda: True)
    monkeypatch.setattr(daemon, "daemon_request", fake_request)
    commands.cmd_remember(Namespace(content=CONTENT, tags="", kind="", json=False,
                                    sync_mode=False, scope=None, shared_with=None, **kwargs))
    return sent.get("body")


def test_cli_remember_sends_the_slot(monkeypatch) -> None:
    body = _cli(monkeypatch, slot="SLM/release cadence", slot_value="monthly")
    assert (body["slot"], body["slot_value"]) == ("SLM/release cadence", "monthly")
    plain = _cli(monkeypatch, slot=None, slot_value=None)
    assert "slot" not in plain and plain["idempotency_key"] != body["idempotency_key"]


def test_cli_remember_refuses_a_malformed_slot_before_contacting_slm(monkeypatch, capsys) -> None:
    import pytest

    with pytest.raises(SystemExit) as exit_info:
        _cli(monkeypatch, slot="a//b", slot_value=None)
    assert exit_info.value.code == 2 and "subject/relation" in capsys.readouterr().err


def test_slm_remember_offers_slot_and_slot_value() -> None:
    import os
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "remember", "--help"],
                         capture_output=True, text=True, timeout=60, check=True,
                         env={**os.environ, "COLUMNS": "200"}).stdout
    assert "--slot SLOT" in out and "--slot-value SLOT_VALUE" in out
