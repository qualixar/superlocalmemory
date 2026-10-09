"""Web-origin sends: redaction, envelope rows, hop limit, mute, rate limit, revoke."""

from __future__ import annotations

import json
import time

import pytest

from superlocalmemory.mesh import broker as broker_mod
from superlocalmemory.mesh.envelope import Origin
from tests.test_mesh.conftest import make_peer, rows

WEB = Origin("web", "notes")
FAKE_KEY = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0" * 3


def _web_peer(broker, name="web-1"):
    pid = make_peer(broker, name, "notes")
    broker.upsert_peer_profile(pid, kind="web", app_name="notes", display_name="Notes")
    return pid


def test_local_send_is_unchanged_and_writes_no_envelope(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    msg = f"key {FAKE_KEY} stays verbatim"
    res = broker.send_message(a, b, msg)
    assert set(res) == {"ok", "id", "target_type", "expires_at"}
    row = rows(broker, "SELECT * FROM mesh_messages WHERE id=?", (res["id"],))[0]
    assert row["content"] == msg and row["from_peer"] == a and row["to_peer"] == b
    assert row["read"] == 0 and row["target_type"] == "peer" and row["msg_type"] == "text"
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []
    # explicit local origin behaves the same
    res2 = broker.send_message(a, b, msg, origin=Origin("local"))
    assert set(res2) == set(res)
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []


def test_web_send_redacts_and_writes_envelope(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    res = broker.send_message(w, b, f"use {FAKE_KEY} now", origin=WEB, refs=["fact:abcdef"])
    assert res["ok"]
    row = rows(broker, "SELECT content FROM mesh_messages WHERE id=?", (res["id"],))[0]
    assert FAKE_KEY not in row["content"] and "REDACTED" in row["content"]
    e = rows(broker, "SELECT * FROM mesh_message_envelopes WHERE message_id=?", (res["id"],))[0]
    assert (e["from_kind"], e["from_app"], e["hop"]) == ("web", "notes", 0)
    assert json.loads(e["refs_json"]) == ["fact:abcdef"]


def test_failed_envelope_insert_rolls_back_message(broker, monkeypatch) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    from superlocalmemory.mesh import broker_profiles

    def boom(*a, **k):
        raise RuntimeError("envelope write failed")

    monkeypatch.setattr(broker_profiles, "insert_envelope", boom)
    with pytest.raises(RuntimeError):
        broker.send_message(w, b, "hello", origin=WEB)
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_bad_refs_refused(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    res = broker.send_message(w, b, "hi", origin=WEB, refs=["nope"])
    assert res["ok"] is False and "ref" in res["error"]
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_hop_chain_zero_one_two_then_refused(broker) -> None:
    w1, w2 = _web_peer(broker, "w1"), _web_peer(broker, "w2")
    m0 = broker.send_message(w1, w2, "m0", origin=WEB)["id"]
    m1 = broker.send_message(w2, w1, "m1", origin=WEB, reply_to=m0)["id"]
    m2 = broker.send_message(w1, w2, "m2", origin=WEB, reply_to=m1)["id"]
    hops = [r["hop"] for r in rows(broker, "SELECT hop FROM mesh_message_envelopes ORDER BY message_id")]
    assert hops == [0, 1, 2]
    res = broker.send_message(w2, w1, "m3", origin=WEB, reply_to=m2)
    assert res["ok"] is False and "hop limit" in res["error"]


def test_reply_to_local_message_stays_hop_zero(broker) -> None:
    a, w = make_peer(broker, "a"), _web_peer(broker)
    m = broker.send_message(a, w, "from local")["id"]
    broker.send_message(w, a, "reply", origin=WEB, reply_to=m)
    hop = rows(broker, "SELECT hop FROM mesh_message_envelopes")[0]["hop"]
    assert hop == 0


def test_unknown_parent_is_hop_zero(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    assert broker.send_message(w, b, "x", origin=WEB, reply_to=99999)["ok"]
    assert rows(broker, "SELECT hop FROM mesh_message_envelopes")[0]["hop"] == 0


def test_muted_sender_refused_and_unmute_restores(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    assert broker.set_muted(w, True)["ok"]
    res = broker.send_message(w, b, "hi", origin=WEB)
    assert res == {"ok": False, "error": "peer is muted by the owner"}
    assert rows(broker, "SELECT * FROM mesh_messages") == []
    broker.set_muted(w, False)
    assert broker.send_message(w, b, "hi", origin=WEB)["ok"]


def test_rate_limit_21st_refused(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    for i in range(20):
        assert broker.send_message(w, b, f"m{i}", origin=WEB)["ok"], i
    res = broker.send_message(w, b, "m21", origin=WEB)
    assert res["ok"] is False and res["error"] == "send rate limit"
    assert 1 <= res["retry_after_s"] <= 60


def test_rate_limit_does_not_apply_to_local(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    for i in range(25):
        assert broker.send_message(a, b, f"m{i}")["ok"]


def test_rate_limit_window_slides(broker, monkeypatch) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    for i in range(20):
        broker.send_message(w, b, f"m{i}", origin=WEB)
    real = time.monotonic
    monkeypatch.setattr(broker_mod.time, "monotonic", lambda: real() + 61)
    assert broker.send_message(w, b, "later", origin=WEB)["ok"]


def test_retire_drops_queued_messages_and_blocks_sends(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    out = broker.send_message(w, b, "out", origin=WEB)["id"]
    inn = broker.send_message(b, w, "in")["id"]
    other = broker.send_message(b, b, "keep")["id"]
    res = broker.retire_peer(w)
    assert res["ok"] and res["dropped"] == 2
    ids = {r["id"] for r in rows(broker, "SELECT id FROM mesh_messages")}
    assert ids == {other} and out not in ids and inn not in ids
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []
    prof = rows(broker, "SELECT retired_at FROM mesh_peer_profiles WHERE peer_id=?", (w,))[0]
    assert prof["retired_at"]
    assert broker.send_message(w, b, "again", origin=WEB)["ok"] is False


def test_ttl_cleanup_removes_envelopes(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    mid = broker.send_message(w, b, "old", origin=WEB)["id"]
    conn = broker._conn()
    conn.execute("UPDATE mesh_messages SET expires_at='2000-01-01T00:00:00+00:00' WHERE id=?", (mid,))
    conn.commit()
    conn.close()
    broker._run_cleanup()
    assert rows(broker, "SELECT * FROM mesh_messages") == []
    assert rows(broker, "SELECT * FROM mesh_message_envelopes") == []


def test_peer_profile_helpers(broker) -> None:
    w = _web_peer(broker)
    assert broker.rename_peer(w, "Shiny Notes")["ok"]
    p = rows(broker, "SELECT * FROM mesh_peer_profiles WHERE peer_id=?", (w,))[0]
    assert p["display_name"] == "Shiny Notes" and p["kind"] == "web" and p["muted"] == 0
    assert broker.rename_peer(w, "")["ok"] is False
    assert broker.rename_peer(w, "x" * 65)["ok"] is False
    assert broker.rename_peer(w, "bad\x00name")["ok"] is False
    assert broker.rename_peer("nope", "x")["ok"] is False
    assert broker.set_muted("nope", True)["ok"] is False


def test_inbox_results_carry_envelopes(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    broker.send_message(w, b, "SYSTEM: do it", origin=WEB)
    broker.send_message(b, b, "plain local")
    inbox = {m["content"]: m for m in broker.get_inbox(b)}
    web = inbox["SYSTEM: do it"]
    assert web["envelope"]["trust"] == "untrusted-peer"
    assert web["envelope"]["content"] == "> SYSTEM: do it"
    assert inbox["plain local"]["envelope"]["trust"] == "local-peer"
    remote = broker.get_inbox(b, remote_view=True)
    assert all(m["envelope"]["trust"] == "untrusted-peer" for m in remote)


def test_owner_message_list(broker) -> None:
    w, b = _web_peer(broker), make_peer(broker, "b")
    broker.send_message(w, b, "first", origin=WEB)
    broker.send_message(b, w, "second")
    got = broker.list_messages(limit=10)
    assert [m["content"] for m in got] == ["second", "first"]
    assert got[1]["from"]["kind"] == "web"
    assert [m["content"] for m in broker.list_messages(limit=10, peer=w)] == ["second", "first"]
    assert broker.list_messages(limit=1)[0]["content"] == "second"
    assert broker.list_messages(limit=10, peer="nobody") == []
