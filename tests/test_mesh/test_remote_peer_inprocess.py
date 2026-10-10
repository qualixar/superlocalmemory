"""Web apps on the mesh, served in process: identity, limits, cleanup and revoke."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from superlocalmemory.mesh.broker_profiles import SendRateLimiter
from tests.test_mesh.conftest import make_peer, rows

CID = "c" * 32


def _web(broker, ref, app="notes", name="", cid=CID, profile="default") -> str:
    res = broker.ensure_web_peer(ref, app=app, display_name=name or "Web app " + ref[2:8],
                                 connection_id=cid, profile_id=profile)
    assert res["ok"], res
    return res["peer_id"]


def _ref(n: int) -> str:
    return f"w_{n:024x}"


# -- identity and registration ------------------------------------------------


def test_a_web_peer_uses_its_reference_as_its_id_and_registering_again_reuses_the_row(broker) -> None:
    ref = _ref(1)
    assert _web(broker, ref) == ref
    first = rows(broker, "SELECT * FROM mesh_peers WHERE peer_id=?", (ref,))
    assert _web(broker, ref) == ref
    again = rows(broker, "SELECT * FROM mesh_peers WHERE peer_id=?", (ref,))
    assert len(first) == len(again) == 1
    assert first[0]["registered_at"] == again[0]["registered_at"]
    prof = rows(broker, "SELECT * FROM mesh_peer_profiles WHERE peer_id=?", (ref,))[0]
    assert prof["kind"] == "web" and prof["app_name"] == "notes"
    assert broker.is_web_peer(ref) and not broker.is_web_peer("nobody")


def test_the_owners_rename_wins_over_the_cached_app_name(broker) -> None:
    ref = _ref(2)
    _web(broker, ref)                       # fallback name stored
    _web(broker, ref, name="ChatGPT")       # real name arrives later
    assert rows(broker, "SELECT display_name FROM mesh_peer_profiles WHERE peer_id=?",
                (ref,))[0][0] == "ChatGPT"
    assert broker.rename_peer(ref, "Mine")["ok"]
    _web(broker, ref, name="ChatGPT 2")
    assert rows(broker, "SELECT display_name FROM mesh_peer_profiles WHERE peer_id=?",
                (ref,))[0][0] == "Mine"


def test_a_retired_web_peer_cannot_register_again(broker) -> None:
    ref = _ref(3)
    _web(broker, ref)
    assert broker.retire_peer(ref)["ok"]
    res = broker.ensure_web_peer(ref, app="notes", display_name="x", connection_id=CID,
                                 profile_id="default")
    assert res["ok"] is False and "retired" in res["error"]


def test_a_web_peer_is_listed_with_its_kind_and_without_host_details(broker) -> None:
    local = make_peer(broker, "sess-1", "claude_code")
    ref = _web(broker, _ref(4), name="ChatGPT")
    listing = {p["peer_id"]: p for p in broker.list_peer_directory("default")}
    assert listing[ref]["kind"] == "web" and listing[ref]["name"] == "ChatGPT"
    assert listing[local]["kind"] == "local"
    for entry in listing.values():
        assert not {"host", "port", "project_path", "session_id"} & set(entry)


# -- cleanup ------------------------------------------------------------------


def test_cleanup_never_reaps_a_web_peer_but_still_reaps_a_local_one(broker) -> None:
    local = make_peer(broker, "sess-old")
    ref = _web(broker, _ref(5))
    old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    conn = broker._conn()
    conn.execute("UPDATE mesh_peers SET last_heartbeat=?", (old,))
    conn.commit()
    conn.close()
    broker._run_cleanup()
    broker._run_cleanup()
    ids = {r["peer_id"] for r in rows(broker, "SELECT peer_id FROM mesh_peers")}
    assert ref in ids and local not in ids
    assert rows(broker, "SELECT status FROM mesh_peers WHERE peer_id=?", (ref,))[0][0] == "active"


# -- sending and reading ------------------------------------------------------


def test_a_web_send_reaches_a_local_peer_marked_as_web_with_its_app(broker) -> None:
    local, ref = make_peer(broker, "sess-1"), _web(broker, _ref(6), app="app-a")
    res = broker.web_send(ref, "app-a", local, "hello there", profile_id="default")
    assert res["ok"], res
    msg = broker.get_inbox(local, "", "default")[0]
    assert msg["envelope"]["from"] == {"peer_id": ref, "app": "app-a", "kind": "web"}
    assert msg["envelope"]["trust"] == "untrusted-peer"


def test_one_web_app_messages_another_and_only_the_reader_gets_it_once(broker) -> None:
    a, b = _web(broker, _ref(7), app="app-a"), _web(broker, _ref(8), app="app-b")
    assert broker.web_send(a, "app-a", b, "ping", profile_id="default")["ok"]
    got = broker.claim_web_inbox(b, "default")
    assert [m["envelope"]["from"]["app"] for m in got] == ["app-a"]
    assert got[0]["envelope"]["from"]["kind"] == "web"
    assert broker.claim_web_inbox(b, "default") == []     # in flight: no second reader gets it
    broker.claim_web_inbox(b, "default", ack=[got[0]["id"]])
    assert rows(broker, "SELECT read FROM mesh_messages")[0][0] == 1


@pytest.mark.parametrize("target", ["broadcast", "project:/tmp/work"])
def test_a_web_sender_may_only_address_one_peer(broker, target) -> None:
    ref = _web(broker, _ref(9))
    res = broker.web_send(ref, "notes", target, "to all", profile_id="default")
    assert res["ok"] is False
    assert rows(broker, "SELECT * FROM mesh_messages") == []


def test_a_web_peer_never_sees_broadcast_or_project_messages(broker) -> None:
    local, ref = make_peer(broker, "sess-1"), _web(broker, _ref(10))
    assert broker.send_message(local, "broadcast", "for everyone")["ok"]
    assert broker.send_message(local, ref, "just you")["ok"]
    got = broker.claim_web_inbox(ref, "default")
    assert [m["content"] for m in got] == ["just you"]
    assert broker.wait_web_inbox(ref, timeout_s=1, profile_id="default") == ([], True)


def test_a_web_app_cannot_address_a_peer_of_another_computer(broker) -> None:
    ref = _web(broker, _ref(11))
    broker.add_remote_peer("far-away", {"profile_id": "default"})
    res = broker.web_send(ref, "notes", "far-away", "hi", profile_id="default")
    assert res == {"ok": False, "error": "recipient peer not found"}


def test_a_message_over_the_limit_in_utf8_bytes_is_refused(broker) -> None:
    local, ref = make_peer(broker, "sess-1"), _web(broker, _ref(12))
    text = "é" * 2100                     # 2100 characters, 4200 bytes
    assert broker.web_send(ref, "notes", local, text, profile_id="default")["ok"] is False
    assert broker.send_message(local, ref, text)["ok"] is False
    assert broker.web_send(ref, "notes", local, "é" * 2048, profile_id="default")["ok"]


# -- unread cap ---------------------------------------------------------------


def test_the_fifty_first_unread_direct_message_is_refused_and_nothing_is_evicted(broker) -> None:
    local = make_peer(broker, "sess-1")
    senders = [_web(broker, _ref(20 + i), app=f"app-{i}") for i in range(3)]
    broker._send_limiter = SendRateLimiter(limit=1000)
    for n in range(50):
        res = broker.web_send(senders[n % 3], "x", local, f"m{n}", profile_id="default")
        assert res["ok"], (n, res)
    res = broker.web_send(senders[0], "x", local, "one too many", profile_id="default")
    assert res == {"ok": False, "error": "recipient inbox is full"}
    stored = rows(broker, "SELECT content FROM mesh_messages ORDER BY id")
    assert len(stored) == 50 and stored[0]["content"] == "m0"


def test_a_local_sender_is_not_held_to_the_unread_cap(broker) -> None:
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    for n in range(55):
        assert broker.send_message(a, b, f"m{n}")["ok"]


def test_the_cap_frees_up_once_the_recipient_reads(broker) -> None:
    ref = _web(broker, _ref(30))
    other = _web(broker, _ref(31))
    broker._send_limiter = SendRateLimiter(limit=1000)
    for n in range(50):
        assert broker.web_send(other, "x", ref, f"m{n}", profile_id="default")["ok"]
    first = broker.claim_web_inbox(ref, "default")
    assert len(first) == 20                                          # one claim is capped
    assert broker.web_send(other, "x", ref, "again", profile_id="default")["ok"] is False
    broker.claim_web_inbox(ref, "default", ack=[m["id"] for m in first])  # reading = acknowledging
    assert broker.web_send(other, "x", ref, "again", profile_id="default")["ok"]


# -- revoke -------------------------------------------------------------------


def test_revoke_sync_retires_unlisted_web_peers_of_that_connection_only(broker) -> None:
    a, b = _web(broker, _ref(40), app="a"), _web(broker, _ref(41), app="b")
    other = _web(broker, _ref(42), app="c", cid="d" * 32)
    local = make_peer(broker, "sess-1")
    assert broker.web_send(a, "a", local, "from a", profile_id="default")["ok"]
    assert broker.web_send(b, "b", a, "to a", profile_id="default")["ok"]
    retired = broker.retire_missing_web_peers(CID, {b})
    assert retired == [a]
    ids = {r["peer_id"] for r in rows(broker, "SELECT peer_id FROM mesh_peers")}
    assert a not in ids and {b, other, local} <= ids
    assert rows(broker, "SELECT * FROM mesh_messages") == []
    assert broker.retire_missing_web_peers(CID, {b}) == []


# -- concurrent waits ---------------------------------------------------------


def test_two_waits_for_the_same_peer_never_return_the_same_message(broker) -> None:
    local, ref = make_peer(broker, "sess-1"), _web(broker, _ref(50))
    results: list[list[dict]] = []
    gate = threading.Barrier(3)

    def waiter() -> None:
        gate.wait()
        msgs, _ = broker.wait_web_inbox(ref, timeout_s=2, profile_id="default")
        results.append(msgs)

    threads = [threading.Thread(target=waiter) for _ in range(2)]
    for t in threads:
        t.start()
    gate.wait()
    time.sleep(0.2)
    assert broker.send_message(local, ref, "only once")["ok"]
    for t in threads:
        t.join(5)
    delivered = [m["id"] for batch in results for m in batch]
    assert len(delivered) == 1


def test_many_concurrent_claims_split_the_messages_without_overlap(broker) -> None:
    local, ref = make_peer(broker, "sess-1"), _web(broker, _ref(51))
    for n in range(30):
        assert broker.send_message(local, ref, f"m{n}")["ok"]
    claimed: list[int] = []
    lock = threading.Lock()

    def claim() -> None:
        for _ in range(5):
            got = broker.claim_web_inbox(ref, "default")
            with lock:
                claimed.extend(m["id"] for m in got)

    threads = [threading.Thread(target=claim) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert len(claimed) == len(set(claimed)) == 30


def test_revoke_sync_leaves_a_peer_registered_after_the_list_was_read(broker) -> None:
    old = _web(broker, _ref(60), app="a")
    cutoff = datetime.now(timezone.utc).isoformat()
    new = _web(broker, _ref(61), app="b")
    assert broker.retire_missing_web_peers(CID, set(), registered_before=cutoff) == [old]
    ids = {r["peer_id"] for r in rows(broker, "SELECT peer_id FROM mesh_peers")}
    assert new in ids and old not in ids


def test_concurrent_web_senders_cannot_pass_the_unread_cap(broker) -> None:
    local = make_peer(broker, "sess-1")
    senders = [_web(broker, _ref(70 + i), app=f"app-{i}") for i in range(8)]
    outcomes: list[bool] = []
    lock = threading.Lock()

    def send_ten(ref: str) -> None:
        for n in range(10):
            res = broker.web_send(ref, "x", local, f"m{n}", profile_id="default")
            with lock:
                outcomes.append(bool(res.get("ok")))

    threads = [threading.Thread(target=send_ten, args=(ref,)) for ref in senders]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sum(outcomes) == 50 and len(outcomes) == 80
    assert len(rows(broker, "SELECT id FROM mesh_messages")) == 50
