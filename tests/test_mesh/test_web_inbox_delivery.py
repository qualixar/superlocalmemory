"""At-least-once delivery of a web app's mail: a dropped reply loses nothing (audit F8)."""

from __future__ import annotations

import threading

import pytest

from superlocalmemory.mesh import broker_web
from tests.test_mesh.conftest import make_peer, rows

WEB_REF = "w_" + "ef" * 12


@pytest.fixture()
def web(broker) -> str:
    assert broker.ensure_web_peer(WEB_REF, app="notes", display_name="Notes",
                                  connection_id="c" * 32)["ok"]
    return WEB_REF


@pytest.fixture()
def clock(monkeypatch):
    """A clock the test can move: the lease is measured against it."""
    now = [1_000_000.0]
    monkeypatch.setattr(broker_web, "_now", lambda: now[0], raising=False)
    return now


def _send(broker, web: str, text: str = "hello") -> int:
    return broker.send_message(make_peer(broker, "sess-" + text), web, text)["id"]


def test_a_claim_does_not_mark_the_message_read_before_it_is_acknowledged(broker, web, clock) -> None:
    mid = _send(broker, web)
    (msg,) = broker.claim_web_inbox(web)
    assert msg["id"] == mid and not msg.get("repeat")
    assert rows(broker, "SELECT read FROM mesh_messages WHERE id=?", (mid,))[0]["read"] == 0


def test_a_dropped_response_is_delivered_again_once_after_the_lease_flagged_repeat(
        broker, web, clock) -> None:
    mid = _send(broker, web)
    assert [m["id"] for m in broker.claim_web_inbox(web)] == [mid]
    assert broker.claim_web_inbox(web) == []  # still inside the lease: in flight
    clock[0] += broker_web.LEASE_S + 1
    (again,) = broker.claim_web_inbox(web)
    assert again["id"] == mid and again["repeat"] is True


def test_an_acknowledged_message_is_never_delivered_again(broker, web, clock) -> None:
    mid = _send(broker, web)
    broker.claim_web_inbox(web)
    clock[0] += broker_web.LEASE_S + 1
    assert broker.claim_web_inbox(web, ack=[mid]) == []
    clock[0] += broker_web.LEASE_S + 1
    assert broker.claim_web_inbox(web) == []
    assert rows(broker, "SELECT read FROM mesh_messages WHERE id=?", (mid,))[0]["read"] == 1


def test_an_ack_returns_the_next_fresh_message_in_the_same_call(broker, web, clock) -> None:
    first = _send(broker, web, "one")
    broker.claim_web_inbox(web)
    second = _send(broker, web, "two")
    got = broker.claim_web_inbox(web, ack=[first])
    assert [m["id"] for m in got] == [second] and not got[0].get("repeat")


def test_delivery_stops_after_the_attempt_limit(broker, web, clock) -> None:
    mid = _send(broker, web)
    seen = 0
    for _ in range(broker_web.MAX_DELIVERIES + 3):
        seen += len(broker.claim_web_inbox(web))
        clock[0] += broker_web.LEASE_S + 1
    assert seen == broker_web.MAX_DELIVERIES
    assert rows(broker, "SELECT read FROM mesh_messages WHERE id=?", (mid,))[0]["read"] == 1


def test_an_app_cannot_acknowledge_mail_that_is_not_its_own(broker, web, clock) -> None:
    other = "w_" + "12" * 12
    assert broker.ensure_web_peer(other, app="chat", display_name="Chat",
                                  connection_id="c" * 32)["ok"]
    mid = _send(broker, web)
    broker.claim_web_inbox(web)
    broker.claim_web_inbox(other, ack=[mid, "x", True, -1])
    clock[0] += broker_web.LEASE_S + 1
    assert [m["id"] for m in broker.claim_web_inbox(web)] == [mid]


def test_two_concurrent_readers_never_both_get_a_fresh_message(broker, web, clock) -> None:
    ids = {_send(broker, web, f"m{i}") for i in range(10)}
    results: list[list[int]] = []
    barrier = threading.Barrier(4)

    def reader() -> None:
        barrier.wait()
        results.append([m["id"] for m in broker.claim_web_inbox(web)])

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [i for r in results for i in r]
    assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))


def test_wait_returns_a_message_whose_lease_ran_out_and_honours_ack(broker, web, clock) -> None:
    mid = _send(broker, web)
    broker.claim_web_inbox(web)
    got, timed_out = broker.wait_web_inbox(web, timeout_s=1)
    assert got == [] and timed_out is True  # in flight, nothing to hand out
    clock[0] += broker_web.LEASE_S + 1
    got, timed_out = broker.wait_web_inbox(web, timeout_s=1)
    assert [m["id"] for m in got] == [mid] and got[0]["repeat"] is True
    clock[0] += broker_web.LEASE_S + 1
    got, _ = broker.wait_web_inbox(web, timeout_s=1, ack=[mid])
    assert got == []


def test_local_agent_inbox_is_unchanged(broker) -> None:
    from superlocalmemory.mesh import broker_inbox
    a, b = make_peer(broker, "a"), make_peer(broker, "b")
    mid = broker.send_message(a, b, "hi")["id"]
    conn = broker._conn()
    try:
        first = broker_inbox.query_inbox(conn, b, "", "default")
        again = broker_inbox.query_inbox(conn, b, "", "default")
    finally:
        conn.close()
    assert [m["id"] for m in first] == [mid] == [m["id"] for m in again]
    assert "repeat" not in first[0]
    assert broker.mark_read(b, [mid]) is not None


def test_cleanup_drops_lease_rows_of_removed_messages(broker, web, clock) -> None:
    mid = _send(broker, web)
    broker.claim_web_inbox(web)
    conn = broker._conn()
    try:
        conn.execute("DELETE FROM mesh_messages WHERE id=?", (mid,))
        from superlocalmemory.mesh import broker_cleanup
        broker_cleanup.run_cleanup(conn)
    finally:
        conn.close()
    assert rows(broker, "SELECT * FROM mesh_web_deliveries") == []
