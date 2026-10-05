# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.recall_queue."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _imports():
    from superlocalmemory.core import recall_queue as rq
    return rq


def _make_queue(tmp_path: Path):
    return _imports().RecallQueue(db_path=tmp_path / "q.db")


# -----------------------------------------------------------------------
# Schema + enqueue
# -----------------------------------------------------------------------

def test_schema_created_on_init(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    q.close()
    # Re-open — must not re-error on schema create
    q2 = _make_queue(tmp_path)
    q2.close()


def test_enqueue_returns_request_id(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="hello", limit_n=10, mode="B", agent_id="a", session_id="s")
    assert isinstance(rid, str)
    assert len(rid) > 0
    q.close()


def test_enqueue_dedup_hit_returns_existing_id_and_bumps_subscriber(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid1 = q.enqueue(query="same", limit_n=10, mode="B", agent_id="a", session_id="s")
    rid2 = q.enqueue(query="same", limit_n=10, mode="B", agent_id="a", session_id="s")
    assert rid1 == rid2
    # subscriber_count should be 2
    row = q._get_row(rid1)
    assert row["subscriber_count"] == 2
    q.close()


def test_enqueue_different_agent_is_not_dedup(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    r1 = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    r2 = q.enqueue(query="x", limit_n=10, mode="B", agent_id="b", session_id="s")
    assert r1 != r2
    q.close()


def test_enqueue_tenant_id_in_dedup(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    r1 = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s", tenant_id="T1")
    r2 = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s", tenant_id="T2")
    assert r1 != r2
    q.close()


# -----------------------------------------------------------------------
# Claim + complete (fenced)
# -----------------------------------------------------------------------

def test_claim_returns_pending_row(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    claim = q.claim_pending(priority="high", stall_timeout_s=25.0)
    assert claim is not None
    assert claim["request_id"] == rid
    assert claim["received"] == 1
    q.close()


def test_claim_returns_none_when_empty(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    assert q.claim_pending(priority="high", stall_timeout_s=25.0) is None
    q.close()


def test_claim_twice_sees_updated_received(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    c1 = q.claim_pending(priority="high", stall_timeout_s=0.01)
    assert c1["received"] == 1
    time.sleep(0.02)
    c2 = q.claim_pending(priority="high", stall_timeout_s=0.01)
    assert c2 is not None
    assert c2["request_id"] == rid
    assert c2["received"] == 2
    q.close()


def test_complete_with_correct_received_writes_result(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    claim = q.claim_pending(priority="high", stall_timeout_s=25.0)
    n = q.complete(rid, received=claim["received"],
                   result_json=json.dumps({"ok": True}))
    assert n == 1
    row = q._get_row(rid)
    assert row["completed"] == 1
    assert row["result_json"] == json.dumps({"ok": True})
    q.close()


def test_complete_with_stale_received_is_fenced_out(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    q.claim_pending(priority="high", stall_timeout_s=0.01)
    time.sleep(0.02)
    q.claim_pending(priority="high", stall_timeout_s=25.0)  # received=2 now
    # Stale worker tries to complete with received=1
    n = q.complete(rid, received=1, result_json=json.dumps({"stale": True}))
    assert n == 0, "Stale write must be fenced out"
    row = q._get_row(rid)
    assert row["completed"] == 0
    assert row["result_json"] is None
    q.close()


def test_complete_on_cancelled_row_is_fenced(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    claim = q.claim_pending(priority="high", stall_timeout_s=25.0)
    q._force_cancelled(rid)
    n = q.complete(rid, received=claim["received"], result_json="{}")
    assert n == 0
    q.close()


def test_complete_on_dlq_row_is_fenced(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    claim = q.claim_pending(priority="high", stall_timeout_s=25.0)
    q.mark_dead_letter(rid, reason="max_receives_exceeded")
    n = q.complete(rid, received=claim["received"], result_json="{}")
    assert n == 0
    q.close()


# -----------------------------------------------------------------------
# DLQ fast-fail polling (Invariant 12)
# -----------------------------------------------------------------------

def test_poll_result_returns_completed(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    claim = q.claim_pending(priority="high", stall_timeout_s=25.0)
    q.complete(rid, received=claim["received"], result_json=json.dumps({"hits": 3}))
    payload = q.poll_result(rid, timeout_s=1.0)
    assert payload == {"hits": 3}
    q.close()


def test_poll_result_raises_dead_letter_fast(tmp_path: Path, monkeypatch) -> None:
    rq = _imports()
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    q.claim_pending(priority="high", stall_timeout_s=25.0)
    q.mark_dead_letter(rid, reason="max_receives_exceeded")
    # Fast-fail means raising on the first read, before any poll sleep. Count
    # the sleeps instead of timing them: the count does not drift under load.
    slept: list[float] = []
    monkeypatch.setattr(rq, "time", SimpleNamespace(
        time=time.time, monotonic=time.monotonic, sleep=slept.append))
    with pytest.raises(rq.DeadLetterError) as exc:
        q.poll_result(rid, timeout_s=5.0)
    assert slept == [], f"DLQ did not fast-fail; slept {slept} on the poll schedule"
    assert exc.value.request_id == rid
    assert "max_receives" in exc.value.reason
    q.close()


def test_poll_result_raises_timeout(tmp_path: Path) -> None:
    rq = _imports()
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    with pytest.raises(rq.QueueTimeoutError):
        q.poll_result(rid, timeout_s=0.2)
    q.close()


def test_poll_result_raises_cancelled(tmp_path: Path) -> None:
    rq = _imports()
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    q._force_cancelled(rid)
    with pytest.raises(rq.QueueCancelledError):
        q.poll_result(rid, timeout_s=1.0)
    q.close()


# -----------------------------------------------------------------------
# State-legality CHECK constraint (Invariant 9)
# -----------------------------------------------------------------------

def test_check_constraint_forbids_completed_and_cancelled(tmp_path: Path) -> None:
    import sqlite3
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    # Direct SQL — try to violate state legality
    with pytest.raises(sqlite3.IntegrityError):
        q._raw_execute(
            "UPDATE recall_requests SET completed=1, cancelled=1, "
            "result_json='{}' WHERE request_id=?",
            (rid,),
        )
    q.close()


def test_completed_without_result_json_forbidden(tmp_path: Path) -> None:
    import sqlite3
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    with pytest.raises(sqlite3.IntegrityError):
        q._raw_execute(
            "UPDATE recall_requests SET completed=1, result_json=NULL "
            "WHERE request_id=?",
            (rid,),
        )
    q.close()


# -----------------------------------------------------------------------
# Idempotency unique constraint (Invariant 11)
# -----------------------------------------------------------------------

def test_idempotency_key_unique_per_job_type(tmp_path: Path) -> None:
    rq = _imports()
    q = _make_queue(tmp_path)
    r1 = q.enqueue_job(
        job_type="consolidate", idempotency_key="mem-1:ep-5",
        agent_id="worker", session_id="consolidate",
    )
    r2 = q.enqueue_job(
        job_type="consolidate", idempotency_key="mem-1:ep-5",
        agent_id="worker", session_id="consolidate",
    )
    assert r1 == r2, "Same idempotency_key must return existing request"
    q.close()


# -----------------------------------------------------------------------
# Cancel / subscriber_count
# -----------------------------------------------------------------------

def test_unsubscribe_decrements_count(tmp_path: Path) -> None:
    q = _make_queue(tmp_path)
    rid = q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")
    q.enqueue(query="x", limit_n=10, mode="B", agent_id="a", session_id="s")  # dedup hit
    assert q._get_row(rid)["subscriber_count"] == 2
    q.unsubscribe(rid)
    assert q._get_row(rid)["subscriber_count"] == 1
    q.unsubscribe(rid)
    assert q._get_row(rid)["subscriber_count"] == 0
    q.close()
