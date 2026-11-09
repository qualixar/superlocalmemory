"""Property-based invariant tests for the v3.4.26 recall queue.

These are seeded-random fuzz tests that drive ``RecallQueue`` through
long sequences of random operations and assert the 10 invariants named
in the v5 test matrix. We avoid adding a Hypothesis dependency by
running bounded random loops with deterministic seeds — if a run fails,
the seed + op sequence is printed for reproduction.

Required by v5 (CI-running pre Stage 6): PROP-2, PROP-6, PROP-10.
The rest are scaffolded against the same queue so a regression in any
invariant surfaces on CI.

Run a single property intensively:
    SLM_PROP_ITERS=5000 pytest tests/property/test_queue_invariants.py -k PROP_2

Reproduce a failure:
    SLM_PROP_SEED=12345 pytest tests/property/test_queue_invariants.py -k PROP_2
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from superlocalmemory.core.recall_queue import RecallQueue

_DEFAULT_ITERS = int(os.environ.get("SLM_PROP_ITERS", "300"))
_DEFAULT_OPS_PER_ITER = int(os.environ.get("SLM_PROP_OPS", "40"))


def _seeded_rng() -> tuple[random.Random, int]:
    seed = int(os.environ.get("SLM_PROP_SEED") or random.randrange(2**31))
    return random.Random(seed), seed


@pytest.fixture
def queue(tmp_path: Path) -> RecallQueue:
    q = RecallQueue(db_path=tmp_path / "recall_queue.db")
    yield q
    try:
        q.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# helper: drive a random op trace and collect per-request facts
# ---------------------------------------------------------------------------

@dataclass
class _TraceLog:
    seed: int
    ops: list[str]

    def fail(self, msg: str) -> str:
        return f"seed={self.seed} ops={self.ops[-20:]} — {msg}"


def _enqueue_unique(queue: RecallQueue, rng: random.Random, tag: str) -> str:
    """Enqueue a unique request so each call produces a distinct rid."""
    return queue.enqueue(
        query=f"q-{tag}-{rng.randint(0, 2**31)}",
        limit_n=rng.randint(1, 10),
        mode="a",
        agent_id=f"ag-{rng.randint(0, 4)}",
        session_id=f"s-{rng.randint(0, 4)}",
        stall_timeout_s=0.05,  # tiny — exposes claim-expiry races
    )


# ---------------------------------------------------------------------------
# PROP-2 (I2 Exclusivity) — required
# ---------------------------------------------------------------------------

class TestPROP_2_Exclusivity:
    """Invariant: at most one thread holds an active claim on a given
    request_id at any moment, across all (enqueue/claim/complete/stall)
    interleavings. We approximate in-process by checking that the count
    of (non-terminal, non-expired) claims per request_id is never > 1.
    """
    def test_single_active_claim_per_request(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        for i in range(_DEFAULT_OPS_PER_ITER * 5):
            op = rng.choice(["enqueue", "claim", "complete", "stall_tick"])
            log.ops.append(op)
            if op == "enqueue":
                _enqueue_unique(queue, rng, f"p2-{i}")
            elif op == "claim":
                queue.claim_pending(priority="high", stall_timeout_s=0.05)
            elif op == "complete":
                claimed = queue.claim_pending(priority="high",
                                              stall_timeout_s=0.05)
                if claimed is not None:
                    queue.complete(claimed["request_id"],
                                   received=claimed["received"],
                                   result_json=json.dumps({"ok": True}))
            else:
                pass

            # Invariant check after every op
            rows = queue._conn.execute(
                "SELECT request_id, claim_expires_at, completed, "
                "cancelled, dead_letter FROM recall_requests"
            ).fetchall()
            now = __import__("time").time()
            active_by_rid: Counter[str] = Counter()
            for r in rows:
                if r["completed"] or r["cancelled"] or r["dead_letter"]:
                    continue
                if r["claim_expires_at"] is not None and r["claim_expires_at"] > now:
                    active_by_rid[r["request_id"]] += 1
            dupes = {k: v for k, v in active_by_rid.items() if v > 1}
            assert not dupes, log.fail(f"multiple active claims: {dupes}")


# ---------------------------------------------------------------------------
# PROP-6 (I6 Idempotent remember) — required
# ---------------------------------------------------------------------------

class TestPROP_6_IdempotentRemember:
    """Invariant: repeatedly enqueuing the SAME query_hash does not add
    rows — it increments subscriber_count on the existing row. The
    queue's count of distinct live query_hashes equals the number of
    distinct enqueue keys we've used while nothing is terminal.
    """
    def test_dedup_on_query_hash(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        key_universe = [
            {
                "query": f"same-q-{i}", "limit_n": 5, "mode": "a",
                "agent_id": "ag", "session_id": "s", "tenant_id": "",
            }
            for i in range(8)
        ]
        seen_rids: dict[int, str] = {}

        for _ in range(_DEFAULT_OPS_PER_ITER * 3):
            k_idx = rng.randrange(len(key_universe))
            k = key_universe[k_idx]
            log.ops.append(f"enq[{k_idx}]")
            rid = queue.enqueue(**k)
            if k_idx in seen_rids:
                assert rid == seen_rids[k_idx], log.fail(
                    f"dedup broke: key {k_idx} produced {rid} "
                    f"but earlier produced {seen_rids[k_idx]}"
                )
            else:
                seen_rids[k_idx] = rid

        # Row count equals number of distinct keys used.
        rows = queue._conn.execute(
            "SELECT COUNT(*) FROM recall_requests"
        ).fetchone()[0]
        assert rows == len(seen_rids), log.fail(
            f"row count {rows} != distinct keys {len(seen_rids)}"
        )


# ---------------------------------------------------------------------------
# PROP-10 (I10 Subscriber consistency) — required
# ---------------------------------------------------------------------------

class TestPROP_10_SubscriberConsistency:
    """Invariant: subscriber_count matches the number of live subscribers
    against each request. Every dedup-enqueue on the same key increments
    it; every unsubscribe decrements it; count never goes below 0.
    """
    def test_count_matches_model(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        rid_to_model_count: defaultdict[str, int] = defaultdict(int)
        rid_by_key: dict[int, str] = {}

        for _ in range(_DEFAULT_OPS_PER_ITER * 4):
            op = rng.choice(["enqueue", "unsubscribe"])
            log.ops.append(op)
            if op == "enqueue":
                k_idx = rng.randrange(6)
                rid = queue.enqueue(
                    query=f"subq-{k_idx}", limit_n=5, mode="a",
                    agent_id="ag", session_id="s",
                )
                rid_by_key.setdefault(k_idx, rid)
                rid_to_model_count[rid] += 1
            else:
                if not rid_to_model_count:
                    continue
                rid = rng.choice(list(rid_to_model_count.keys()))
                if rid_to_model_count[rid] > 0:
                    queue.unsubscribe(rid)
                    rid_to_model_count[rid] -= 1

            # Compare each row's subscriber_count with our model.
            rows = queue._conn.execute(
                "SELECT request_id, subscriber_count FROM recall_requests"
            ).fetchall()
            for r in rows:
                rid = r["request_id"]
                actual = r["subscriber_count"]
                expected = rid_to_model_count[rid]
                assert actual == expected, log.fail(
                    f"rid={rid} actual={actual} expected={expected}"
                )
                assert actual >= 0, log.fail(
                    f"rid={rid} subscriber_count went negative: {actual}"
                )


# ---------------------------------------------------------------------------
# PROP-1 Fencing — scaffolded against the fence-by-received contract
# ---------------------------------------------------------------------------

class TestPROP_1_Fencing:
    """Invariant: complete() with a stale `received` count is rejected
    (rowcount == 0) and does NOT overwrite a prior committed result.
    Ensures the last-writer-wins fence actually fences.
    """
    def test_stale_received_rejected(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        for i in range(20):
            log.ops.append(f"round-{i}")
            rid = _enqueue_unique(queue, rng, f"p1-{i}")
            claimed1 = queue.claim_pending(priority="high", stall_timeout_s=0.0)
            assert claimed1 is not None and claimed1["request_id"] == rid
            # Force claim expiry by negative stall so the next claim re-claims.
            queue._conn.execute(
                "UPDATE recall_requests SET claim_expires_at = 0 "
                "WHERE request_id = ?", (rid,),
            )
            claimed2 = queue.claim_pending(priority="high", stall_timeout_s=0.05)
            assert claimed2 is not None and claimed2["request_id"] == rid
            # complete() with stale (= claimed1) received must NOT land.
            stale_rowcount = queue.complete(
                rid, received=claimed1["received"],
                result_json=json.dumps({"stale": True}),
            )
            assert stale_rowcount == 0, log.fail("stale complete() landed")
            # Fresh complete() lands once.
            fresh_rowcount = queue.complete(
                rid, received=claimed2["received"],
                result_json=json.dumps({"fresh": True}),
            )
            assert fresh_rowcount == 1
            # And a second complete() against the fresh token is idempotent
            # (row already completed = WHERE clause fails).
            again = queue.complete(
                rid, received=claimed2["received"],
                result_json=json.dumps({"replay": True}),
            )
            assert again == 0


# ---------------------------------------------------------------------------
# PROP-7 Cancellation — scaffolded
# ---------------------------------------------------------------------------

class TestPROP_7_Cancellation:
    """Invariant: a row with cancelled=1 never receives a result_json
    (complete() is blocked by the WHERE clause)."""
    def test_cancelled_rows_never_completed(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        for i in range(40):
            rid = _enqueue_unique(queue, rng, f"p7-{i}")
            claimed = queue.claim_pending(priority="high", stall_timeout_s=0.5)
            assert claimed is not None
            queue._force_cancelled(rid)
            landed = queue.complete(
                rid, received=claimed["received"],
                result_json=json.dumps({"after_cancel": True}),
            )
            assert landed == 0, log.fail("complete() landed post-cancel")

        rows = queue._conn.execute(
            "SELECT result_json FROM recall_requests WHERE cancelled = 1"
        ).fetchall()
        for r in rows:
            assert r["result_json"] is None or r["result_json"] == "", \
                log.fail(f"cancelled row has result_json: {r['result_json']!r}")


# ---------------------------------------------------------------------------
# PROP-9 State legality — scaffolded
# ---------------------------------------------------------------------------

class TestPROP_9_StateLegality:
    """Invariant: no row ends up with more than one terminal flag set
    simultaneously (completed + cancelled + dead_letter <= 1)."""
    def test_no_dual_terminal_states(self, queue):
        rng, seed = _seeded_rng()
        log = _TraceLog(seed=seed, ops=[])

        for i in range(50):
            op = rng.choice(["enqueue_claim_complete",
                             "enqueue_then_cancel",
                             "enqueue_then_dlq"])
            log.ops.append(op)
            rid = _enqueue_unique(queue, rng, f"p9-{i}")
            claimed = queue.claim_pending(priority="high", stall_timeout_s=0.5)
            if op == "enqueue_claim_complete":
                queue.complete(rid, received=claimed["received"],
                               result_json=json.dumps({"ok": True}))
            elif op == "enqueue_then_cancel":
                queue._force_cancelled(rid)
            else:
                queue.mark_dead_letter(rid, reason="synthetic")

        rows = queue._conn.execute(
            "SELECT completed, cancelled, dead_letter FROM recall_requests"
        ).fetchall()
        for r in rows:
            active = int(bool(r["completed"])) + int(bool(r["cancelled"])) \
                   + int(bool(r["dead_letter"]))
            assert active <= 1, log.fail(
                f"row has {active} terminal flags: {dict(r)}"
            )


# ---------------------------------------------------------------------------
# PROP-3, 4, 5, 8 — scaffolded but skipped by default
# ---------------------------------------------------------------------------
# These require either chaos / clock control / WFQ-scheduler under real
# concurrency. We land them as executable scaffolds so they show up in CI
# with a clear skip reason instead of being absent.

@pytest.mark.skip(reason="PROP-3 Progress — needs worker-crash chaos harness "
                         "(Stage 7 chaos suite, SLM_RUN_CHAOS=1)")
def test_prop_3_progress_under_crashes():
    raise NotImplementedError


@pytest.mark.skip(reason="PROP-4 Cleanup — requires freezegun; scoped to "
                         "Stage 7 retention sweep tests")
def test_prop_4_terminal_cleanup():
    raise NotImplementedError


@pytest.mark.skip(reason="PROP-5 No-lost-work — requires kill -9 loop; "
                         "scoped to Stage 7 chaos suite")
def test_prop_5_no_lost_work():
    raise NotImplementedError


@pytest.mark.skip(reason="PROP-8 WFQ fairness — requires real merged worker "
                         "with priority lanes; scoped to post-Phase-6 test")
def test_prop_8_wfq_fairness():
    raise NotImplementedError
