# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for core.rate_limit."""

from __future__ import annotations

import time

import pytest


def _imports():
    from superlocalmemory.core import rate_limit as rl
    return rl


def test_token_bucket_admits_burst_up_to_capacity():
    rl = _imports()
    b = rl.TokenBucket(rate_per_sec=10, capacity=10)
    admitted = sum(1 for _ in range(10) if b.try_consume())
    assert admitted == 10
    assert not b.try_consume(), "11th should be rejected"


def test_token_bucket_refills_over_time():
    rl = _imports()
    b = rl.TokenBucket(rate_per_sec=100, capacity=10)
    for _ in range(10):
        assert b.try_consume()
    assert not b.try_consume()
    time.sleep(0.11)  # ~11 tokens generated
    admitted = sum(1 for _ in range(10) if b.try_consume())
    assert admitted >= 5, f"Expected refill, got {admitted}"


def test_layered_admits_when_all_have_tokens():
    rl = _imports()
    lim = rl.LayeredRateLimiter(
        global_rps=100, per_pid_rps=30, per_agent_rps=10,
    )
    lim.check_and_consume(pid=1000, agent_id="a")  # should not raise


def test_layered_rejects_when_global_drained():
    rl = _imports()
    lim = rl.LayeredRateLimiter(
        global_rps=1, per_pid_rps=10, per_agent_rps=10,
    )
    lim.check_and_consume(pid=1000, agent_id="a")
    with pytest.raises(rl.RateLimitedError) as exc:
        lim.check_and_consume(pid=1000, agent_id="a")
    assert exc.value.layer == "global"


def test_layered_peek_then_consume_does_not_burn_outer_on_inner_reject():
    # Canonical HTB: a rejection on an inner bucket must NOT drain the
    # outer bucket. Before the fix, 20 rejected calls on agent A would
    # each have drained 1 global token, starving clean agent B.
    rl = _imports()
    lim = rl.LayeredRateLimiter(
        global_rps=100, per_pid_rps=100, per_agent_rps=5,
    )
    # Agent A drains its 5 tokens
    for _ in range(5):
        lim.check_and_consume(pid=1, agent_id="a")
    # Rejection burst must not touch global
    for _ in range(20):
        with pytest.raises(rl.RateLimitedError) as exc:
            lim.check_and_consume(pid=1, agent_id="a")
        assert exc.value.layer == "per-agent"
    # Agent B (clean, different agent) must still be admitted.
    # If global was burned, agent B would fail with layer="global".
    for _ in range(5):
        lim.check_and_consume(pid=2, agent_id="b")


def test_retry_after_ms_populated_on_reject():
    rl = _imports()
    lim = rl.LayeredRateLimiter(global_rps=1, per_pid_rps=10, per_agent_rps=10)
    lim.check_and_consume(pid=1, agent_id="a")
    with pytest.raises(rl.RateLimitedError) as exc:
        lim.check_and_consume(pid=1, agent_id="a")
    assert exc.value.retry_after_ms > 0
    assert exc.value.retry_after_ms < 2000  # sanity


def test_per_pid_dict_ttl_eviction():
    rl = _imports()
    lim = rl.LayeredRateLimiter(
        global_rps=1000, per_pid_rps=30, per_agent_rps=1000, idle_ttl_s=0.05,
    )
    for pid in range(50):
        lim.check_and_consume(pid=pid, agent_id=None)
    assert lim.per_pid_size() >= 40  # all fresh
    time.sleep(0.2)
    lim.check_and_consume(pid=99999, agent_id=None)  # triggers sweep
    assert lim.per_pid_size() <= 5, (
        f"Expected TTL eviction; per-pid dict is {lim.per_pid_size()}"
    )


def test_per_agent_dict_ttl_eviction_does_not_raise():
    rl = _imports()
    lim = rl.LayeredRateLimiter(
        global_rps=1000, per_pid_rps=1000, per_agent_rps=1000, idle_ttl_s=1,
    )
    lim.check_and_consume(pid=1, agent_id="expired-agent")
    lim._per_agent["expired-agent"].last_used = 0

    lim._sweep(time.monotonic())

    assert "expired-agent" not in lim._per_agent


def test_rate_limited_error_carries_envelope_code():
    rl = _imports()
    from superlocalmemory.core.error_envelope import ErrorCode
    lim = rl.LayeredRateLimiter(global_rps=1, per_pid_rps=10, per_agent_rps=10)
    lim.check_and_consume(pid=1, agent_id=None)
    with pytest.raises(rl.RateLimitedError) as exc:
        lim.check_and_consume(pid=1, agent_id=None)
    assert exc.value.code == ErrorCode.RATE_LIMITED
