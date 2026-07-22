# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for runtime-configurable rate limits (task #47)."""

from __future__ import annotations

import pytest

from superlocalmemory.infra import rate_limiter as rl
from superlocalmemory.infra.rate_limiter import RateLimiter


@pytest.fixture(autouse=True)
def _clean_registry():
    rl.reset_managed()
    # Restore known defaults so cross-test order can't leak state.
    rl.set_limits(write=100, read=300, window=60)
    rl.reset_managed()
    yield
    rl.reset_managed()


class TestConfigure:
    def test_configure_changes_max_and_window(self) -> None:
        lim = RateLimiter(max_requests=5, window_seconds=60)
        lim.configure(max_requests=10)
        assert lim.max_requests == 10
        assert lim.window == 60
        lim.configure(window_seconds=30)
        assert lim.window == 30

    def test_configure_floors_at_one(self) -> None:
        lim = RateLimiter(max_requests=5)
        lim.configure(max_requests=0)
        assert lim.max_requests == 1

    def test_raised_limit_unblocks_immediately(self) -> None:
        lim = RateLimiter(max_requests=2, window_seconds=60)
        assert lim.allow("c")
        assert lim.allow("c")
        assert not lim.allow("c")           # blocked at 2
        lim.configure(max_requests=5)
        assert lim.allow("c")               # raised → unblocked, same window


class TestRegistry:
    def test_set_limits_applies_to_all_roles(self) -> None:
        w, r = RateLimiter(max_requests=1), RateLimiter(max_requests=1)
        lbw, lbr = RateLimiter(max_requests=1), RateLimiter(max_requests=1)
        rl.register_managed("write", w)
        rl.register_managed("read", r)
        rl.register_managed("lb_write", lbw)
        rl.register_managed("lb_read", lbr)

        cur = rl.set_limits(write=50, read=200, window=90)
        assert w.max_requests == 50
        assert r.max_requests == 200
        assert lbw.max_requests == max(300, 50 * 10)      # derived loopback
        assert lbr.max_requests == max(2000, 200 * 20)
        assert w.window == r.window == 90
        assert cur == {"write": 50, "read": 200, "window": 90}
        assert rl.get_limits() == {"write": 50, "read": 200, "window": 90}

    def test_partial_update_keeps_others(self) -> None:
        rl.set_limits(write=40, read=150, window=60)
        cur = rl.set_limits(write=99)
        assert cur["write"] == 99
        assert cur["read"] == 150            # unchanged
        assert cur["window"] == 60

    def test_reset_managed_stops_reconfiguring(self) -> None:
        w = RateLimiter(max_requests=10)
        rl.register_managed("write", w)
        rl.reset_managed()
        rl.set_limits(write=999)
        assert w.max_requests == 10          # no longer managed
