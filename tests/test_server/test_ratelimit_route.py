# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for the dashboard-editable rate-limit route (task #47)."""

from __future__ import annotations

import asyncio
import json

import pytest

import superlocalmemory.server.routes.ratelimit as rlr
from superlocalmemory.infra import rate_limiter as rl
from superlocalmemory.infra.rate_limiter import RateLimiter


class _Req:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


@pytest.fixture()
def managed(monkeypatch):
    """Fresh registry with observable limiters + in-memory config.json."""
    rl.reset_managed()
    rl.set_limits(write=100, read=300, window=60)
    w, r = RateLimiter(max_requests=100), RateLimiter(max_requests=300)
    lbw, lbr = RateLimiter(max_requests=1000), RateLimiter(max_requests=6000)
    rl.reset_managed()
    rl.register_managed("write", w)
    rl.register_managed("read", r)
    rl.register_managed("lb_write", lbw)
    rl.register_managed("lb_read", lbr)

    store: dict = {}
    monkeypatch.setattr(rlr, "_read_config", lambda: dict(store))
    monkeypatch.setattr(rlr, "_atomic_write", lambda d: store.update(d))
    monkeypatch.setattr(rlr, "_require_admin", lambda request: None)
    yield {"w": w, "r": r, "lbw": lbw, "lbr": lbr, "store": store}
    rl.reset_managed()


def _body(resp) -> dict:
    return json.loads(resp.body)


def test_get_returns_effective_limits(managed) -> None:
    body = _body(rlr.get_ratelimit())
    assert body["write"] == 100
    assert body["read"] == 300
    assert body["window"] == 60
    assert body["loopback_write"] == max(300, 100 * 10)


def test_put_applies_at_runtime_and_persists(managed) -> None:
    resp = asyncio.run(rlr.put_ratelimit(_Req({"write": 500, "read": 1500})))
    body = _body(resp)
    assert body["success"] is True
    assert body["write"] == 500
    # Runtime apply reached the live limiters.
    assert managed["w"].max_requests == 500
    assert managed["r"].max_requests == 1500
    assert managed["lbw"].max_requests == max(300, 500 * 10)
    # Persisted to config.json.
    assert managed["store"]["rate_limit"]["write"] == 500
    assert managed["store"]["rate_limit"]["read"] == 1500


def test_put_partial_keeps_others(managed) -> None:
    asyncio.run(rlr.put_ratelimit(_Req({"window": 120})))
    assert rl.get_limits() == {"write": 100, "read": 300, "window": 120}


def test_put_empty_is_422(managed) -> None:
    resp = asyncio.run(rlr.put_ratelimit(_Req({})))
    assert resp.status_code == 422
    assert managed["w"].max_requests == 100  # unchanged


def test_put_out_of_range_is_422(managed) -> None:
    resp = asyncio.run(rlr.put_ratelimit(_Req({"write": 999999999})))
    assert resp.status_code == 422


def test_load_persisted_limits_reapplies(managed) -> None:
    managed["store"]["rate_limit"] = {"write": 250, "read": 900, "window": 90}
    rlr.load_persisted_limits()
    assert rl.get_limits() == {"write": 250, "read": 900, "window": 90}
    assert managed["w"].max_requests == 250
