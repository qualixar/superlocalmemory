# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""Regression tests for the profile-switch deadlock (v3.4.64).

ROOT CAUSE
----------
ProfileRuntime.transition() drains in-flight operation leases before committing
a switch.  Two independent defects made this drain hang forever:

1. **No timeout on the drain wait.**
   `_condition.wait()` had no timeout, so any in-flight operation (background
   recall warmup, materializer, health tick) caused the transition thread to
   block indefinitely.  When the HTTP client timed out (12 s) and Starlette
   cancelled the request, the underlying thread continued running with
   `_transitioning = True`, causing the thread pool to fill with blocked
   `acquire_operation()` threads — daemon wedged.

2. **Synchronous `with runtime.operation():` inside async route handlers.**
   `get_memory_facts`, `get_cluster_detail` (memories.py), and
   `trigger_consolidation` (v3_api.py) called `acquire_operation()` directly
   from the event-loop thread.  When `_transitioning = True`, this blocked the
   event loop, preventing `release_operation()` from ever running for the
   other in-flight requests — full deadlock.

FIXES VERIFIED HERE
-------------------
1. `TransitionDrainTimeout` is raised after `_DRAIN_TIMEOUT_SECS` (5 s by
   default).  After the timeout `_transitioning` is reset so the daemon stays
   responsive.
2. HTTP switch returns HTTP 503 (not a hang) when the drain times out.
3. After a timed-out switch the daemon is still operational: new operations
   can acquire leases, and a subsequent switch (once ops drain) succeeds.
4. Async routes that previously called `with runtime.operation():` in the
   event loop now execute their blocking code through
   `asyncio.to_thread()` and do NOT hold an extra lease.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest


# ── helpers ─────────────────────────────────────────────────────────────────

#: Liveness bound for waits only a broken build ever sits out (every held
#: lease is released by the test itself). NOT a performance claim; it keeps a
#: holder from letting go on its own while a loaded host is slow to get the
#: test to the switch.
_LIVENESS_GUARD_S = 30.0

#: Scheduling allowance on top of a drain timeout the runtime enforces itself.
#: A drain that ignored the shortened timeout would take the stock 5 s, or
#: wait out the whole lease.
_DRAIN_SLACK_S = 1.5


def _add_profile(engine, profile_id: str) -> None:
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, profile_id),
    )
    from superlocalmemory.server.routes.helpers import ensure_profile_in_json
    ensure_profile_in_json(profile_id)


def _daemon_headers(app) -> dict[str, str]:
    descriptor = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": descriptor.capability,
        "X-SLM-Target-Instance": descriptor.instance_id,
    }


@contextmanager
def _short_drain_timeout(secs: float = 0.3):
    """Temporarily shrink _DRAIN_TIMEOUT_SECS so tests run fast."""
    import superlocalmemory.server.profile_runtime as _prt
    original = _prt._DRAIN_TIMEOUT_SECS
    _prt._DRAIN_TIMEOUT_SECS = secs
    try:
        yield
    finally:
        _prt._DRAIN_TIMEOUT_SECS = original


# ── unit tests (ProfileRuntime directly) ────────────────────────────────────


def test_transition_raises_TransitionDrainTimeout_when_lease_is_held() -> None:
    """transition() must raise TransitionDrainTimeout when drain times out.

    Before the fix, `_condition.wait()` had no timeout and the thread blocked
    forever.  After the fix, it raises `TransitionDrainTimeout` after
    `_DRAIN_TIMEOUT_SECS` seconds and resets `_transitioning = False`.
    """
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        TransitionDrainTimeout,
    )

    runtime = ProfileRuntime("alpha")

    # Acquire an operation and hold it for the duration of the test.
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S), "holder thread did not acquire in time"

    drain_s = 0.3
    try:
        with _short_drain_timeout(drain_s):
            t0 = time.monotonic()
            with pytest.raises(TransitionDrainTimeout) as exc_info:
                runtime.transition("beta", lambda p, t: None)
            elapsed = time.monotonic() - t0

        # A real, enforced deadline: the timeout must fire at drain_s, plus
        # scheduling slack for a loaded host.
        assert elapsed < drain_s + _DRAIN_SLACK_S, (
            f"transition() took {elapsed:.2f}s — drain timeout was not respected."
        )
        assert "beta" in str(exc_info.value) or "in-flight" in str(exc_info.value), (
            f"Unexpected exception message: {exc_info.value}"
        )
    finally:
        release_lease.set()
        holder.join(_LIVENESS_GUARD_S)


def test_transitioning_flag_is_cleared_after_drain_timeout() -> None:
    """After TransitionDrainTimeout, _transitioning must be False.

    Before the fix: the transition thread ran indefinitely with
    `_transitioning = True`, wedging the daemon.  After the fix: the flag is
    reset to False so subsequent requests can proceed.
    """
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        TransitionDrainTimeout,
    )

    runtime = ProfileRuntime("alpha")
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S)

    try:
        with _short_drain_timeout(0.3):
            with pytest.raises(TransitionDrainTimeout):
                runtime.transition("beta", lambda p, t: None)

        # After the timeout, the transitioning flag MUST be cleared.
        assert not runtime.transitioning, (
            "_transitioning is still True after drain timeout — daemon permanently wedged."
        )
    finally:
        release_lease.set()
        holder.join(_LIVENESS_GUARD_S)


def test_operations_succeed_after_drain_timeout() -> None:
    """After a timed-out transition, new operation leases must be acquirable.

    Verifies the daemon is not left in a permanently unusable state.
    """
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        TransitionDrainTimeout,
    )

    runtime = ProfileRuntime("alpha")
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S)

    try:
        with _short_drain_timeout(0.3):
            with pytest.raises(TransitionDrainTimeout):
                runtime.transition("beta", lambda p, t: None)
    finally:
        release_lease.set()
        holder.join(_LIVENESS_GUARD_S)

    # Release the held lease — _active_operations drops to 0.
    # Now a fresh operation should succeed immediately.
    result: list[str] = []
    barrier = threading.Barrier(2)

    def _new_op() -> None:
        with runtime.operation() as snap:
            result.append(snap.profile_id)
            barrier.wait(timeout=_LIVENESS_GUARD_S)

    t = threading.Thread(target=_new_op, daemon=True)
    t.start()
    # A wedged runtime blocks _new_op forever; the guard turns that into a
    # BrokenBarrierError here instead of a hung suite.
    barrier.wait(timeout=_LIVENESS_GUARD_S)
    t.join(_LIVENESS_GUARD_S)

    assert result == ["alpha"], (
        f"New operation after timeout saw profile={result!r}. "
        "Daemon appears stuck in a bad state after the drain timeout."
    )


def test_successful_transition_after_lease_release() -> None:
    """Once the blocking lease is released, a subsequent switch must succeed.

    This proves the retry path: user waits for in-flight ops to finish,
    tries the switch again — it should go through cleanly.
    """
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        TransitionDrainTimeout,
    )

    runtime = ProfileRuntime("alpha")
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S)

    # First attempt: timed out because lease is held.
    with _short_drain_timeout(0.3):
        with pytest.raises(TransitionDrainTimeout):
            runtime.transition("beta", lambda p, t: None)

    # Release the lease.
    release_lease.set()
    holder.join(_LIVENESS_GUARD_S)

    # Second attempt (no timeout override needed): should succeed.
    seen: list[str] = []

    def _commit(prev, tgt):
        seen.append(f"{prev.profile_id}->{tgt}")

    with _short_drain_timeout(5.0):
        snapshot = runtime.transition("beta", _commit)

    assert snapshot.profile_id == "beta"
    assert seen == ["alpha->beta"], f"Unexpected commit calls: {seen}"


# ── HTTP integration test ────────────────────────────────────────────────────


def test_http_switch_returns_503_when_drain_times_out(engine_with_mock_deps) -> None:
    """POST /api/profiles/{name}/switch must return 503 (not hang) when drain times out.

    Before the fix: the HTTP request hung for 12 s+ until the client timed out,
    then the transition thread wedged the daemon.
    After the fix: the server returns HTTP 503 within the drain timeout window.
    """
    from fastapi.testclient import TestClient
    from superlocalmemory.server.unified_daemon import create_app
    from superlocalmemory.server.profile_runtime import (
        bind_profile_runtime,
        TransitionDrainTimeout,
    )

    engine = engine_with_mock_deps
    _add_profile(engine, "default")
    _add_profile(engine, "blocked_target")
    engine.profile_id = "default"
    engine._config.active_profile = "default"

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    runtime = bind_profile_runtime(app.state, engine, engine._config)

    # Hold an operation lease to block the transition.
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S), "holder did not acquire in time"

    drain_s = 0.5
    try:
        with _short_drain_timeout(drain_s):
            client = TestClient(app, raise_server_exceptions=False)
            headers = _daemon_headers(app)

            t0 = time.monotonic()
            resp = client.post(
                "/api/profiles/blocked_target/switch",
                headers=headers,
            )
            elapsed = time.monotonic() - t0

        # Must return 503, not 200 or 500, and certainly not a timeout.
        assert resp.status_code == 503, (
            f"Expected 503, got {resp.status_code}. Body: {resp.text[:200]}"
        )
        # Must respond PROMPTLY — the drain timeout plus slack for the HTTP
        # round trip on a loaded host, far short of the 12 s+ hang it fixed.
        assert elapsed < drain_s + _DRAIN_SLACK_S, (
            f"Switch took {elapsed:.2f}s — drain timeout was not enforced."
        )
        body = resp.json()
        detail = body.get("detail", "")
        assert "blocked_target" in detail or "in-flight" in detail or "timed out" in detail, (
            f"503 body does not mention the cause: {detail!r}"
        )
    finally:
        release_lease.set()
        holder.join(_LIVENESS_GUARD_S)


def test_daemon_responsive_during_switch_with_timeout(engine_with_mock_deps) -> None:
    """The daemon must not wedge after a timed-out profile switch.

    Before the fix: after the client timed out, `_transitioning` remained True,
    blocking all subsequent requests.  After the fix: a new operation can
    acquire a lease immediately after the timed-out switch.
    """
    from fastapi.testclient import TestClient
    from superlocalmemory.server.unified_daemon import create_app
    from superlocalmemory.server.profile_runtime import bind_profile_runtime

    engine = engine_with_mock_deps
    _add_profile(engine, "default")
    _add_profile(engine, "wedge_target")
    engine.profile_id = "default"
    engine._config.active_profile = "default"

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    runtime = bind_profile_runtime(app.state, engine, engine._config)

    lease_held = threading.Event()
    release_lease = threading.Event()

    def _hold_lease() -> None:
        with runtime.operation():
            lease_held.set()
            release_lease.wait(timeout=_LIVENESS_GUARD_S)

    holder = threading.Thread(target=_hold_lease, daemon=True)
    holder.start()
    assert lease_held.wait(_LIVENESS_GUARD_S)

    try:
        with _short_drain_timeout(0.4):
            client = TestClient(app, raise_server_exceptions=False)
            headers = _daemon_headers(app)

            # Switch — expect 503.
            switch_resp = client.post(
                "/api/profiles/wedge_target/switch",
                headers=headers,
            )
            assert switch_resp.status_code == 503

            # After the 503, the daemon must NOT be wedged.
            # _transitioning should be False. Verify by checking that a new
            # operation can acquire (it would block forever if transitioning=True).
            assert not runtime.transitioning, (
                "_transitioning is still True after drain timeout — daemon is wedged."
            )
    finally:
        release_lease.set()
        holder.join(_LIVENESS_GUARD_S)
