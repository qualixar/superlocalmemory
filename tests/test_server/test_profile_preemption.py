# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""Cooperative preemption for background maintenance ops (v3.x).

ROOT CAUSE
----------
Profile switches always returned 503 "5s timeout: 1 in-flight operation
did not drain" because background maintenance tasks held the operation
lease for longer than the 5s drain window:

  * _warmup_recall (unified_daemon.py):  ONE lease across TWO full-fusion
    recall queries (each 2–10s) → total hold up to 20s.
  * run_health_tick (recall_health.py):  ONE lease across a full-fusion
    recall with fast=False (2–10s).
  * _run_materializer_operation:         ONE lease per materialization
    (embedding call ≈ 1–5s), looped without a transition check.

FIX — COOPERATIVE PREEMPTION
-----------------------------
1. ProfileRuntime.operation_nowait() — a non-blocking context manager that
   yields ``None`` (preempted, skip this cycle) instead of acquiring when
   _transitioning is True.  Background tasks that do NOT mutate profile
   state should use this rather than operation().

2. _warmup_recall — splits the single lease into per-query leases via
   operation_nowait(); a pending transition preempts remaining queries.

3. run_health_tick — uses operation_nowait(); skips the tick entirely if
   a switch is pending (the next scheduled tick re-warms).

4. _run_materializer_operation — checks runtime.transitioning BEFORE
   acquiring the operation lease and returns None to skip this cycle;
   the materializer loop retries after the switch commits.

TESTS HERE
----------
1. unit: operation_nowait() admits immediately when no transition is
   pending and yields None when _transitioning is True.
2. integration: a background op using operation_nowait() + transitioning
   check allows a profile switch to succeed within ~2s even when the op
   would otherwise hold the lease for 10s.
3. data isolation: after the switch, the target profile is empty (its
   fact count is 0 while the source profile had ≥1 stored fact).
4. regression: the 5s safety-timeout path (TransitionDrainTimeout) still
   fires for ops that do NOT cooperate (protects the daemon from wedging).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import pytest


# ── helpers ──────────────────────────────────────────────────────────────────


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
    """Temporarily shrink _DRAIN_TIMEOUT_SECS so timeout-path tests run fast."""
    import superlocalmemory.server.profile_runtime as _prt
    original = _prt._DRAIN_TIMEOUT_SECS
    _prt._DRAIN_TIMEOUT_SECS = secs
    try:
        yield
    finally:
        _prt._DRAIN_TIMEOUT_SECS = original


# ── unit: operation_nowait() ─────────────────────────────────────────────────


def test_operation_nowait_admits_when_no_transition_is_pending() -> None:
    """operation_nowait() behaves like operation() when no switch is in progress."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    seen: list[str] = []

    with runtime.operation_nowait() as snap:
        assert snap is not None, "operation_nowait() must admit when not transitioning"
        assert snap.profile_id == "alpha"
        seen.append(snap.profile_id)

    # After the context exits, active_operations must be 0.
    assert seen == ["alpha"]
    # Confirm the lease was released (a transition would block otherwise).
    committed: list[str] = []
    runtime.transition("beta", lambda p, t: committed.append(t))
    assert committed == ["beta"]


def test_operation_nowait_yields_none_when_transition_is_pending() -> None:
    """operation_nowait() preempts immediately when _transitioning is True."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    held = threading.Event()
    release = threading.Event()

    # Hold a regular operation to trigger the transition drain wait.
    def _hold():
        with runtime.operation():
            held.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert held.wait(2.0)

    # Start a transition — this sets _transitioning=True and waits for drain.
    switch_started = threading.Event()

    def _switch():
        switch_started.set()
        with _short_drain_timeout(5.0):
            runtime.transition("beta", lambda p, t: None)

    switch_thread = threading.Thread(target=_switch, daemon=True)
    switch_thread.start()
    # Give the transition thread enough time to set _transitioning=True.
    switch_started.wait(1.0)
    deadline = time.monotonic() + 1.0
    while not runtime.transitioning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime.transitioning, "_transitioning must be True at this point"

    # operation_nowait() must yield None immediately (not block waiting).
    t0 = time.monotonic()
    with runtime.operation_nowait() as snap:
        elapsed = time.monotonic() - t0
        assert snap is None, (
            f"Expected operation_nowait() to yield None when transitioning, "
            f"got {snap!r}"
        )
        assert elapsed < 0.1, (
            f"operation_nowait() took {elapsed:.3f}s — it must return immediately"
        )

    # Clean up.
    release.set()
    holder.join(2.0)
    switch_thread.join(2.0)


def test_operation_nowait_does_not_leave_orphaned_lease_on_preemption() -> None:
    """After a preempted operation_nowait(), active_operations stays at zero."""
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with runtime.operation():
            held.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert held.wait(2.0)

    transition_started = threading.Event()

    def _switch():
        transition_started.set()
        with _short_drain_timeout(5.0):
            runtime.transition("beta", lambda p, t: None)

    switch_thread = threading.Thread(target=_switch, daemon=True)
    switch_thread.start()
    transition_started.wait(1.0)
    deadline = time.monotonic() + 1.0
    while not runtime.transitioning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime.transitioning

    # operation_nowait() preempts — must not increment active_operations.
    with runtime.operation_nowait() as snap:
        assert snap is None

    # Release the real lease — transition should now complete cleanly.
    release.set()
    holder.join(2.0)
    switch_thread.join(2.0)

    # If there was an orphaned lease, this transition would timeout.
    final_snap = runtime.snapshot
    assert final_snap.profile_id == "beta"


# ── integration: cooperative preemption allows switch within 2s ──────────────


def test_cooperative_background_op_yields_and_switch_succeeds(
    engine_with_mock_deps,
) -> None:
    """A background maintenance op using cooperative preemption allows a profile
    switch to succeed within ~2s even when the op would otherwise hold the lease
    for 10s (simulating a slow full-fusion recall or embedding call).

    RED: operation_nowait() does not exist → AttributeError.
    GREEN: operation_nowait() is added → background op detects _transitioning
           and releases the lease → switch commits in <2.5s.
    """
    from fastapi.testclient import TestClient
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    _add_profile(engine, "default")
    _add_profile(engine, "work")

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    runtime = bind_profile_runtime(app.state, engine, engine._config)
    client = TestClient(app)
    headers = _daemon_headers(app)

    # Write a memory to the SOURCE profile so data isolation is verifiable.
    stored = client.post(
        "/remember?wait=true",
        json={"content": "preemption-test-source-marker-6491"},
        headers=headers,
    )
    assert stored.status_code == 200, stored.text
    source_count = client.get("/status").json()["fact_count"]
    assert source_count >= 1, "Source profile must have stored facts"

    # --- Simulate a background maintenance op (health tick / warmup style) ---
    # It acquires an operation_nowait() lease and does "expensive" work in a
    # tight loop.  With cooperative preemption it detects _transitioning and
    # exits the lease early.  Without operation_nowait() the lease would be
    # held for up to 10s — exceeding the 5s drain timeout and producing 503.
    op_lease_acquired = threading.Event()
    op_yielded_early = threading.Event()

    def _simulated_maintenance_op() -> None:
        with runtime.operation_nowait() as snap:
            if snap is None:
                # Already transitioning before we could even start — this is
                # the "don't acquire new leases" path; not the focus of this
                # test but still correct behaviour.
                return
            op_lease_acquired.set()
            # Inner work loop — each iteration simulates ~50ms of CPU/IO work.
            # Without preemption, this would hold the lease for 200 * 0.05 = 10s.
            for _ in range(200):
                if runtime.transitioning:
                    # Cooperative preemption: transition is pending.
                    # Exit the context manager to release the lease promptly.
                    op_yielded_early.set()
                    return
                time.sleep(0.05)
        # Natural completion — op ran to end without a pending transition.
        op_yielded_early.set()

    bg_thread = threading.Thread(
        target=_simulated_maintenance_op, daemon=True, name="test-bg-op",
    )
    bg_thread.start()
    assert op_lease_acquired.wait(2.0), "Background op did not acquire its lease"

    # Issue the profile switch.  Because the background op will yield via the
    # transitioning check, the drain completes and this returns 200.
    t0 = time.monotonic()
    switch_resp = client.post("/api/profiles/work/switch", headers=headers)
    elapsed = time.monotonic() - t0

    assert switch_resp.status_code == 200, (
        f"Expected HTTP 200 from profile switch (cooperative preemption), "
        f"got {switch_resp.status_code}.  Body: {switch_resp.text[:300]}"
    )
    assert elapsed < 2.5, (
        f"Profile switch took {elapsed:.2f}s — cooperative preemption should "
        f"allow the drain to complete well within the 5s window."
    )

    bg_thread.join(2.0)
    assert op_yielded_early.is_set(), (
        "Background op did not signal early yield — preemption mechanism missing."
    )

    # --- Data isolation: target profile must be empty -------------------------
    payload = switch_resp.json()
    assert payload["active_profile"] == "work", (
        f"Switch payload reports wrong profile: {payload}"
    )
    assert payload["generation"] >= 1

    status_after = client.get("/status").json()
    assert status_after["profile"] == "work", (
        f"Daemon still reports 'default' after switch: {status_after}"
    )
    assert status_after["fact_count"] == 0, (
        f"Target profile 'work' should have 0 facts after switch (isolation), "
        f"got {status_after['fact_count']}."
    )


# ── regression: 5s timeout safety still fires for non-cooperative ops ────────


def test_drain_timeout_still_fires_for_uncooperative_ops() -> None:
    """The 5s safety timeout (TransitionDrainTimeout → 503) still fires when a
    background op holds a regular operation() lease and never yields.

    This regression guard ensures the preemption fix does not accidentally remove
    the timeout safety net that prevents permanent daemon wedging.
    """
    from superlocalmemory.server.profile_runtime import (
        ProfileRuntime,
        TransitionDrainTimeout,
    )

    runtime = ProfileRuntime("alpha")
    lease_held = threading.Event()
    release_lease = threading.Event()

    def _stubborn_op() -> None:
        with runtime.operation():   # Regular operation — no preemption
            lease_held.set()
            release_lease.wait(timeout=10.0)

    holder = threading.Thread(target=_stubborn_op, daemon=True)
    holder.start()
    assert lease_held.wait(2.0)

    try:
        with _short_drain_timeout(0.3):
            t0 = time.monotonic()
            with pytest.raises(TransitionDrainTimeout):
                runtime.transition("beta", lambda p, t: None)
            elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"Drain timeout took {elapsed:.2f}s — should fire within 2× the "
            f"_DRAIN_TIMEOUT_SECS window."
        )
        assert not runtime.transitioning, (
            "_transitioning must be False after drain timeout (daemon must not wedge)"
        )
    finally:
        release_lease.set()
        holder.join(2.0)


# ── regression: production background ops honour preemption ──────────────────


def test_run_health_tick_skips_when_transition_is_pending() -> None:
    """recall_health.run_health_tick() must cooperatively skip its recall when a
    profile switch is pending, so it does not hold the operation lease during
    the drain window.
    """
    from unittest.mock import MagicMock

    from superlocalmemory.server.profile_runtime import ProfileRuntime
    from superlocalmemory.server.recall_health import RecallHealth, run_health_tick

    runtime = ProfileRuntime("alpha")
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with runtime.operation():
            held.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert held.wait(2.0)

    transition_started = threading.Event()

    def _switch():
        transition_started.set()
        with _short_drain_timeout(5.0):
            runtime.transition("beta", lambda p, t: None)

    switch_thread = threading.Thread(target=_switch, daemon=True)
    switch_thread.start()
    transition_started.wait(1.0)
    deadline = time.monotonic() + 1.0
    while not runtime.transitioning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime.transitioning, "Transition must be in progress"

    # run_health_tick() should skip (not call engine.recall) when transitioning.
    engine = MagicMock()
    state = RecallHealth()
    result = run_health_tick(engine, state, runtime=runtime)

    # The tick must have skipped — engine.recall must NOT have been called.
    engine.recall.assert_not_called()
    assert result.checks == 1  # checks counter incremented
    assert result.healthy is True  # health unchanged (not degraded)

    # Clean up.
    release.set()
    holder.join(2.0)
    switch_thread.join(2.0)


def test_materializer_operation_skips_when_transition_is_pending() -> None:
    """_run_materializer_operation() must return None immediately (not acquire
    the operation lease) when a profile transition is already in progress.
    """
    from unittest.mock import MagicMock

    import superlocalmemory.server.unified_daemon as _ud
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    runtime = ProfileRuntime("alpha")
    held = threading.Event()
    release = threading.Event()

    def _hold():
        with runtime.operation():
            held.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=_hold, daemon=True)
    holder.start()
    assert held.wait(2.0)

    transition_started = threading.Event()

    def _switch():
        transition_started.set()
        with _short_drain_timeout(5.0):
            runtime.transition("beta", lambda p, t: None)

    switch_thread = threading.Thread(target=_switch, daemon=True)
    switch_thread.start()
    transition_started.wait(1.0)
    deadline = time.monotonic() + 1.0
    while not runtime.transitioning and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime.transitioning, "Transition must be in progress"

    # _run_materializer_operation must skip (return None) when transitioning.
    operation_called = []

    def _slow_op(engine):
        operation_called.append(True)
        time.sleep(10)  # This would hold the lease for 10s without the fix
        return (1, 0)

    result = _ud._run_materializer_operation(runtime, lambda: MagicMock(), _slow_op)

    assert result is None, (
        f"Expected None (skipped), got {result!r}. "
        "Materializer must not acquire the operation lease when transitioning."
    )
    assert not operation_called, (
        "Materializer must not call the operation function when transitioning."
    )

    # Clean up.
    release.set()
    holder.join(2.0)
    switch_thread.join(2.0)
