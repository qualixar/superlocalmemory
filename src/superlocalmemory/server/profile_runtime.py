"""Linearizable active-profile runtime state for the unified daemon."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time as _time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from superlocalmemory.infra.data_root import state_path

_RUNTIME_BIND_LOCK = threading.Lock()
_REQUEST_PROFILE: ContextVar[str | None] = ContextVar(
    "slm_request_profile", default=None,
)

# Maximum seconds to wait for in-flight operations to drain before a
# profile switch or engine reconfigure is aborted with HTTP 503.
# Tests can monkeypatch this module-level value to keep runs fast.
_DRAIN_TIMEOUT_SECS: float = 5.0


class TransitionDrainTimeout(Exception):
    """Raised when in-flight operations do not drain within _DRAIN_TIMEOUT_SECS.

    The caller (HTTP switch route) maps this to HTTP 503 so the client
    receives a clear error instead of a silent hang.  After the timeout,
    `_transitioning` is always reset to False so the daemon stays responsive.
    """


def current_request_profile() -> str | None:
    """Return the immutable profile snapshot admitted for this request."""
    return _REQUEST_PROFILE.get()


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """One immutable active-profile generation."""

    profile_id: str
    generation: int


class ProfileRuntime:
    """Coordinate profile-sensitive operations and exclusive transitions."""

    def __init__(self, profile_id: str, *, generation: int = 0) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._snapshot = ProfileSnapshot(profile_id, generation)
        self._active_operations = 0
        self._transitioning = False

    @property
    def snapshot(self) -> ProfileSnapshot:
        with self._condition:
            return self._snapshot

    @property
    def transitioning(self) -> bool:
        with self._condition:
            return self._transitioning

    def acquire_operation(self) -> ProfileSnapshot:
        """Admit an operation only when no profile transition is active."""
        with self._condition:
            while self._transitioning:
                self._condition.wait()
            self._active_operations += 1
            return self._snapshot

    def _try_acquire_operation_nowait(self) -> ProfileSnapshot | None:
        """Non-blocking acquire: returns None immediately when a transition is pending.

        Used by cooperative background maintenance tasks so they never block the
        drain window of a pending profile switch.
        """
        with self._condition:
            if self._transitioning:
                return None
            self._active_operations += 1
            return self._snapshot

    def release_operation(self) -> None:
        with self._condition:
            if self._active_operations <= 0:
                raise RuntimeError("profile runtime operation lease underflow")
            self._active_operations -= 1
            if self._active_operations == 0:
                self._condition.notify_all()

    @contextmanager
    def operation(self) -> Iterator[ProfileSnapshot]:
        snapshot = self.acquire_operation()
        try:
            yield snapshot
        finally:
            self.release_operation()

    @contextmanager
    def operation_nowait(self) -> Iterator[ProfileSnapshot | None]:
        """Cooperatively skip when a profile transition is pending.

        Background maintenance tasks that do NOT mutate profile-scoped engine
        state (cache warmup, health probes) MUST use this instead of
        ``operation()`` so they never hold the drain window hostage.

        Yields ``None`` (preempted — skip this work cycle) when a switch is
        already in progress.  Yields a :class:`ProfileSnapshot` and holds the
        lease normally when no transition is pending.

        The caller is responsible for checking ``if snap is None: return``.

        Unlike ``operation()``, this context manager never blocks — it either
        admits immediately or preempts immediately.
        """
        snapshot = self._try_acquire_operation_nowait()
        acquired = snapshot is not None
        try:
            yield snapshot
        finally:
            if acquired:
                self.release_operation()

    def transition(
        self,
        target_profile: str,
        commit: Callable[[ProfileSnapshot, str], None],
    ) -> ProfileSnapshot:
        """Drain admitted operations, commit, then publish a new generation.

        Raises TransitionDrainTimeout if in-flight operations do not drain
        within _DRAIN_TIMEOUT_SECS.  The flag is always reset on timeout so
        the daemon remains responsive (no permanent _transitioning=True wedge).
        """
        deadline = _time.monotonic() + _DRAIN_TIMEOUT_SECS
        with self._condition:
            while self._transitioning:
                self._condition.wait()
            if target_profile == self._snapshot.profile_id:
                return self._snapshot
            self._transitioning = True
            while self._active_operations > 0:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    # Reset before raising — never leave the daemon wedged.
                    self._transitioning = False
                    self._condition.notify_all()
                    raise TransitionDrainTimeout(
                        f"Profile switch to '{target_profile}' timed out after "
                        f"{_DRAIN_TIMEOUT_SECS:.0f}s: {self._active_operations} "
                        "in-flight operation(s) did not drain. "
                        "Try again when no active requests are in progress."
                    )
                # Use short sleep slices so we react quickly to notifications
                # and to timeout expiry without busy-spinning.
                self._condition.wait(timeout=min(remaining, 0.25))
            previous = self._snapshot

        try:
            commit(previous, target_profile)
        except BaseException:
            with self._condition:
                self._transitioning = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._snapshot = ProfileSnapshot(
                profile_id=target_profile,
                generation=previous.generation + 1,
            )
            self._transitioning = False
            self._condition.notify_all()
            return self._snapshot

    def reconfigure(self, commit: Callable[[ProfileSnapshot], None]) -> ProfileSnapshot:
        """Run a same-profile engine transition behind the operation barrier.

        Raises TransitionDrainTimeout if in-flight operations do not drain
        within _DRAIN_TIMEOUT_SECS (same semantics as transition()).
        """
        deadline = _time.monotonic() + _DRAIN_TIMEOUT_SECS
        with self._condition:
            while self._transitioning:
                self._condition.wait()
            self._transitioning = True
            while self._active_operations > 0:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    self._transitioning = False
                    self._condition.notify_all()
                    raise TransitionDrainTimeout(
                        f"Engine reconfigure timed out after {_DRAIN_TIMEOUT_SECS:.0f}s: "
                        f"{self._active_operations} in-flight operation(s) did not drain."
                    )
                self._condition.wait(timeout=min(remaining, 0.25))
            snapshot = self._snapshot

        try:
            commit(snapshot)
        except BaseException:
            with self._condition:
                self._transitioning = False
                self._condition.notify_all()
            raise

        with self._condition:
            self._transitioning = False
            self._condition.notify_all()
            return self._snapshot


@dataclass(slots=True)
class ActiveProfilePersistence:
    """Rollback handle for the two compatibility configuration stores."""

    previous: dict[Path, bytes | None]
    _rolled_back: bool = False

    def rollback(self) -> None:
        if self._rolled_back:
            return
        for path, content in self.previous.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write_bytes(path, content)
        self._rolled_back = True


def _load_json_object(path: Path, *, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def persist_active_profile(profile_id: str) -> ActiveProfilePersistence:
    """Atomically update config plus the legacy profiles cache, with rollback."""
    config_path = Path(state_path("config.json"))
    profiles_path = Path(state_path("profiles.json"))
    paths = (config_path, profiles_path)
    previous = {
        path: path.read_bytes() if path.exists() else None
        for path in paths
    }
    config = _load_json_object(config_path, default={})
    profiles = _load_json_object(
        profiles_path,
        default={
            "profiles": {
                "default": {
                    "name": "default",
                    "description": "Default memory profile",
                },
            },
        },
    )
    config["active_profile"] = profile_id
    profiles["active_profile"] = profile_id
    profile_catalog = profiles.get("profiles")
    if isinstance(profile_catalog, dict):
        selected = profile_catalog.get(profile_id)
        if isinstance(selected, dict):
            selected["last_used"] = datetime.now(timezone.utc).isoformat()
    rollback = ActiveProfilePersistence(previous)
    try:
        _atomic_write_bytes(config_path, _json_bytes(config))
        _atomic_write_bytes(profiles_path, _json_bytes(profiles))
    except BaseException:
        rollback.rollback()
        raise
    return rollback


def get_profile_runtime(app_state) -> ProfileRuntime:
    """Return the daemon runtime, lazily deriving its initial profile."""
    runtime = getattr(app_state, "profile_runtime", None)
    if runtime is not None:
        return runtime
    with _RUNTIME_BIND_LOCK:
        runtime = getattr(app_state, "profile_runtime", None)
        if runtime is not None:
            return runtime
        engine = getattr(app_state, "engine", None)
        config = getattr(app_state, "config", None)
        profile_id = (
            getattr(engine, "profile_id", "")
            or getattr(config, "active_profile", "")
            or "default"
        )
        runtime = ProfileRuntime(str(profile_id))
        app_state.profile_runtime = runtime
        return runtime


def bind_profile_runtime(app_state, engine, config) -> ProfileRuntime:
    """Attach the authoritative runtime after daemon engine initialization."""
    runtime = ProfileRuntime(str(engine.profile_id))
    app_state.profile_runtime = runtime
    app_state.engine = engine
    app_state.config = config
    return runtime


def commit_daemon_profile_switch(
    app_state,
    previous: ProfileSnapshot,
    target_profile: str,
) -> None:
    """Persist and rebind a quiescent resident engine, rolling back on error."""
    engine = getattr(app_state, "engine", None)
    if engine is None:
        raise RuntimeError("resident engine is unavailable")
    rows = engine._db.execute(
        "SELECT 1 FROM profiles WHERE profile_id = ?",
        (target_profile,),
    )
    if not rows:
        raise RuntimeError(
            f"profile '{target_profile}' no longer exists at commit time"
        )
    app_config = getattr(app_state, "config", None)
    engine_config = getattr(engine, "_config", None)
    persistence = None
    try:
        # Requests are drained and runtime generation is not yet published.
        # Rebind the in-memory engine first, then make compatibility files the
        # final commit step so they can never lead daemon runtime truth.
        engine.profile_id = target_profile
        if app_config is not None:
            app_config.active_profile = target_profile
        if engine_config is not None:
            engine_config.active_profile = target_profile
        persistence = persist_active_profile(target_profile)
    except BaseException:
        engine.profile_id = previous.profile_id
        if app_config is not None:
            app_config.active_profile = previous.profile_id
        if engine_config is not None:
            engine_config.active_profile = previous.profile_id
        if persistence is not None:
            persistence.rollback()
        raise


def reconfigure_daemon_engine(app_state, new_config, *, mode_change: bool) -> None:
    """Exclusively rebuild every daemon engine reference for a new config."""
    callback = getattr(app_state, "reconfigure_engine", None)
    if not callable(callback):
        # Direct route/unit usage without a resident daemon retains the
        # established persistence-only behavior.
        new_config.save(mode_change=mode_change)
        return
    runtime = get_profile_runtime(app_state)

    def _commit(snapshot: ProfileSnapshot) -> None:
        # The candidate may have been loaded before a concurrent profile
        # switch. Runtime truth always wins over that stale config snapshot.
        new_config.active_profile = snapshot.profile_id
        callback(new_config, mode_change=mode_change)

    runtime.reconfigure(_commit)


class ProfileRuntimeMiddleware:
    """Hold an operation lease for each daemon HTTP request."""

    def __init__(self, app, *, app_state) -> None:
        self._app = app
        self._app_state = app_state

    @staticmethod
    def _is_transition_request(path: str, method: str) -> bool:
        profile_switch = (
            path.startswith("/api/profiles/")
            and path.endswith("/switch")
        )
        config_transition = method in {"POST", "PUT", "PATCH"} and path in {
            "/api/v3/mode",
            "/api/v3/mode/set",
            "/api/v3/embedding/config",
            "/api/v3/scope/config",
        }
        return profile_switch or config_transition

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()
        # Mounted MCP tools proxy profile-sensitive work back through the
        # canonical daemon routes. Leasing the outer MCP request would make an
        # embedded switch wait on itself.
        if path.startswith("/mcp") or self._is_transition_request(path, method):
            await self._app(scope, receive, send)
            return

        import asyncio

        runtime = get_profile_runtime(self._app_state)
        acquire_task = asyncio.create_task(
            asyncio.to_thread(runtime.acquire_operation)
        )
        try:
            # Shield the worker so cancellation cannot strand a lease that the
            # thread acquires after this coroutine has already unwound.
            snapshot = await asyncio.shield(acquire_task)
        except asyncio.CancelledError:
            await acquire_task
            runtime.release_operation()
            raise
        scope.setdefault("state", {})["profile_snapshot"] = snapshot
        token = _REQUEST_PROFILE.set(snapshot.profile_id)

        # The operation lease guards the SYNCHRONOUS route work that produces
        # the response — not the streaming of its body. A long-lived SSE body
        # would otherwise hold the lease for its whole lifetime and make every
        # profile switch time out during drain (HTTP 503):
        #   * /events/stream runs `while True: await sleep(1)` FOREVER while a
        #     dashboard tab is open;
        #   * /api/v3/chat/stream streams LLM tokens for up to 120s.
        # Release the lease the instant the response starts. By then the route
        # handler has finished its engine work and returned its Response
        # (FastAPI buffers normal responses before sending response.start).
        # Streaming generators that still need the engine (chat recall) already
        # re-acquire their own short-lived lease internally, and /events/stream
        # only reads the EventBus DB — neither depends on this outer lease.
        released = False

        def _release_lease_once() -> None:
            # Single-task coroutine chain — a plain flag is sufficient; the
            # underlying release_operation() is lock-guarded and idempotent-safe
            # only via this guard, so never call it twice.
            nonlocal released
            if not released:
                released = True
                runtime.release_operation()

        async def _send(message) -> None:
            if message.get("type") == "http.response.start":
                _release_lease_once()
            await send(message)

        try:
            await self._app(scope, receive, _send)
        finally:
            _REQUEST_PROFILE.reset(token)
            # Safety net: a request that errors before emitting response.start
            # (or an ASGI app that never sends one) must still release its lease.
            # Release is lock-only and must not itself be cancellation-prone.
            _release_lease_once()


__all__ = [
    "ActiveProfilePersistence",
    "ProfileRuntime",
    "ProfileRuntimeMiddleware",
    "ProfileSnapshot",
    "TransitionDrainTimeout",
    "bind_profile_runtime",
    "commit_daemon_profile_switch",
    "current_request_profile",
    "get_profile_runtime",
    "persist_active_profile",
    "reconfigure_daemon_engine",
]
