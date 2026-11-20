# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``slm doctor``: is the running daemon the one its installed service runs?

A daemon started outside its service keeps the starter's environment and is
invisible to the service manager (which then reports the service as idle and
never restarts it). This check compares the PID the service manager reports
with the PID in this data folder's daemon record, and, when they differ,
names the service settings the running daemon does not have. Values are never
printed, only variable names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupervisorCheck:
    status: str  # "PASS" | "WARN"
    detail: str
    fix: str = ""


_FIX = "Run `slm restart` so the installed service starts and owns the daemon"
_STALE_FIX = ("Run `slm serve install` so the service manager loads the installed "
              "definition, then `slm restart`")


def _environment_of(pid: int) -> dict[str, str] | None:
    try:
        import psutil

        return dict(psutil.Process(pid).environ())
    except Exception:  # noqa: BLE001 - another account's or an exited process
        return None


def _missing_settings(pid: int, expected: dict[str, str]) -> list[str]:
    actual = _environment_of(pid)
    if actual is None:
        return []
    # Only SLM settings: PATH and HOME legitimately differ between starters.
    return sorted(k for k, v in expected.items()
                  if k.startswith("SLM_") and actual.get(k) != v)


def check_supervisor_ownership() -> SupervisorCheck | None:
    """None when there is nothing to judge (no service, no live daemon, Windows)."""
    try:
        return _check()
    except Exception:  # noqa: BLE001 - diagnosis is advisory only
        return None


def _check() -> SupervisorCheck | None:
    from superlocalmemory.cli import daemon as client
    from superlocalmemory.cli import daemon_supervisor as supervisor

    descriptor = client.read_descriptor()
    if descriptor is None or not client._descriptor_process_is_alive(descriptor):
        return None
    jobs = [job for job in supervisor.jobs_for(descriptor.port) if job.platform != "windows"]
    if not jobs:
        return None
    # The daemon may be run by this folder's service for another port (#338).
    folder = [job for job in supervisor.jobs_for(None) if job.platform != "windows"]
    states = {job.name: supervisor.job_state(job) for job in [*jobs, *folder]}
    owner = next((name for name, state in states.items()
                  if state.status == "running" and state.pid == descriptor.pid), None)
    if owner is not None:
        return SupervisorCheck("PASS", f"pid {descriptor.pid} is run by service {owner}")
    if any(state.status == "unknown" for state in states.values()):
        # The service manager cannot be asked (no user bus, no GUI domain in a
        # remote shell). Nothing here is fixable by `slm restart`, and Linux
        # cannot even discover the unit then, so skip on every platform.
        return None
    job = jobs[0]
    state = states[job.name]
    if state.status == "stale":
        return SupervisorCheck(
            "WARN",
            f"the service job loaded as {job.name} does not match its installed "
            f"definition, so it is not trusted to own daemon pid {descriptor.pid}",
            _STALE_FIX,
        )
    detail = (
        f"daemon pid {descriptor.pid} is not owned by its service {job.name} "
        f"(service pid: {state.pid if state.pid else 'none'}); the service "
        "manager will not restart it and it may run without the service's settings"
    )
    missing = _missing_settings(descriptor.pid, dict(job.environment))
    if missing:
        detail += f"; differs from the service on: {', '.join(missing)}"
    return SupervisorCheck("WARN", detail, _FIX)
