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
    pids = {job.name: supervisor.job_pid(job) for job in jobs}
    owner = next((name for name, pid in pids.items() if pid == descriptor.pid), None)
    if owner is not None:
        return SupervisorCheck("PASS", f"pid {descriptor.pid} is run by service {owner}")
    job = jobs[0]
    service_pid = pids[job.name]
    detail = (
        f"daemon pid {descriptor.pid} is not owned by its service {job.name} "
        f"(service pid: {service_pid if service_pid else 'none'}); the service "
        "manager will not restart it and it may run without the service's settings"
    )
    missing = _missing_settings(descriptor.pid, dict(job.environment))
    if missing:
        detail += f"; differs from the service on: {', '.join(missing)}"
    return SupervisorCheck("WARN", detail, _FIX)
