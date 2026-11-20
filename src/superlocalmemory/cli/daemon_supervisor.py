# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Let the installed operating-system service own the daemon.

When ``slm serve install`` has registered a service for this data folder and
port (a LaunchAgent on macOS, a systemd user unit on Linux, a logon task on
Windows), that service is the daemon's owner and its definition is the
daemon's configuration. A client that needs the daemon asks the service
manager to start it instead of spawning a detached child of its own: a child
would carry the client's environment (agent name, MCP profile, channel) into a
shared service, would miss settings that live only in the service definition,
and would hold the port so the service's own start later finds it taken.

Only definitions that ``service_inventory`` proves belong to this data folder
and port are considered; a service of another installation is never touched.
Every query is bounded and fails closed to "no service", so the caller falls
back to a direct start whose environment is scrubbed (``daemon_environment``).
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

#: Variables that describe the client process, never the shared daemon. A
#: daemon that inherits them attributes every caller's work to one agent,
#: reports one client's tool profile as its own, or reuses the client's
#: identity material. They are removed from every directly started daemon.
CLIENT_SCOPED_ENV = frozenset({
    "SLM_AGENT_ID",
    "SLM_CLIENT",
    "SLM_CLIENT_CHANNEL",
    "SLM_SESSION_ID",
    "SLM_PROJECT_PATH",
    "SLM_PROFILE_ID",
    "SLM_MCP_PROFILE",
    "SLM_MCP_TOOLS",
    "SLM_MCP_ALL_TOOLS",
    "SLM_MCP_MESH_TOOLS",
    "SLM_MCP_HOST",
    "SLM_MCP_EMBEDDED",
    "SLM_SKIP_DEP_CHECK",
    "SLM_CODEX_HOOK",
    "SLM_DAEMON_INSTANCE_ID",
    "SLM_DAEMON_CAPABILITY",
})
_CLIENT_SCOPED_PREFIXES = ("SLM_HOOK_",)

_TIMEOUT_S = 10


@dataclass(frozen=True)
class SupervisorJob:
    """One installed service definition proven to own this data folder and port."""

    platform: str
    name: str
    path: Path
    port: int
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StartOutcome:
    started: bool
    how: str  # "kickstarted" | "loaded" | "started" | "failed"
    detail: str = ""


def current_platform() -> str | None:
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return None


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT_S)


# --- definition environment ----------------------------------------------------

def _macos_environment(content: str) -> dict[str, str]:
    import plistlib

    raw = plistlib.loads(content.encode()).get("EnvironmentVariables", {})
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _linux_environment(content: str) -> dict[str, str]:
    environment: dict[str, str] = {}
    section = ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
        if section != "[Service]" or not line.startswith("Environment="):
            continue
        value = line.split("=", 1)[1]
        if not value:
            environment = {}
        for pair in shlex.split(value):
            if "=" in pair:
                key, item = pair.split("=", 1)
                environment[key] = item
    return environment


_WINDOWS_ASSIGNMENT = re.compile(
    r'WshShell.Environment\("Process"\)\("([A-Z_]+)"\) = "((?:[^"]|"")*)"'
)


def _windows_environment(content: str) -> dict[str, str]:
    return {
        match[1]: match[2].replace('""', '"')
        for match in (_WINDOWS_ASSIGNMENT.fullmatch(line) for line in content.splitlines())
        if match
    }


_ENVIRONMENT_PARSERS = {
    "macos": _macos_environment,
    "linux": _linux_environment,
    "windows": _windows_environment,
}


# --- discovery -------------------------------------------------------------------

def jobs_for(port: int) -> list[SupervisorJob]:
    """Every installed definition that owns this data folder on ``port``."""
    platform = current_platform()
    if platform is None:
        return []
    try:
        from superlocalmemory.cli import service_inventory

        rows = service_inventory.definitions(platform)
    except Exception as exc:  # noqa: BLE001 - discovery must never block a start
        logger.debug("service discovery failed: %s", exc)
        return []
    jobs = []
    for row in rows:
        if row.port != port:
            continue
        try:
            environment = _ENVIRONMENT_PARSERS[platform](row.content)
        except Exception:  # noqa: BLE001 - a definition we cannot read is not ours to start
            continue
        jobs.append(SupervisorJob(platform, row.name, row.path, row.port, environment))
    return jobs


def find_job(port: int) -> SupervisorJob | None:
    jobs = jobs_for(port)
    return jobs[0] if jobs else None


def _macos_domain() -> str:
    return f"gui/{os.getuid()}"


def _macos_record(job: SupervisorJob) -> dict | None:
    from superlocalmemory.cli.service_live import launchctl_list

    return launchctl_list(job.name)


def _linux_properties(job: SupervisorJob) -> dict[str, str]:
    result = _run(["systemctl", "--user", "show", job.name,
                   "--property=LoadState", "--property=MainPID"])
    if result.returncode:
        raise OSError("Cannot inspect systemd user unit")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def job_loaded(job: SupervisorJob) -> bool:
    """Is the service manager currently holding this definition?"""
    try:
        if job.platform == "macos":
            return _macos_record(job) is not None
        if job.platform == "linux":
            return _linux_properties(job).get("LoadState") == "loaded"
        if job.platform == "windows":
            return _run(["schtasks", "/Query", "/TN", job.name]).returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return False


def job_pid(job: SupervisorJob) -> int | None:
    """PID of the process the service manager runs for this job, if any.

    Windows logon tasks hand the daemon to a script host and never restart
    it, so the task has no daemon PID to report: always None there.
    """
    try:
        if job.platform == "macos":
            record = _macos_record(job)
            pid = record.get("PID") if record else None
        elif job.platform == "linux":
            pid = _linux_properties(job).get("MainPID")
        else:
            return None
        pid = int(pid) if pid is not None else 0
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    return pid if pid > 0 else None


# --- start -----------------------------------------------------------------------

def _start_macos(job: SupervisorJob) -> StartOutcome:
    domain = _macos_domain()
    target = f"{domain}/{job.name}"
    if not job_loaded(job):
        # The definition is installed but not loaded (for example after
        # ``launchctl bootout``). launchd loads it at the next login anyway;
        # loading it now keeps the daemon under its owner instead of
        # starting an unowned copy that the service would later collide with.
        loaded = _run(["launchctl", "bootstrap", domain, str(job.path)])
        if loaded.returncode == 0:
            return StartOutcome(True, "loaded")
        if not job_loaded(job):  # another client may have loaded it first
            return StartOutcome(False, "failed", (loaded.stderr or "").strip()
                                or f"launchctl bootstrap exited {loaded.returncode}")
    kicked = _run(["launchctl", "kickstart", target])
    if kicked.returncode == 0:
        return StartOutcome(True, "kickstarted")
    return StartOutcome(False, "failed", (kicked.stderr or "").strip()
                        or f"launchctl kickstart exited {kicked.returncode}")


def _start_linux(job: SupervisorJob) -> StartOutcome:
    result = _run(["systemctl", "--user", "start", job.name])
    if result.returncode == 0:
        return StartOutcome(True, "started")
    return StartOutcome(False, "failed", (result.stderr or "").strip()
                        or f"systemctl start exited {result.returncode}")


def _start_windows(job: SupervisorJob) -> StartOutcome:
    result = _run(["schtasks", "/Run", "/TN", job.name])
    if result.returncode == 0:
        return StartOutcome(True, "started")
    return StartOutcome(False, "failed", (result.stderr or "").strip()
                        or f"schtasks /Run exited {result.returncode}")


_STARTERS = {"macos": _start_macos, "linux": _start_linux, "windows": _start_windows}


def request_start(job: SupervisorJob) -> StartOutcome:
    """Ask the service manager to start ``job``; never raises."""
    try:
        return _STARTERS[job.platform](job)
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        return StartOutcome(False, "failed", str(exc))


def start_through_service(port: int, wait_for_daemon) -> tuple[bool | None, SupervisorJob | None]:
    """Start the daemon through its installed service when there is one.

    Returns ``(ready, job)``. ``ready`` is None when the caller must start the
    daemon directly: no service owns this folder and port, or the service
    manager refused. ``job`` is the service found (if any), whose settings
    the direct start must then carry.
    """
    job = find_job(port)
    if job is None:
        logger.debug("no installed service for port %d; starting the daemon directly", port)
        return None, None
    outcome = request_start(job)
    if outcome.started:
        logger.info("asked the installed service %s to start the daemon (%s)",
                    job.name, outcome.how)
        return wait_for_daemon(), job
    logger.warning(
        "could not start the daemon through the installed service %s (%s); "
        "starting it directly with the service's settings", job.name, outcome.detail,
    )
    return None, job


# --- direct start environment ----------------------------------------------------

def _client_scoped(key: str) -> bool:
    return key in CLIENT_SCOPED_ENV or key.startswith(_CLIENT_SCOPED_PREFIXES)


def daemon_environment(
    base: Mapping[str, str], job: SupervisorJob | None = None,
) -> dict[str, str]:
    """Environment for a directly started daemon.

    Starts from ``base`` without any client-scoped variable, then applies the
    service definition's own variables (they win: the service is the daemon's
    configuration), so a fallback start still runs with the service's
    settings. Returns a new dict; ``base`` is not modified.
    """
    environment = {k: v for k, v in base.items() if not _client_scoped(k)}
    if job is not None:
        environment.update(
            {k: v for k, v in job.environment.items() if not _client_scoped(k)}
        )
    return environment
