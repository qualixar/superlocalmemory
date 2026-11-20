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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from superlocalmemory.cli.service_inventory import ServiceDefinition

logger = logging.getLogger(__name__)

#: Variables that describe the client process, never the shared daemon. A
#: daemon that inherits them attributes every caller's work to one agent,
#: reports one client's tool profile as its own, or reuses the client's
#: identity material. They are removed from every directly started daemon.
#: The session ids matter most: the daemon's HTTP MCP mesh tools take their
#: session from its environment, so an inherited one let any credential-less
#: local caller read that agent's inbox and send as it.
CLIENT_SCOPED_ENV = frozenset({
    "SLM_AGENT_ID",
    "SLM_CLIENT",
    "SLM_CLIENT_CHANNEL",
    "SLM_SESSION_ID",
    "CLAUDE_SESSION_ID",
    "CLAUDE_CODE_SESSION_ID",
    "SLM_USER_SESSION",
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
#: How long a started job may show no process before it is taken as not
#: running. launchd and systemd spawn within a second; the service's own
#: restart throttle (30 s) is far longer, so a crash loop is caught here.
_NO_PID_GRACE_S = 5.0


@dataclass(frozen=True)
class SupervisorJob:
    """One installed service definition proven to own this data folder and port."""

    platform: str
    name: str
    path: Path
    port: int
    environment: Mapping[str, str] = field(default_factory=dict)
    #: The verified definition. The loaded job must match it before it is
    #: started or trusted; a job without one is never trusted.
    definition: ServiceDefinition | None = None


@dataclass(frozen=True)
class JobState:
    """What the service manager holds under a job's name, checked against its file.

    ``running`` / ``idle``: loaded and exactly the installed definition.
    ``not_loaded``: nothing is loaded under the name.
    ``stale``: something else is loaded under the name (a definition edited
    or replaced without a reload, or another installation's job).
    ``unknown``: the service manager could not be asked or understood.
    """

    status: str
    pid: int | None = None
    detail: str = ""


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
        jobs.append(SupervisorJob(platform, row.name, row.path, row.port, environment, row))
    return jobs


def find_job(port: int) -> SupervisorJob | None:
    jobs = jobs_for(port)
    return jobs[0] if jobs else None


def _macos_domain() -> str:
    return f"gui/{os.getuid()}"


def _owned_pid(raw) -> int | None:
    pid = int(raw) if raw is not None else 0
    return pid if pid > 0 else None


def _macos_state(job: SupervisorJob) -> JobState:
    from superlocalmemory.cli.service_live import macos_owned_record

    record = macos_owned_record(job.definition)
    if record is None:
        return JobState("not_loaded")
    pid = _owned_pid(record.get("PID"))
    return JobState("running" if pid else "idle", pid)


def _linux_properties(job: SupervisorJob) -> dict[str, str]:
    from superlocalmemory.cli.service_live import ManagerQueryFailed

    result = _run(["systemctl", "--user", "show", job.name,
                   "--property=LoadState", "--property=MainPID"])
    if result.returncode:
        raise ManagerQueryFailed("Cannot inspect systemd user unit")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _linux_state(job: SupervisorJob) -> JobState:
    from superlocalmemory.cli.service_live import verified_live

    verified_live("linux", job.definition)  # raises unless systemd runs this file
    properties = _linux_properties(job)
    if properties.get("LoadState") != "loaded":
        return JobState("not_loaded")
    pid = _owned_pid(properties.get("MainPID"))
    return JobState("running" if pid else "idle", pid)


def _windows_state(job: SupervisorJob) -> JobState:
    # Logon tasks hand the daemon to a script host and never restart it, so
    # the task has no daemon PID to report.
    found = _run(["schtasks", "/Query", "/TN", job.name]).returncode == 0
    return JobState("idle" if found else "not_loaded")


_STATES = {"macos": _macos_state, "linux": _linux_state, "windows": _windows_state}


def job_state(job: SupervisorJob) -> JobState:
    """The loaded job under ``job``'s name, trusted only if it is ``job``. Never raises."""
    from superlocalmemory.cli.service_live import JobMismatch

    if job.definition is None and job.platform != "windows":
        return JobState("unknown", detail="no verified definition")
    try:
        return _STATES[job.platform](job)
    except JobMismatch as exc:
        return JobState("stale", detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - an unreadable manager is never trusted
        return JobState("unknown", detail=f"{type(exc).__name__}: {exc}")


def job_loaded(job: SupervisorJob) -> bool:
    """Is the service manager holding exactly this definition?"""
    return job_state(job).status in {"running", "idle"}


def job_pid(job: SupervisorJob) -> int | None:
    """PID of the process the service manager runs for exactly this definition.

    None when nothing runs, when the loaded job is not this definition, when
    the manager cannot be asked, and always on Windows.
    """
    state = job_state(job)
    return state.pid if state.status == "running" else None


_STALE = ("the loaded service job does not match its installed definition "
          "(edited or replaced without a reload)")


# --- start -----------------------------------------------------------------------

def _refusal(state: JobState) -> StartOutcome | None:
    if state.status == "stale":
        return StartOutcome(False, "failed", f"{_STALE}: {state.detail}")
    if state.status == "unknown":
        return StartOutcome(False, "failed", f"cannot inspect the service job: {state.detail}")
    return None


def _start_macos(job: SupervisorJob) -> StartOutcome:
    domain = _macos_domain()
    target = f"{domain}/{job.name}"
    state = job_state(job)
    if (refused := _refusal(state)) is not None:
        return refused
    how = "kickstarted"
    if state.status == "not_loaded":
        # The definition is installed but not loaded (for example after
        # ``launchctl bootout``). launchd loads it at the next login anyway;
        # loading it now keeps the daemon under its owner instead of
        # starting an unowned copy that the service would later collide with.
        loaded = _run(["launchctl", "bootstrap", domain, str(job.path)])
        after = job_state(job)  # another client may have loaded it first
        if (refused := _refusal(after)) is not None:
            return refused
        if after.status == "not_loaded":
            return StartOutcome(False, "failed", (loaded.stderr or "").strip()
                                or f"launchctl bootstrap exited {loaded.returncode}")
        how = "loaded" if loaded.returncode == 0 else how
    # Always kick, even right after loading: a loaded job is not a running
    # one (it may be disabled, or RunAtLoad may be off). Without -k a job
    # that is already running is left alone.
    kicked = _run(["launchctl", "kickstart", target])
    if kicked.returncode == 0:
        return StartOutcome(True, how)
    return StartOutcome(False, "failed", (kicked.stderr or "").strip()
                        or f"launchctl kickstart exited {kicked.returncode}")


def _start_linux(job: SupervisorJob) -> StartOutcome:
    if (refused := _refusal(job_state(job))) is not None:
        return refused
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
    except Exception as exc:  # noqa: BLE001 - a failed ask means a direct start
        return StartOutcome(False, "failed", f"{type(exc).__name__}: {exc}")


class ServiceProcess:
    """A ``Popen``-like view of the daemon the service manager runs for a job.

    The readiness wait polls it like a child process. ``poll()`` stays None
    while the manager runs a process for the job, and while it may still be
    spawning one; it returns 1 (and keeps returning it) once the manager has
    reported the job with no process for ``_NO_PID_GRACE_S``: the job is
    disabled, or its program exits at once. It never reports an exit it
    cannot observe -- on Windows, when the manager cannot be asked, or when
    the loaded job stopped matching its definition, it stays None and the
    wait runs its normal course, so a slow service daemon is never doubled.
    """

    def __init__(self, job: SupervisorJob, clock=time.monotonic):
        self.job, self._clock = job, clock
        self._quiet_since = clock()
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None or self.job.platform == "windows":
            return self.returncode
        if job_state(self.job).status not in {"idle", "not_loaded"}:
            self._quiet_since = self._clock()
            return None
        if self._clock() - self._quiet_since >= _NO_PID_GRACE_S:
            self.returncode = 1
        return self.returncode


def _started_nothing(job: SupervisorJob, nothing_started) -> bool:
    """After a full wait: no service process, and no daemon anywhere near the folder.

    The service manager must report no process for the job (always so on
    Windows, whose task has no daemon PID), and ``nothing_started()`` must
    confirm that no daemon record names a live process, the instance lock is
    free and nothing listens on the port. Any doubt means "something started".
    """
    if nothing_started is None or job_pid(job) is not None:
        return False
    try:
        return bool(nothing_started())
    except Exception as exc:  # noqa: BLE001 - doubt must never double a daemon
        logger.debug("cannot tell whether the service started a daemon: %s", exc)
        return False


def start_through_service(
    port: int, wait_for_daemon, *, nothing_started=None,
) -> tuple[bool | None, SupervisorJob | None]:
    """Start the daemon through its installed service when there is one.

    Returns ``(ready, job)``. ``ready`` is None when the caller must start the
    daemon directly: no service owns this folder and port, the service
    manager refused, it accepted the start but runs no daemon for the job, or
    the wait ran out with no trace of a daemon (``nothing_started``; without
    it a timed-out wait is a failure). ``job`` is the service found (if any),
    whose settings the direct start must then carry. ``wait_for_daemon(proc)``
    waits for the daemon, where ``proc`` is a ``ServiceProcess`` it may poll
    like a child.
    """
    job = find_job(port)
    if job is None:
        logger.debug("no installed service for port %d; starting the daemon directly", port)
        return None, None
    outcome = request_start(job)
    if not outcome.started:
        logger.warning(
            "could not start the daemon through the installed service %s (%s); "
            "starting it directly with the service's settings", job.name, outcome.detail,
        )
        return None, job
    logger.info("asked the installed service %s to start the daemon (%s)",
                job.name, outcome.how)
    process = ServiceProcess(job)
    ready = wait_for_daemon(process)
    if ready:
        return ready, job
    if process.returncode is not None:
        logger.warning(
            "the installed service %s accepted the start but runs no daemon "
            "(disabled, or its program exits at once); starting it directly with "
            "the service's settings", job.name,
        )
        return None, job
    if _started_nothing(job, nothing_started):
        logger.warning(
            "the installed service %s accepted the start but no daemon appeared "
            "(its program may point at a Python that is gone); starting it "
            "directly with the service's settings. Run `slm serve install` to "
            "repair the service", job.name,
        )
        return None, job
    return ready, job


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
