# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A service-started daemon takes the data folder back from an unowned one.

The installed service (LaunchAgent / systemd user unit) is the daemon's owner
and its definition is the daemon's configuration. When a client started a
daemon directly (an older client, or one whose service start failed) and the
service later starts its own, the service's daemon used to find the folder
taken, exit 0, and leave the unowned daemon running with the client's
settings while the service manager -- which only restarts on failure --
reported nothing wrong.

``take_over_if_supervised`` runs once at daemon start, before the instance
lock is requested. If, and only if, this process *is* the service's job and
the folder is held by a live daemon of this data folder that is *not* any
service's job, it asks that daemon to stop through the same authenticated
``/stop`` that ``slm restart`` uses (graceful drain), then lets the normal
lock acquisition proceed. It tries once per process: a failed hand-over keeps
the old exit-0 behaviour, so a healthy unowned daemon never causes a respawn
loop, and ``slm doctor`` reports the unowned daemon.

Windows logon tasks are launchers, not supervisors (they never restart the
daemon), so nothing here applies there.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_HANDOVER_WAIT_S = 30.0


def _own_job_names(jobs, pid: int, job_pid) -> tuple[list[str], set[int]]:
    """Names of jobs this process runs, and the PIDs of every job's process."""
    mine, pids = [], set()
    for job in jobs:
        found = job_pid(job)
        if found is None:
            continue
        pids.add(found)
        if found == pid:
            mine.append(job.name)
    return mine, pids


def take_over_if_supervised(port: int) -> str:
    """Hand the data folder to this service-started daemon. Never raises.

    Returns what happened, for the log and for tests: ``unsupervised``,
    ``free``, ``holder_unknown``, ``holder_supervised``, ``took_over``,
    ``stop_failed`` or ``error``.
    """
    try:
        return _take_over(port)
    except Exception as exc:  # noqa: BLE001 - a start must never fail here
        logger.warning("service hand-over check failed: %s", exc)
        return "error"


def _take_over(port: int) -> str:
    from superlocalmemory.cli import daemon as client
    from superlocalmemory.cli import daemon_supervisor as supervisor
    from superlocalmemory.infra.instance_lock import (
        instance_lock_is_held,
        wait_until_instance_lock_free,
    )

    jobs = supervisor.jobs_for(port)
    if not jobs:
        return "unsupervised"
    me = os.getpid()
    mine, _ = _own_job_names(jobs, me, supervisor.job_pid)
    if not mine:
        return "unsupervised"
    if not instance_lock_is_held():
        return "free"
    holder = client.read_descriptor()
    if holder is None or not client._descriptor_process_is_alive(holder):
        # Cannot prove who holds the folder: never stop an unknown process.
        return "holder_unknown"
    # Every service of this folder counts, whatever its port (#338): stopping a
    # daemon another of its services runs would let the two take the folder
    # from each other forever.
    _, job_pids = _own_job_names(supervisor.jobs_for(None), me, supervisor.job_pid)
    if holder.pid in job_pids:
        logger.error(
            "SLM daemon will not start: pid %d, run by another installed service "
            "for this data folder, already owns it", holder.pid,
        )
        return "holder_supervised"
    logger.warning(
        "SLM daemon pid %d owns this data folder but was not started by the "
        "installed service %s; asking it to hand over so the service's daemon "
        "and settings take effect", holder.pid, mine[0],
    )
    # Bound to the holder checked above: if another daemon has replaced it
    # since, nothing is stopped.
    if (client.stop_daemon(expected=holder)
            and wait_until_instance_lock_free(timeout_s=_HANDOVER_WAIT_S)):
        logger.warning("hand-over complete: pid %d stopped", holder.pid)
        return "took_over"
    logger.error(
        "SLM daemon pid %d did not hand over; this service will not retry. "
        "Run `slm restart` to move the daemon under the service.", holder.pid,
    )
    return "stop_failed"
