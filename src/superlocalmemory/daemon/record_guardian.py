# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Keep the daemon record true for as long as this process owns the data folder.

The instance-lock holder is the daemon of its data folder, so a record that does
not name this process is wrong by construction (deleted, damaged, or written by
something else). The guardian notices within one interval and publishes the
record again. It never runs, and never writes, in a process without the lock.
"""

from __future__ import annotations

import hmac
import logging
import threading
from pathlib import Path
from typing import Callable

from superlocalmemory.infra.daemon_identity import (
    DaemonDescriptor,
    descriptor_path,
    publish_if_owner,
    read_descriptor,
)
from superlocalmemory.infra.instance_lock import InstanceLock

logger = logging.getLogger(__name__)


class RecordGuardian:
    """Poll the on-disk record and restore it when it stops naming this daemon."""

    name = "record_guardian"

    def __init__(
        self,
        *,
        descriptor_provider: Callable[[], DaemonDescriptor | None],
        lock: InstanceLock,
        interval_s: float = 5.0,
        data_root: str | Path | None = None,
    ) -> None:
        self._provider = descriptor_provider
        self._lock = lock
        self._interval = interval_s
        self._data_root = data_root
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "stopped"
        self._detail = ""
        self._lock_loss_logged = False

    def _drift_reason(self, ours: DaemonDescriptor) -> str | None:
        """Why the record is wrong, or None when it already names this process."""
        recorded = read_descriptor(data_root=self._data_root)
        if recorded is None:
            exists = descriptor_path(self._data_root).exists()
            return "unreadable" if exists else "missing"
        if not hmac.compare_digest(recorded.instance_id, ours.instance_id):
            return f"names another instance (pid {recorded.pid})"
        if recorded.state != ours.state or recorded.port != ours.port:
            return f"state or port differs ({recorded.state}, {recorded.port})"
        base = descriptor_path(self._data_root)
        for name, expected in (("daemon.pid", ours.pid), ("daemon.port", ours.port)):
            try:
                text = base.with_name(name).read_text(encoding="utf-8").strip()
            except OSError:
                return f"mirror {name} missing"
            if text != str(expected):
                return f"mirror {name} differs"
        return None

    def check_once(self) -> str:
        """One pass: ``ok``, ``republished``, ``not_owner``, ``lock_lost`` or
        ``no_descriptor``."""
        if not self._lock.held:
            return "not_owner"
        if not self._lock.still_owns_file():
            if not self._lock_loss_logged:
                self._lock_loss_logged = True
                logger.error(
                    "the data-folder lock file was removed or replaced; this "
                    "daemon no longer writes the record",
                )
            return "lock_lost"
        ours = self._provider()
        if ours is None:
            return "no_descriptor"
        reason = self._drift_reason(ours)
        if reason is None:
            return "ok"
        logger.warning("daemon record was wrong (%s); republishing it", reason)
        # Publish what is current now, never a state read before the check.
        publish_if_owner(self._provider() or ours, self._lock)
        return "republished"

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._detail = self.check_once()
                self._state = "running"
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.warning("record guardian check failed: %s", exc)
                self._state, self._detail = "failed", str(exc)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._state = "running"
        self._thread = threading.Thread(
            target=self._run, name="slm-record-guardian", daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout_s)
        self._state = "stopped"

    def health(self) -> dict:
        return {"state": self._state, "detail": self._detail}
