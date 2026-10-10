"""The record of a daemon started with ``python -m`` must reach ``ready``.

Running ``python -m superlocalmemory.server.unified_daemon`` loads the module as
``__main__`` while uvicorn imports ``superlocalmemory.server.unified_daemon``
again for the app. Two copies of the module mean two copies of the active
descriptor: the app copy published ``ready`` and the record guardian, holding
the ``__main__`` copy, kept writing ``starting`` back every few seconds.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SLM_SKIP_REAL_DAEMON") == "1",
    reason="real daemon start disabled by SLM_SKIP_REAL_DAEMON=1",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _record_state(data_dir: Path) -> str | None:
    try:
        return json.loads((data_dir / "daemon.json").read_text()).get("state")
    except (OSError, ValueError):
        return None


def test_module_run_daemon_record_reaches_ready(tmp_path: Path) -> None:
    port = _free_port()
    env = {
        **os.environ,
        "SLM_DATA_DIR": str(tmp_path),
        "SLM_DAEMON_PORT": str(port),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
         "--start", f"--port={port}"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        state = None
        while time.monotonic() < deadline:
            state = _record_state(tmp_path)
            if state == "ready":
                break
            assert proc.poll() is None, "daemon exited before becoming ready"
            time.sleep(0.5)
        assert state == "ready"
        # The guardian checks every 5 s; across two checks the record must stay ready.
        time.sleep(11)
        assert _record_state(tmp_path) == "ready"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
