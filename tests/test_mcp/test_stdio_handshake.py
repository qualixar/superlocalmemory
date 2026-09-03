# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""MCP stdio handshake: notifications/initialized must never be rejected.

Issue #128 Bug 7: standard MCP clients send ``notifications/initialized``
after ``initialize`` and one client library received
``{"code": -32601, "message": "Method not found"}``. The stdio path runs
the pinned MCP SDK, which absorbs that notification gracefully — this test
locks the behavior with a live child over the same JSON-RPC round trip a
real IDE performs. No daemon is involved (tools/list needs none), and the
child environment is fully constructed: isolated data root, warmup side
effects off, production ports never touched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

_PASSTHROUGH_VARS = ("PATH", "LANG", "LC_ALL", "TMPDIR", "TERM")
_PROXY_VARS = (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
)


def _handshake_env(data_root: Path, home: Path) -> dict:
    env = {name: os.environ[name] for name in _PASSTHROUGH_VARS if name in os.environ}
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(SRC_ROOT),
            "SLM_DATA_DIR": str(data_root),
            "SLM_MCP_TOOLS": "remember,recall",
            # Never auto-start or contact any daemon from this lane.
            "SLM_DISABLE_WARMUP_SIDE_EFFECTS": "1",
            "CI": "1",
            "SLM_NON_INTERACTIVE": "1",
            "SLM_TEST_ISOLATION": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for var in _PROXY_VARS:
        env.pop(var, None)
    return env


class _HandshakeClient:
    """Minimal newline-delimited JSON-RPC driver recording EVERY message."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc
        self._responses: dict[object, dict] = {}
        self._seen: list[dict] = []
        self._reader = threading.Thread(target=self._read_forever, daemon=True)
        self._reader.start()

    def _read_forever(self) -> None:
        try:
            for raw in self._proc.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    self._seen.append(message)
                    if "id" in message:
                        self._responses[message["id"]] = message
        except Exception:
            pass

    def call(self, method: str, params: dict | None = None, *, timeout: float = 180.0) -> dict:
        request_id = len(self._responses) + len([m for m in self._seen if "id" not in m]) + 1
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if request_id in self._responses:
                return self._responses[request_id]
            if self._proc.poll() is not None:
                raise AssertionError(f"mcp child exited rc={self._proc.returncode} during {method}")
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {method} response")

    def notify(self, method: str, params: dict | None = None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10)


def test_notifications_initialized_is_gracefully_absorbed(tmp_path) -> None:
    """#128 Bug 7: initialized notification draws no -32601 and kills nothing.

    4.1.14 audit: the stderr log the child writes is INSPECTED (a -32601
    on stderr, or a child that dies right after the notification, fails
    even if stdout stays quiet); readiness is polled via tools/list
    instead of a fixed sleep; and the PYTHONPATH choice is deliberate —
    this lane tests the REPO checkout, while the shipped npm launcher
    strips PYTHONPATH and resolves only its venv (see
    test_npm_runtime_isolation).
    """
    data_root = tmp_path / "slm-data"
    home = tmp_path / "home"
    data_root.mkdir()
    home.mkdir()
    stderr_path = tmp_path / "mcp-stderr.log"
    with stderr_path.open("wb") as stderr_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.cli.main", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=stderr_log, env=_handshake_env(data_root, home),
            cwd=str(REPO_ROOT),
        )
    client = _HandshakeClient(proc)
    try:
        initialized = client.call("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "ws7-handshake", "version": "1.0"},
        })
        assert "result" in initialized, initialized

        # The exact reporter shape: a bare notification, no id, so a
        # compliant server answers nothing and stays alive.
        client.notify("notifications/initialized")

        # Poll for post-notification readiness instead of sleeping: the
        # first tools/list that answers proves the connection survived.
        listed = None
        deadline = time.monotonic() + 60.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            try:
                listed = client.call("tools/list", {}, timeout=10.0)
                break
            except AssertionError as exc:
                last_error = exc
        assert proc.poll() is None, (
            "mcp child died on notifications/initialized; "
            f"stderr tail: {stderr_path.read_text(errors='replace')[-1500:]}"
        )
        assert listed is not None, f"tools/list never answered: {last_error}"
        assert "result" in listed, listed
        tool_names = {tool["name"] for tool in listed["result"]["tools"]}
        assert {"remember", "recall"} <= tool_names, tool_names

        for message in client._seen:
            error = message.get("error") or {}
            assert error.get("code") != -32601, message
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        assert "-32601" not in stderr_text, stderr_text[-1500:]
        assert "Method not found" not in stderr_text, stderr_text[-1500:]
    finally:
        client.close()
        assert proc.poll() is not None, "mcp child must exit after stdin EOF"
