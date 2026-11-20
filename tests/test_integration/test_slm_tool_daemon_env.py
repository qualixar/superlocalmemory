# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A daemon that ``slm tool`` starts never carries the caller's identity.

``slm tool`` runs in an agent's shell, where the agent's session id, agent name
and channel are set. When no daemon is running it starts one, and that daemon
serves every local caller over HTTP MCP. If the daemon inherited the caller's
session, a credential-less ``POST /mcp/`` ``mesh_inbox`` read that agent's
private messages (and could send as it); if it inherited the agent name, a
credential-less ``remember`` was recorded as that agent.

One real daemon, started by ``slm tool`` itself (private port, its own data root
and HOME; the daemon on 8765 is never touched). A ``sitecustomize`` records the
environment the daemon process actually received.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
)

TAG = uuid.uuid4().hex[:8]

#: What the caller's shell carries; none of it may reach the shared daemon.
CALLER_IDENTITY = {
    "SLM_SESSION_ID": f"caller-session-{TAG}",
    "CLAUDE_SESSION_ID": f"caller-claude-{TAG}",
    "CLAUDE_CODE_SESSION_ID": f"caller-claude-code-{TAG}",
    "SLM_AGENT_ID": f"caller-agent-{TAG}",
    "SLM_CLIENT_CHANNEL": "mcp",
    "SLM_USER_SESSION": f"caller-user-session-{TAG}",
}
#: What the daemon genuinely needs from the same shell; all of it must arrive.
DAEMON_SETTINGS = {"OPENROUTER_API_KEY": "test-provider-key-not-real", "SLM_RANKING": "test-ranking"}

_DUMPER = '''
import json, os, sys
_dir = os.environ.get("SLM_TEST_ENV_DUMP_DIR")
if _dir and "superlocalmemory.server.unified_daemon" in getattr(sys, "orig_argv", []):
    with open(os.path.join(_dir, f"daemon-{os.getpid()}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(os.environ), fh)
'''


class Started:
    def __init__(self, root: Path, env: dict[str, str], port: int, data_root: Path) -> None:
        self.root, self.env, self.port, self.data_root = root, env, port, data_root

    def cli(self, *args: str, **extra: str) -> tuple[int, dict]:
        env = {**self.env, **extra}
        proc = subprocess.run(
            [sys.executable, "-m", "superlocalmemory.cli.main", "tool", *args],
            capture_output=True, text=True, timeout=240, env=env, cwd=str(self.root / "home"))
        assert proc.stdout.strip(), f"no output (rc={proc.returncode}): {proc.stderr[-3000:]}"
        return proc.returncode, json.loads(proc.stdout)

    def http_mcp(self, name: str, arguments: dict) -> dict:
        """One credential-less ``tools/call`` on the daemon's HTTP MCP door."""
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                           "params": {"name": name, "arguments": arguments}}).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/mcp/", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Accept": "application/json, text/event-stream"})
        with urllib.request.urlopen(request, timeout=120) as response:
            text = response.read().decode("utf-8").strip()
        if not text.startswith("{"):
            text = next(line[5:].strip() for line in text.splitlines() if line.startswith("data:"))
        result = json.loads(text)["result"]
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        return json.loads(next(b["text"] for b in result["content"] if b["type"] == "text"))

    def daemon_env(self) -> dict[str, str]:
        dumps = sorted((self.root / "env-dumps").glob("daemon-*.json"))
        assert dumps, "the daemon was not started by slm tool"
        return json.loads(dumps[-1].read_text(encoding="utf-8"))


def _stop(data_root: Path, foreign_before: set[int]) -> None:
    descriptor = data_root / "daemon.json"
    try:
        pid = int(json.loads(descriptor.read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError):
        return
    if pid in foreign_before or pid == os.getpid():
        return
    try:
        os.kill(pid, 15)
        for _ in range(100):
            os.kill(pid, 0)
            time.sleep(0.1)
        os.kill(pid, 9)
    except OSError:
        pass


@pytest.fixture(scope="module")
def started(tmp_path_factory):
    root = tmp_path_factory.mktemp("slm-tool-env")
    data_root = root / "data"
    data_root.mkdir()
    (root / "home").mkdir()
    (root / "env-dumps").mkdir()
    custom = root / "sitecustomize"
    custom.mkdir()
    (custom / "sitecustomize.py").write_text(_DUMPER, encoding="utf-8")
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    env = _child_env(data_root, port, root / "home", root / "cache")
    env["PYTHONPATH"] = os.pathsep.join([str(custom), env["PYTHONPATH"]])
    env["SLM_TEST_ENV_DUMP_DIR"] = str(root / "env-dumps")
    env["SLM_TEST_ALLOW_DAEMON_SPAWN"] = "1"  # this test is about the daemon slm tool starts
    env.update(DAEMON_SETTINGS)
    foreign_before = _foreign_daemon_pids()
    found = Started(root, env, port, data_root)
    try:
        # The caller's first call starts the daemon: a mesh tool, so the session is bound too.
        rc, registered = found.cli("mesh_summary", "--summary", "caller at work",
                                   **CALLER_IDENTITY)
        assert rc == 0 and registered.get("registered") is True, registered
        found.caller_peer = registered["peer_id"]
        yield found
    finally:
        _stop(data_root, foreign_before)


def test_the_daemon_started_by_slm_tool_has_no_caller_identity(started):
    env = started.daemon_env()
    leaked = sorted(k for k in CALLER_IDENTITY if k in env)
    assert leaked == [], f"the daemon inherited the caller's {leaked}"


def test_the_daemon_keeps_its_own_settings(started):
    """The scrub removes identity, not configuration: data folder, port, provider keys."""
    env = started.daemon_env()
    assert env["SLM_DATA_DIR"] == str(started.data_root)
    assert env["SLM_DAEMON_PORT"] == str(started.port)
    for key, value in DAEMON_SETTINGS.items():
        assert env.get(key) == value, key


def test_a_credential_less_http_inbox_does_not_read_the_callers_messages(started):
    """The realistic caller: a Claude Code shell, whose mesh session is CLAUDE_SESSION_ID."""
    claude_caller = {k: v for k, v in CALLER_IDENTITY.items() if k != "SLM_SESSION_ID"}
    rc, registered = started.cli("mesh_summary", "--summary", "claude caller", **claude_caller)
    assert rc == 0 and registered.get("registered") is True, registered
    rc, sent = started.cli("mesh_send", "--to", registered["peer_id"],
                           "--message", f"private for the caller {TAG}",
                           "--session", f"sender-{TAG}", SLM_AGENT_ID="sender")
    assert rc == 0 and sent.get("ok") is True, sent

    anonymous = started.http_mcp("mesh_inbox", {})
    contents = [m.get("content") for m in anonymous.get("messages", [])]
    assert f"private for the caller {TAG}" not in contents, anonymous
    assert anonymous.get("peer_id") not in (started.caller_peer, registered["peer_id"]), anonymous

    # Control: the caller itself still sees the message.
    rc, own = started.cli("mesh_inbox", **claude_caller)
    assert rc == 0 and f"private for the caller {TAG}" in [m["content"] for m in own["messages"]]


def test_a_credential_less_http_remember_is_not_the_callers_agent(started):
    saved = started.http_mcp("remember", {"content": f"anonymous note {TAG}",
                                          "idempotency_key": f"anon-{TAG}"})
    assert saved.get("success") is True, saved
    conn = sqlite3.connect(f"file:{started.data_root / 'memory.db'}?mode=ro", uri=True, timeout=30)
    try:
        rows = conn.execute("SELECT metadata_json FROM memories WHERE content LIKE ?",
                            (f"%anonymous note {TAG}%",)).fetchall()
    finally:
        conn.close()
    assert rows, "the memory was not saved"
    meta = json.loads(rows[0][0] or "{}")
    assert meta.get("agent_id") != CALLER_IDENTITY["SLM_AGENT_ID"], meta
