# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""``slm tool`` against a REAL daemon, beside an ``slm mcp`` stdio child on the same daemon.

One isolated daemon (private port, its own data root and HOME; the daemon on
8765 is never touched) serves both doors. For one tool of each family the
command line and the MCP server are asked the same thing and must answer in
the same shape: memory, pictures, the agent mesh, the code graph and a
learning signal. Then the boundaries: a tool meant only for other computers
stays refused, the channel the daemon is told is ``cli`` even when the
environment says ``mcp`` (and the MCP child, as a control, says ``mcp``), and
a channel or identity named in the arguments is refused or ignored.

Every outgoing daemon request of the child processes is recorded by a small
``sitecustomize`` that wraps ``core.outbound_http.urlopen``: headers only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import subprocess
import sys
import uuid
import zlib
from pathlib import Path

import pytest

from tests.test_integration.test_per_request_profile_e2e import (
    PRODUCTION_PORTS,
    REPO_ROOT,
    RealDaemon,
    _child_env,
    _foreign_daemon_pids,
    _reserve_private_port,
    _RpcClient,
)

TAG = uuid.uuid4().hex[:8]

_RECORDER = '''
import json, os
_log = os.environ.get("SLM_TEST_HEADER_LOG")
if _log:
    from superlocalmemory.core import outbound_http as _o
    _real = _o.urlopen
    def urlopen(req, *args, **kwargs):
        try:
            headers = {k.lower(): v for k, v in req.header_items()}
            headers.pop("x-slm-daemon-capability", None)
            with open(_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"url": req.full_url, "headers": headers}) + "\\n")
        except Exception:
            pass
        return _real(req, *args, **kwargs)
    _o.urlopen = urlopen
    import atexit, sys
    def _at_exit():
        with open(_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"exit": True,
                                 "stdio_server_imported": "superlocalmemory.mcp.server" in sys.modules,
                                 "channel": os.environ.get("SLM_CLIENT_CHANNEL")}) + "\\n")
    atexit.register(_at_exit)
'''


def _png() -> bytes:
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Doors:
    """The daemon, an ``slm mcp`` child and a way to run ``slm tool``."""

    def __init__(self, daemon: RealDaemon, root: Path) -> None:
        self.daemon = daemon
        self.root = root
        self.home = Path(daemon.env["HOME"])
        self.custom = root / "sitecustomize"
        self.custom.mkdir()
        (self.custom / "sitecustomize.py").write_text(_RECORDER, encoding="utf-8")
        self.mcp: _RpcClient | None = None

    def env(self, log: Path | None = None, **extra: str) -> dict[str, str]:
        env = dict(self.daemon.env)
        env["PYTHONPATH"] = os.pathsep.join([str(self.custom), env["PYTHONPATH"]])
        env["SLM_DISABLE_WARMUP_SIDE_EFFECTS"] = "1"  # never start another daemon
        if log is not None:
            env["SLM_TEST_HEADER_LOG"] = str(log)
        env.update(extra)
        return env

    def cli(self, *args: str, log: Path | None = None, **extra: str) -> tuple[int, dict]:
        proc = subprocess.run(
            [sys.executable, "-m", "superlocalmemory.cli.main", "tool", *args],
            capture_output=True, text=True, timeout=180, env=self.env(log, **extra),
            cwd=str(self.home))
        assert proc.stdout.strip(), f"no output (rc={proc.returncode}): {proc.stderr[-3000:]}"
        return proc.returncode, json.loads(proc.stdout)

    def start_mcp(self, log: Path) -> None:
        env = self.env(log, SLM_MCP_ALL_TOOLS="1", SLM_AGENT_ID="mcp-agent",
                       SLM_SESSION_ID=f"mcp-{TAG}")
        stderr = self.root / "mcp-stderr.log"
        with stderr.open("wb") as err:
            proc = subprocess.Popen([sys.executable, "-m", "superlocalmemory.cli.main", "mcp"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                    env=env, cwd=str(self.home))
        self.mcp = _RpcClient(proc, stderr)
        assert "result" in self.mcp.call("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "slm-tool-e2e", "version": "1"}})
        self.mcp.notify("notifications/initialized")

    def via_mcp(self, name: str, arguments: dict) -> tuple[bool, dict]:
        """``(is_error, payload)`` of one MCP ``tools/call``, the same rule as the CLI's."""
        assert self.mcp is not None
        reply = self.mcp.call("tools/call", {"name": name, "arguments": arguments})
        result = reply["result"]
        if result.get("structuredContent") is not None:
            payload = result["structuredContent"]
        else:
            texts = [b["text"] for b in result["content"] if b["type"] == "text"]
            try:
                payload = json.loads(texts[0])
            except ValueError:
                payload = {"text": texts[0]}
        return bool(result.get("isError")), payload


@pytest.fixture(scope="module")
def doors(tmp_path_factory):
    root = tmp_path_factory.mktemp("slm-tool-e2e")
    data_root = root / "data"
    data_root.mkdir()
    port = _reserve_private_port()
    assert port not in PRODUCTION_PORTS
    env = _child_env(data_root, port, root / "home", root / "cache")
    (root / "home").mkdir(exist_ok=True)
    foreign_before = _foreign_daemon_pids()
    stdout_log = root / "daemon-stdout.log"
    with stdout_log.open("wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon",
             "--start", f"--port={port}"],
            stdout=log_file, stderr=log_file, env=env, cwd=str(REPO_ROOT),
            start_new_session=os.name == "posix")
    daemon = RealDaemon(proc, port, data_root, stdout_log, env)
    found = Doors(daemon, root)
    try:
        daemon.wait_ready()
        daemon.wait_health_fast()
        found.start_mcp(root / "mcp-headers.jsonl")
        yield found
    finally:
        if found.mcp is not None:
            found.mcp.close()
        daemon.stop(foreign_before)


def _as_cli_failure(mcp_answer: object) -> object:
    """An MCP failure as ``slm tool`` reports it: the same body, ``error`` as {code, message}."""
    from superlocalmemory.cli.tool_host import _failure_body  # noqa: PLC0415

    return _failure_body(mcp_answer)


def _same_shape(left: object, right: object) -> bool:
    """Same keys at the top level (and the same type for each), values aside."""
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            type(left[k]) is type(right[k]) or None in (left[k], right[k]) for k in left)
    return type(left) is type(right)


# -- one tool of each family, both doors -----------------------------------------------------

def test_memory_remember_and_recall_match_mcp(doors):
    rc, saved = doors.cli("remember", "--content", f"SlmTool{TAG} ferry schedule moves to Pier 7",
                          "--idempotency-key", f"cli-{TAG}")
    assert rc == 0 and saved["success"] is True and saved["fact_ids"], saved
    is_error, mcp_saved = doors.via_mcp("remember", {
        "content": f"SlmTool{TAG} ferry crew rota is on the notice board",
        "idempotency_key": f"mcp-{TAG}"})
    assert not is_error and {"success", "fact_ids", "count"} <= set(saved) & set(mcp_saved)
    rc, found = doors.cli("recall", "--query", f"SlmTool{TAG} ferry", "--limit", "5")
    is_error, mcp_found = doors.via_mcp("recall", {"query": f"SlmTool{TAG} ferry", "limit": 5})
    assert rc == 0 and not is_error
    assert _same_shape(found, mcp_found), (sorted(found), sorted(mcp_found))
    contents = " ".join(r["content"] for r in found["results"])
    assert "Pier 7" in contents and "notice board" in contents  # one store, both doors


def test_pictures_answer_the_same_through_both_doors(doors):
    picture = doors.home / "tiny.png"
    picture.write_bytes(_png())
    args = {"path": str(picture), "content": f"a tiny red square {TAG}"}
    rc, cli_answer = doors.cli("remember_media", "--json", json.dumps(args))
    is_error, mcp_answer = doors.via_mcp("remember_media", args)
    assert (rc == 1) == is_error
    expected = _as_cli_failure(mcp_answer) if is_error else mcp_answer
    assert cli_answer == expected or _same_shape(cli_answer, expected), (cli_answer, expected)
    missing = {"media_id": "0" * 32}
    rc, cli_missing = doors.cli("get_media", "--json", json.dumps(missing))
    is_error, mcp_missing = doors.via_mcp("get_media", missing)
    assert rc == 1 and is_error
    assert cli_missing["error"]["message"] == mcp_missing["text"]


def test_mesh_send_and_inbox_between_two_agents(doors):
    rc, alpha = doors.cli("mesh_summary", "--summary", "alpha at work", "--session", f"a-{TAG}",
                          SLM_AGENT_ID="alpha")
    assert rc == 0 and alpha["registered"] is True, alpha
    rc, sent = doors.cli("mesh_send", "--to", alpha["peer_id"], "--message", f"hello {TAG}",
                         "--session", f"b-{TAG}", SLM_AGENT_ID="beta")
    assert rc == 0 and sent["ok"] is True, sent
    rc, inbox = doors.cli("mesh_inbox", "--session", f"a-{TAG}", SLM_AGENT_ID="alpha")
    assert rc == 0, inbox
    mine = [m for m in inbox["messages"] if m["content"] == f"hello {TAG}"]
    assert len(mine) == 1 and mine[0]["to_peer"] == alpha["peer_id"]
    assert mine[0]["from_peer"] != alpha["peer_id"]
    is_error, mcp_inbox = doors.via_mcp("mesh_inbox", {})
    assert not is_error and _same_shape(inbox, mcp_inbox)
    # The long poll returns within its bound, read messages are not delivered twice.
    rc, waited = doors.cli("mesh_wait", "--timeout-s", "1", "--session", f"a-{TAG}",
                           SLM_AGENT_ID="alpha")
    assert rc == 0 and waited["timed_out"] is True and waited["messages"] == []


def test_code_graph_on_a_tiny_repo(doors):
    repo = doors.home / "tinyrepo"
    repo.mkdir(exist_ok=True)
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef twice(x):\n    return add(x, x)\n",
        encoding="utf-8")
    rc, built = doors.cli("build_code_graph", "--repo-path", str(repo))
    assert rc == 0 and built["success"] is True and built["nodes"] >= 2, built
    rc, radius = doors.cli("get_blast_radius", "--changed-files", "calc.py")
    is_error, mcp_radius = doors.via_mcp("get_blast_radius", {"changed_files": "calc.py"})
    assert rc == 0 and not is_error and radius["changed_nodes"]
    assert _same_shape(radius, mcp_radius) and sorted(radius["changed_nodes"]) == sorted(
        mcp_radius["changed_nodes"])


def test_a_learning_signal_matches_mcp(doors):
    rc, saved = doors.cli("remember", "--content", f"SlmTool{TAG} feedback target",
                          "--idempotency-key", f"fb-{TAG}")
    fact = saved["fact_ids"][0]
    rc, cli_fb = doors.cli("report_feedback", "--fact-id", fact, "--feedback", "relevant",
                           "--query", "feedback target")
    is_error, mcp_fb = doors.via_mcp("report_feedback", {
        "fact_id": fact, "feedback": "relevant", "query": "feedback target"})
    assert (rc == 1) == is_error and _same_shape(cli_fb, mcp_fb), (cli_fb, mcp_fb)


# -- boundaries --------------------------------------------------------------------------------

def test_a_tool_for_other_computers_stays_refused(doors):
    rc, answer = doors.cli("media_upload_link", "--kind", "image")
    is_error, mcp_answer = doors.via_mcp("media_upload_link", {"kind": "image"})
    assert rc == 1 and answer["code"] == "not_for_local"
    assert is_error and answer == _as_cli_failure(mcp_answer)


def test_the_daemon_is_told_cli_even_when_the_environment_says_mcp(doors, tmp_path):
    log = tmp_path / "cli-headers.jsonl"
    rc, saved = doors.cli("remember", "--content", f"SlmTool{TAG} channel probe",
                          log=log, SLM_CLIENT_CHANNEL="mcp", SLM_AGENT_ID="alpha")
    assert rc == 0, saved
    sent = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    sent = [r for r in sent if "url" in r]
    with_channel = [r for r in sent if "x-slm-channel" in r["headers"]]
    assert with_channel, sent
    assert {r["headers"]["x-slm-channel"] for r in with_channel} == {"cli"}
    # Control: the MCP child on the same daemon says mcp, so the recorder sees the header.
    doors.via_mcp("get_status", {})
    mcp_sent = [r for r in (json.loads(line) for line in
                (doors.root / "mcp-headers.jsonl").read_text(encoding="utf-8").splitlines())
                if "url" in r]
    assert {r["headers"].get("x-slm-channel") for r in mcp_sent} - {None} == {"mcp"}


def test_a_channel_in_the_arguments_is_refused_before_any_request(doors, tmp_path):
    log = tmp_path / "spoof.jsonl"
    rc, body = doors.cli("remember", "--json", json.dumps(
        {"content": "spoof", "channel": "dashboard"}), log=log)
    assert rc == 2 and body["code"] == "invalid_arguments"
    assert any("unknown argument 'channel'" in d for d in body["details"])
    requests = [r for r in map(json.loads, log.read_text(encoding="utf-8").splitlines())
                if "url" in r]
    assert requests == []  # nothing reached the daemon


def test_an_agent_named_in_the_arguments_is_only_a_claim(doors):
    rc, saved = doors.cli("remember", "--content", f"SlmTool{TAG} who said this",
                          "--agent-id", "someone-else", SLM_AGENT_ID="alpha")
    assert rc == 0, saved
    conn = sqlite3.connect(f"file:{doors.daemon.data_root / 'memory.db'}?mode=ro", uri=True,
                           timeout=30)
    try:
        rows = conn.execute("SELECT metadata_json FROM memories WHERE content LIKE ?",
                            (f"%SlmTool{TAG} who said this%",)).fetchall()
    finally:
        conn.close()
    assert rows, "the memory was not saved"
    meta = json.loads(rows[0][0] or "{}")
    assert meta.get("agent_id") == "alpha" and meta.get("agent_id_source") == "connection"
    assert meta.get("claimed_agent_id") == "someone-else"


def test_set_mode_never_loads_the_stdio_server_into_the_command_line(doors, tmp_path):
    """``set_mode`` resets its host's engine; loading ``mcp/server.py`` would start the stdio
    server's threads in this process and turn its channel into ``mcp``."""
    log = tmp_path / "mode.jsonl"
    rc, answer = doors.cli("set_mode", "--mode", "a", log=log)
    assert rc == 0 and answer.get("success") is True, answer
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    end = [r for r in records if r.get("exit")]
    assert end == [{"exit": True, "stdio_server_imported": False, "channel": "cli"}], end
    assert {r["headers"].get("x-slm-channel") for r in records if "url" in r} <= {"cli", None}
