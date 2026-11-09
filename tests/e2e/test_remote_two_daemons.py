# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Two real daemons on one machine: the SLM host (S) and the Hermes host's own SLM (H).

Opt-in (slow, starts two daemons): ``SLM_E2E_REMOTE=1 pytest tests/e2e -m e2e_remote``.

S runs remote access on a TLS listener with a certificate made by ``slm remote tls
init`` in this test. The real MCP Python client talks to S over HTTPS with a remote
key. Because the remote listener treats every caller as remote - even 127.0.0.1 -
loopback exercises the full remote path. Nothing here touches the user's daemon,
data or configuration: every root, port and key is created under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from tests._portable import child_env_base

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PRODUCTION_PORTS = {8765, 8767}

pytestmark = [
    pytest.mark.e2e_remote,
    pytest.mark.skipif(os.environ.get("SLM_E2E_REMOTE") != "1",
                       reason="opt-in: set SLM_E2E_REMOTE=1"),
]


def _free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in PRODUCTION_PORTS:
            return port


def _env(root: Path, port: int) -> dict:
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    env.update({
        **child_env_base(root / "home"), "PYTHONPATH": str(SRC),
        "SLM_DATA_DIR": str(root / "data"),
        "SLM_DAEMON_PORT": str(port), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(root / "cache" / "hf"), "XDG_CACHE_HOME": str(root / "cache"),
        "SENTENCE_TRANSFORMERS_HOME": str(root / "cache" / "st"), "CI": "1",
        # Never bind or redirect from the shared legacy port on this machine.
        "SLM_DISABLE_LEGACY_PORT": "1",
        "SLM_NON_INTERACTIVE": "1", "SLM_DISABLE_HF_DOWNLOAD": "1", "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "NO_PROXY": "127.0.0.1,localhost",
        # Each one-shot client call here is initialize + notification + call.
        "SLM_RATE_LIMIT_WRITE": "1000",
    })
    (root / "data").mkdir(parents=True, exist_ok=True)
    return env


def _slm(env: dict, *argv: str) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *argv],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (argv, result.stdout[-2000:], result.stderr[-2000:])
    return result


class Daemon:
    def __init__(self, root: Path, port: int) -> None:
        self.root, self.port, self.env = root, port, _env(root, port)
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        log = (self.root / "daemon.log").open("ab")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "superlocalmemory.server.unified_daemon", "--start",
             f"--port={self.port}"], stdout=log, stderr=log, env=self.env, cwd=str(REPO),
            start_new_session=True)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, self.log_tail()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health",
                                            timeout=3) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.5)
        raise AssertionError("daemon did not start\n" + self.log_tail())

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=20)

    def log_tail(self) -> str:
        return (self.root / "daemon.log").read_text(errors="replace", encoding="utf-8")[-4000:]

    def fact_count(self) -> int:
        import sqlite3

        db = self.root / "data" / "memory.db"
        if not db.exists():
            return 0
        conn = sqlite3.connect(db, timeout=30)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0])
        finally:
            conn.close()


# -- the real MCP client ---------------------------------------------------------------


async def _mcp(url: str, headers: dict, verify, tool: str | None, args: dict | None):
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx2.AsyncClient(headers=headers, verify=verify, follow_redirects=True,
                                  timeout=httpx2.Timeout(60.0)) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                if tool is None:
                    return await session.list_tools()
                return await session.call_tool(tool, args or {})


def mcp_call(url, key, verify, tool, args=None, timeout=90):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return asyncio.run(asyncio.wait_for(_mcp(url, headers, verify, tool, args), timeout))


def _text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


class HermesLikeCtx:
    """``ctx.call_mcp`` with Hermes's envelope semantics over the real client."""

    def __init__(self, url, key, verify, release):
        self.url, self.key, self.verify, self.release = url, key, verify, release

    def get_config(self, key, default=None):
        return "remote" if key == "connection" else default

    def call_mcp(self, server, tool, arguments=None, timeout=30):
        assert server == "superlocalmemory"
        try:
            result = mcp_call(self.url, self.key, self.verify, tool, arguments, timeout=timeout)
        except BaseException as exc:  # noqa: BLE001 — Hermes turns failures into envelopes
            while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
                exc = exc.exceptions[0]
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        if result.is_error:
            return {"ok": False, "error": _text(result)}
        return {"ok": True, "result": _text(result)}


# -- the scenario ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("remote-e2e")
    s = Daemon(root / "s", _free_port())
    h = Daemon(root / "h", _free_port())
    remote_port = _free_port()
    _slm(s.env, "remote", "tls", "init", "--name", "localhost", "--ip", "127.0.0.1")
    _slm(s.env, "remote", "enable", "--listen", f"127.0.0.1:{remote_port}")
    added = _slm(s.env, "remote", "keys", "add", "hermes")
    key = next(t for t in added.stdout.split() if t.startswith("slmr_"))
    read_key = next(t for t in _slm(s.env, "remote", "keys", "add", "viewer",
                                    "--read-only").stdout.split() if t.startswith("slmr_"))
    try:
        s.start()
        h.start()
        # A second workspace on S that no remote key is bound to.
        _slm(s.env, "profile", "create", "clientx")
        ca = str(root / "s" / "data" / "remote" / "tls" / "ca.pem")
        yield {"s": s, "h": h, "key": key, "read_key": read_key, "ca": ca,
               "port": remote_port, "url": f"https://localhost:{remote_port}/mcp/hermes"}
    finally:
        s.stop()
        h.stop()


def test_remember_then_recall_over_tls_and_the_other_store_is_untouched(pair) -> None:
    ctx = ssl.create_default_context(cafile=pair["ca"])
    h_before = pair["h"].fact_count()
    token = f"remote-e2e-{os.getpid()}-ferry"
    saved = mcp_call(pair["url"], pair["key"], ctx, "remember",
                     {"content": f"{token} leaves pier nine at dawn", "agent_id": "hermes"})
    assert not saved.is_error, _text(saved)
    assert json.loads(_text(saved))["success"] is True
    found = mcp_call(pair["url"], pair["key"], ctx, "recall", {"query": f"{token} pier"})
    assert token in _text(found)
    assert pair["h"].fact_count() == h_before


def test_tools_list_over_the_wire_hides_host_only_tools(pair) -> None:
    ctx = ssl.create_default_context(cafile=pair["ca"])
    listed = {t.name for t in mcp_call(pair["url"], pair["key"], ctx, None).tools}
    assert "remember" in listed and "switch_profile" not in listed and "forget" not in listed
    read_listed = {t.name for t in mcp_call(pair["url"], pair["read_key"], ctx, None).tools}
    assert "recall" in read_listed and "remember" not in read_listed


def test_without_the_ca_tls_fails_and_plain_http_fails(pair) -> None:
    with pytest.raises(BaseException):
        mcp_call(pair["url"], pair["key"], True, "get_status", timeout=30)
    with pytest.raises(BaseException):
        mcp_call(pair["url"].replace("https://", "http://"), pair["key"], True, "get_status",
                 timeout=30)


def test_wrong_and_revoked_keys_are_refused(pair) -> None:
    ctx = ssl.create_default_context(cafile=pair["ca"])
    with pytest.raises(BaseException):
        mcp_call(pair["url"], "slmr_" + "A" * 43, ctx, "get_status", timeout=30)
    added = _slm(pair["s"].env, "remote", "keys", "add", "temp")
    temp = next(t for t in added.stdout.split() if t.startswith("slmr_"))
    assert not mcp_call(pair["url"], temp, ctx, "get_status").is_error
    _slm(pair["s"].env, "remote", "keys", "revoke", "temp")
    with pytest.raises(BaseException):
        mcp_call(pair["url"], temp, ctx, "get_status", timeout=30)


def test_hook_token_never_leaves_loopback(pair, monkeypatch) -> None:
    # (a) A non-loopback hook URL is ignored: the hook talks to H's own daemon.
    probe = (
        "import json, superlocalmemory.hooks.post_tool_async_hook as h; "
        "print(json.dumps(h.DAEMON_URL))"
    )
    env = {**pair["h"].env, "SLM_HOOK_DAEMON_URL": f"https://slm-remote.test:{pair['port']}"}
    out = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True,
                         text=True, timeout=60, check=True)
    assert json.loads(out.stdout.strip()) == f"http://127.0.0.1:{pair['h'].port}"
    # (b) A loopback name pointed at S's remote listener: plain HTTP is refused by TLS,
    #     and over HTTPS the hook endpoint does not exist there.
    token = (pair["h"].root / "data" / ".install_token").read_text(encoding="utf-8").strip()
    request = urllib.request.Request(f"http://localhost:{pair['port']}/internal/prewarm",
                                     data=b"{}", method="POST",
                                     headers={"X-SLM-Hook-Token": token})
    with pytest.raises(Exception):
        urllib.request.urlopen(request, timeout=10)
    secure = urllib.request.Request(f"https://localhost:{pair['port']}/internal/prewarm",
                                    data=b"{}", method="POST",
                                    headers={"X-SLM-Hook-Token": token})
    ctx = ssl.create_default_context(cafile=pair["ca"])
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(secure, timeout=10, context=ctx)
    assert err.value.code == 404


def test_tool_policy_end_to_end_switch_profile_is_refused(pair) -> None:
    ctx = ssl.create_default_context(cafile=pair["ca"])
    result = mcp_call(pair["url"], pair["key"], ctx, "switch_profile", {"profile": "evil"})
    assert result.is_error and "not available over remote access" in _text(result)
    status = json.loads(_text(mcp_call(pair["url"], pair["key"], ctx, "get_status")))
    assert status["profile"] == "default"


def test_plugin_reports_not_saved_when_the_server_is_down_then_saved(pair) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("slm_hermes_e2e",
                                                  REPO / "hermes-plugin" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = HermesLikeCtx(pair["url"], pair["key"],
                        ssl.create_default_context(cafile=pair["ca"]), module._RELEASE_SLM_VERSION)
    plugin = module.SlmHermesPlugin(ctx)
    assert plugin.slash_router("remember the e2e plugin path works").startswith("Saved")
    pair["s"].stop()
    try:
        plugin._remote_health._version = None  # force a fresh status probe
        out = plugin.slash_router("remember this must not be reported as saved")
        assert out.startswith("NOT SAVED"), out
        plugin.post_tool_call(session_id="s", tool_name="t", args={}, result="r")
        plugin.post_tool_call(session_id="s", tool_name="t", args={}, result="r")
        line = plugin._remote_health.status_line()
        assert "lifecycle captures skipped while unreachable: 2" in line, line
    finally:
        pair["s"].start()
    plugin._remote_health._open_until = 0.0
    assert plugin.slash_router("remember back again after restart").startswith("Saved")


_READ_TOOL_ARGS = {
    "fetch": {"fact_ids": "nonexistent"}, "recall": {"query": "pier"},
    "search": {"query": "pier"}, "recall_trace": {"query": "pier"},
    "prestage_context": {"query": "pier"}, "skill_lineage": {"skill_name": "x"},
    "slm_retrieve": {"ccr_id": "00000000-0000-4000-8000-000000000000"},
    "slm_cache_get": {"key": "x"}, "slm_loop_show": {"run_id": "x"},
}


def test_no_read_tool_shows_a_remote_caller_the_hosts_paths(pair) -> None:
    import getpass

    from superlocalmemory.server.remote_tool_policy import READ_TOOLS

    ctx = ssl.create_default_context(cafile=pair["ca"])
    # A memory that mentions a path: memory text comes back exactly as written.
    note = f"runbook-{os.getpid()} lives at /srv/runbooks/deploy.md"
    mcp_call(pair["url"], pair["key"], ctx, "remember", {"content": note})
    host = [str(pair["s"].root), "/Users/", "/home/", getpass.getuser()]
    leaks: dict[str, str] = {}
    listed = {t.name for t in mcp_call(pair["url"], pair["read_key"], ctx, None).tools}
    for tool in sorted(READ_TOOLS & listed):
        result = mcp_call(pair["url"], pair["read_key"], ctx, tool, _READ_TOOL_ARGS.get(tool, {}))
        blob = _text(result) + json.dumps(result.structured_content or {})
        for value in host:
            if value and value in blob:
                leaks[tool] = value
    assert leaks == {}, leaks
    recalled = _text(mcp_call(pair["url"], pair["read_key"], ctx, "recall",
                              {"query": f"runbook-{os.getpid()}"}))
    assert "/srv/runbooks/deploy.md" in recalled
    # The same tool on this computer keeps the full detail.
    local = mcp_call(f"http://127.0.0.1:{pair['s'].port}/mcp/hermes", None, True, "get_status")
    assert str(pair["s"].root) in _text(local)


# -- exploit regressions: a remote key stays inside its own cache and its own profile ---


def _local_url(pair, agent: str = "claude") -> str:
    return f"http://127.0.0.1:{pair['s'].port}/mcp/{agent}"


def _remote_url(pair, agent: str = "claude") -> str:
    return f"https://localhost:{pair['port']}/mcp/{agent}"


def _answer(result) -> dict:
    return json.loads(_text(result))


def _secret(stdout: str) -> str:
    return next(t for t in stdout.split() if t.startswith("slmr_"))


def test_remote_keys_cannot_read_or_overwrite_a_local_agents_cache(pair) -> None:
    """Audit L2 F1: /mcp/claude from another computer used to be the local claude's cache."""
    ctx = ssl.create_default_context(cafile=pair["ca"])
    entry = "read:/Users/me/project/.env"
    secret = f"API_TOKEN=local-only-{os.getpid()}"
    stored = mcp_call(_local_url(pair), None, True, "slm_cache_set",
                      {"key": entry, "value": secret})
    assert _answer(stored)["stored"] is True
    for key in (pair["read_key"], pair["key"]):
        got = mcp_call(_remote_url(pair), key, ctx, "slm_cache_get", {"key": entry})
        assert secret not in _text(got)
        assert _answer(got)["hit"] is False
    planted = mcp_call(_remote_url(pair), pair["key"], ctx, "slm_cache_set",
                       {"key": entry, "value": "INJECTED"})
    assert _answer(planted)["stored"] is True  # into the key's own cache
    local = _answer(mcp_call(_local_url(pair), None, True, "slm_cache_get", {"key": entry}))
    assert local["hit"] is True and local["value"] == secret
    # The remote key keeps a working cache of its own that no other key can read.
    own = _answer(mcp_call(_remote_url(pair), pair["key"], ctx, "slm_cache_get", {"key": entry}))
    assert own["hit"] is True and own["value"] == "INJECTED"
    other = _answer(mcp_call(_remote_url(pair), pair["read_key"], ctx, "slm_cache_get",
                             {"key": entry}))
    assert other["hit"] is False


def test_a_remote_key_reaches_only_the_profile_it_is_bound_to(pair) -> None:
    """Audit L2 F2: a read key used to recall another profile by naming it."""
    ctx = ssl.create_default_context(cafile=pair["ca"])
    token = f"clientx-secret-{os.getpid()}"
    saved = mcp_call(_local_url(pair), None, True, "remember",
                     {"content": f"{token} the merger closes on friday", "profile_id": "clientx"})
    assert _answer(saved)["success"] is True
    remote = _remote_url(pair, "hermes")
    for key in (pair["read_key"], pair["key"]):
        for tool, args in (("recall", {"query": f"{token} merger", "profile_id": "clientx"}),
                           ("list_corrections", {"profile_id": "clientx"}),
                           ("prestage_context", {"query": token, "profile_id": "clientx"})):
            refused = mcp_call(remote, key, ctx, tool, args)
            assert refused.is_error, (tool, _text(refused))
            assert token not in _text(refused)
            assert "bound to profile 'default'" in _text(refused), _text(refused)
    for args in ({"content": "planted into clientx", "profile_id": "clientx"},
                 {"content": "planted everywhere", "scope": "global"},
                 {"content": "planted for clientx", "scope": "shared", "shared_with": "clientx"}):
        refused = mcp_call(remote, pair["key"], ctx, "remember", args)
        assert refused.is_error, (args, _text(refused))
    # Its own profile still works, named or not.
    for args in ({"query": "pier"}, {"query": "pier", "profile_id": "default"}):
        assert not mcp_call(remote, pair["read_key"], ctx, "recall", args).is_error


def _facts_in(daemon, profile: str, needle: str) -> int:
    import sqlite3

    conn = sqlite3.connect(daemon.root / "data" / "memory.db", timeout=30)
    try:
        return int(conn.execute(
            "SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ? AND content LIKE ?",
            (profile, f"%{needle}%")).fetchone()[0])
    finally:
        conn.close()


def _wait_for_fact(daemon, profile: str, needle: str, timeout: float = 60.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = _facts_in(daemon, profile, needle)
        if count:
            return count
        time.sleep(0.5)
    return 0


def test_a_key_bound_to_work_is_served_for_work_while_the_host_uses_personal(pair) -> None:
    """Remote recall and remember are routed to the key's profile; the host's is untouched."""
    ctx = ssl.create_default_context(cafile=pair["ca"])
    s, env = pair["s"], pair["s"].env
    for name in ("personal", "work"):
        _slm(env, "profile", "create", name)
    key = _secret(_slm(env, "remote", "keys", "add", "work-hermes", "--profile", "work").stdout)
    rows = {r["name"]: r for r in json.loads(_slm(env, "remote", "keys", "list",
                                                  "--json").stdout)["keys"]}
    assert rows["work-hermes"]["profile"] == "work"
    assert rows["hermes"]["profile"] == "default" and rows["viewer"]["profile"] == "default"
    _slm(env, "profile", "switch", "personal")
    try:
        url = _remote_url(pair, "hermes")
        token = f"work-only-{os.getpid()}"
        saved = mcp_call(url, key, ctx, "remember",
                         {"content": f"{token} the quarterly plan is due monday"})
        assert not saved.is_error and _answer(saved)["success"] is True, _text(saved)
        assert _wait_for_fact(s, "work", token) == 1
        assert _facts_in(s, "personal", token) == 0
        found = mcp_call(url, key, ctx, "recall", {"query": f"{token} quarterly plan"})
        assert not found.is_error and token in _text(found)
        # The host is still on 'personal', and its own recall does not see it.
        local = _answer(mcp_call(_local_url(pair), None, True, "get_status"))
        assert local["profile"] == "personal"
        mine = mcp_call(_local_url(pair), None, True, "recall", {"query": f"{token} plan"})
        assert token not in _text(mine)
        # A tool that can only work on the active profile says what to do.
        refused = mcp_call(url, key, ctx, "search", {"query": token})
        assert refused.is_error and "ask the host owner" in _text(refused)
        assert "personal" not in _text(refused)
    finally:
        _slm(env, "profile", "switch", "default")
    bad = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "remote", "keys",
                          "add", "ghost", "--profile", "no-such-profile"], env=env,
                         capture_output=True, text=True, timeout=120)
    assert bad.returncode != 0 and "slmr_" not in bad.stdout


@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT"])
def test_non_post_on_the_mcp_endpoint_from_a_remote_caller_is_405_at_once(pair, method) -> None:
    """Audit L2: GET /mcp/ with a valid key used to hold an empty event stream open."""
    ctx = ssl.create_default_context(cafile=pair["ca"])
    request = urllib.request.Request(_remote_url(pair, "hermes"), method=method, headers={
        "Authorization": f"Bearer {pair['key']}", "Accept": "text/event-stream"})
    started = time.monotonic()
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(request, timeout=10, context=ctx)
    assert err.value.code == 405
    assert time.monotonic() - started < 5
