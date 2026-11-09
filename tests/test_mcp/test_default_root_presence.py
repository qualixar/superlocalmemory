"""Hosts started from an unpinned install are found, and one bad process hides nothing."""

import os
import time
from types import SimpleNamespace

import pytest

from superlocalmemory.infra import mcp_legacy, mcp_processes


class Proc:
    def __init__(self, cmdline, env, pid=401):
        self.pid = pid
        self.info = {"pid": pid, "ppid": 300, "cmdline": cmdline}
        self._env = env

    def environ(self):
        return self._env

    def uids(self):
        return SimpleNamespace(real=os.getuid())

    def parent(self):
        return SimpleNamespace(is_running=lambda: True)

    def parents(self):
        return []

    def create_time(self):
        return time.time() - 300


@pytest.fixture(autouse=True)
def identity(monkeypatch):
    monkeypatch.setattr(mcp_processes, "process_identity", lambda pid: f"start-{pid}")


def _serve(monkeypatch, *procs):
    monkeypatch.setattr(mcp_legacy, "_processes", lambda: list(procs))


def test_script_launched_server_with_default_root_is_listed(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".superlocalmemory").mkdir(parents=True)
    _serve(monkeypatch, Proc(
        ["/opt/py/bin/python3.14", "/opt/py/bin/slm", "mcp"],
        {"HOME": str(home), "CLAUDECODE": "1"},
    ))
    rows = mcp_legacy.discover(home / ".superlocalmemory", set())
    assert [r["host"] for r in rows] == ["claude-code"]


def test_default_root_of_another_home_is_not_listed(monkeypatch, tmp_path):
    _serve(monkeypatch, Proc(
        ["/opt/py/bin/python3.14", "/opt/py/bin/slm", "mcp"],
        {"HOME": str(tmp_path / "other"), "CLAUDECODE": "1"},
    ))
    assert mcp_legacy.discover(tmp_path / "home" / ".superlocalmemory", set()) == []


def test_explicit_other_root_still_wins_over_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _serve(monkeypatch, Proc(
        ["/opt/py/bin/python3.14", "/opt/py/bin/slm", "mcp"],
        {"HOME": str(home), "SLM_DATA_DIR": str(tmp_path / "elsewhere")},
    ))
    assert mcp_legacy.discover(home / ".superlocalmemory", set()) == []


def test_one_unexpected_process_error_does_not_hide_the_rest(monkeypatch, tmp_path):
    class Broken(Proc):
        def environ(self):
            raise RuntimeError("platform quirk")

    good = Proc(["/x/python", "-m", "superlocalmemory.cli.main", "mcp"],
                {"SLM_DATA_DIR": str(tmp_path), "SLM_MCP_HOST": "codex"}, pid=402)
    _serve(monkeypatch, Broken(["/x/python", "-m", "superlocalmemory.cli.main", "mcp"], {}), good)
    assert [r["host"] for r in mcp_legacy.discover(tmp_path, set())] == ["codex"]


def test_snapshot_keeps_registered_apps_when_discovery_fails(monkeypatch, tmp_path):
    mcp_processes.register(tmp_path, "s1", pid=os.getpid(), host="codex", version="1", tools=["a"])
    monkeypatch.setattr(mcp_legacy, "discover", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    view = mcp_processes.snapshot(tmp_path, installed_version="1")
    assert view["count"] == 1 and view["apps"][0]["host"] == "codex"
