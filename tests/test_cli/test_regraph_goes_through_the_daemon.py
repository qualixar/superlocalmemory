"""``slm db regraph`` checks the graph copy against the store and rebuilds a drifted one (#296).

It used to look for the graph through a singleton only the daemon sets, so from the CLI it always
answered "no graph copy is configured", and re-queueing facts could not reach an edge whose fact is
gone anyway. With the daemon up it now asks the daemon (RocksDB lets one process open the copy);
with SLM stopped it does the work itself.
"""

from __future__ import annotations

import inspect
from argparse import Namespace

import pytest

from superlocalmemory.cli import commands as cli_commands

_REPORT = {"profile_id": "default", "stored": 2, "copied": 5, "pending": 0, "in_step": False,
           "rebuilt": True, "after": 2, "skipped": "", "error": ""}


def _alive(monkeypatch, *, answering: bool = True) -> list[tuple]:
    calls: list[tuple] = []
    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: answering)
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.daemon_request",
        lambda method, path, body=None, **kw: calls.append((method, path, body)) or {
            "reports": [_REPORT]},
    )
    return calls


def test_with_the_daemon_up_it_asks_the_daemon(monkeypatch, capsys):
    calls = _alive(monkeypatch)
    cli_commands._cmd_db_regraph(Namespace(check=False, profile=""))
    assert calls == [("POST", "/maintenance/regraph", {"check": False, "profile": ""})]
    out = capsys.readouterr().out
    assert "rebuilt" in out and "5 in the graph copy" in out


def test_check_is_passed_through(monkeypatch):
    calls = _alive(monkeypatch)
    cli_commands._cmd_db_regraph(Namespace(check=True, profile="work"))
    assert calls[0][2] == {"check": True, "profile": "work"}


def test_a_daemon_that_does_not_answer_is_an_error(monkeypatch, capsys):
    _alive(monkeypatch, answering=False)
    with pytest.raises(SystemExit) as stop:
        cli_commands._cmd_db_regraph(Namespace(check=False, profile=""))
    assert stop.value.code == 1


def test_a_failed_rebuild_exits_non_zero(monkeypatch):
    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.is_daemon_running", lambda: True)
    monkeypatch.setattr("superlocalmemory.cli.daemon.daemon_request",
                        lambda *a, **k: {"reports": [{**_REPORT, "rebuilt": False,
                                                      "error": "disk full"}]})
    with pytest.raises(SystemExit) as stop:
        cli_commands._cmd_db_regraph(Namespace(check=False, profile=""))
    assert stop.value.code == 1


def test_offline_on_a_store_without_a_graph_copy_says_so(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("superlocalmemory.cli.daemon.owned_daemon_process_alive", lambda: False)
    cli_commands._cmd_db_regraph(Namespace(check=True, profile=""))
    assert "no graph copy" in capsys.readouterr().out


def test_the_daemon_route_exists_and_is_gated():
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server import unified_daemon
    from superlocalmemory.server.mutation_gate import rule_for

    source = inspect.getsource(unified_daemon)
    start = source.index('@application.post("/maintenance/regraph")')
    body = source[start:start + 1800]
    assert "require_global_manage(request)" in body
    assert "authorize_route_mutation(" in body
    assert "from superlocalmemory.core.graph_projection_check import regraph" in body
    rule = rule_for("POST", "/maintenance/regraph")
    assert rule.scope == "global" and rule.permission == Permission.MANAGE
