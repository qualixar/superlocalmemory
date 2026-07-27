"""CLI mutation commands must fail closed outside the owned daemon."""

from __future__ import annotations

import json
import sqlite3
from argparse import Namespace

import pytest


@pytest.mark.parametrize(
    ("command_name", "args"),
    [
        (
            "cmd_remember",
            Namespace(
                content="must not be written locally",
                tags="",
                json=True,
                sync_mode=False,
                scope=None,
                shared_with=None,
            ),
        ),
        ("cmd_delete", Namespace(fact_id="fact-1", yes=True, json=True)),
        (
            "cmd_update",
            Namespace(fact_id="fact-1", content="replacement", json=True),
        ),
        (
            "cmd_recall",
            Namespace(
                query="offline search must not warm a writer",
                limit=10,
                json=True,
                fast=False,
                include_global=None,
                include_shared=None,
                window="",
            ),
        ),
    ],
)
def test_cli_mutation_fails_closed_when_daemon_unavailable(
    command_name,
    args,
    monkeypatch,
    capsys,
) -> None:
    """No CLI mutation may cold-start MemoryEngine or SQLite as a fallback."""
    from superlocalmemory.cli import commands
    from superlocalmemory.core.engine import MemoryEngine

    def forbidden(*args, **kwargs):
        raise AssertionError("CLI client must not open a local canonical writer")

    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.is_daemon_running", lambda: False,
    )
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.ensure_daemon", lambda: False,
    )
    monkeypatch.setattr(MemoryEngine, "__init__", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    with pytest.raises(SystemExit) as stopped:
        getattr(commands, command_name)(args)

    assert stopped.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["error"] == {
        "code": "DAEMON_UNAVAILABLE",
        "message": "Owned daemon is unavailable; retry later.",
        "retryable": True,
    }
