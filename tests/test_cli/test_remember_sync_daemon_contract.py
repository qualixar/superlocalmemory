"""Synchronous CLI writes must use the owned daemon with a bounded request."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch

import pytest


def _args() -> Namespace:
    return Namespace(
        content="Alice owns Atlas.", tags="witness", json=True,
        sync_mode=True, scope=None, shared_with=None,
    )


def test_sync_remember_routes_to_daemon_wait_endpoint(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_remember

    response = {
        "fact_ids": ["fact-1"], "count": 1,
        "operation_id": "op-1", "materialization_state": "complete",
    }
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request", return_value=response,
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
    ):
        cmd_remember(_args())

    request.assert_called_once()
    assert request.call_args.args[1] == "/remember?wait=true"
    assert request.call_args.kwargs["timeout_seconds"] == 30
    assert json.loads(capsys.readouterr().out)["data"]["operation_id"] == "op-1"


def test_sync_remember_timeout_never_falls_back_to_unbounded_local_engine(
    capsys,
) -> None:
    from superlocalmemory.cli.commands import cmd_remember

    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch("superlocalmemory.cli.daemon.ensure_daemon", return_value=True),
        patch("superlocalmemory.cli.daemon.daemon_request", return_value=None),
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
        pytest.raises(SystemExit) as stopped,
    ):
        cmd_remember(_args())

    assert stopped.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "SYNC_TIMEOUT"

