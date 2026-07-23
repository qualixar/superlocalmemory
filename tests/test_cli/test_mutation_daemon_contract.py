"""CLI mutations must use the resident daemon as the single SQLite writer."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch


def test_delete_routes_to_resident_daemon(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_delete

    args = Namespace(fact_id="fact-1", yes=True, json=True)
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={"success": True, "deleted": "fact-1"},
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
    ):
        cmd_delete(args)

    request.assert_called_once_with("DELETE", "/api/memories/fact-1")
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["deleted"] == "fact-1"


def test_update_routes_to_resident_daemon(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_update

    args = Namespace(fact_id="fact 1", content="Updated content.", json=True)
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            return_value={
                "success": True,
                "fact_id": "fact 1",
                "content": "Updated content.",
            },
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine.initialize",
            side_effect=AssertionError("must not cold-start a second engine"),
        ),
    ):
        cmd_update(args)

    request.assert_called_once_with(
        "PATCH",
        "/api/memories/fact%201",
        {"content": "Updated content."},
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["fact_id"] == "fact 1"
