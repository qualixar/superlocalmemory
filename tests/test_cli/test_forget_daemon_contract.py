"""``slm forget`` must never construct a local canonical writer."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch


def test_forget_queries_and_deletes_only_through_owned_daemon(capsys) -> None:
    """The query-and-delete batch stays on the daemon side of the boundary."""
    from superlocalmemory.cli.commands import cmd_forget

    args = Namespace(query="needle", dry_run=False, json=True, yes=True)
    responses = iter((
        {
            "memories": [
                {"id": "fact-1", "content": "needle one"},
                {"id": "fact-2", "content": "unrelated"},
            ],
            "has_more": False,
        },
        {"success": True, "deleted": "fact-1"},
    ))
    with (
        patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
        patch(
            "superlocalmemory.cli.daemon.daemon_request",
            side_effect=lambda *_args, **_kwargs: next(responses),
        ) as request,
        patch(
            "superlocalmemory.core.engine.MemoryEngine",
            side_effect=AssertionError("forget must not create a local writer"),
        ),
    ):
        cmd_forget(args)

    assert request.call_args_list == [
        (("GET", "/api/memories?limit=200&offset=0"), {}),
        (("DELETE", "/api/memories/fact-1"), {}),
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"]["deleted"] == ["fact-1"]
