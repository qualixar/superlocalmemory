# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm remember`` sends an idempotency key, so a retry never stores twice.

The daemon's "accepted" answer says to resend with the same idempotency_key
for the final receipt. The CLI sent none, so re-running the command (the only
resend a CLI user has) stored the memory a second time.
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

from tests.test_server.test_canonical_remember_route import _client


def _args(content: str, **extra) -> Namespace:
    values = dict(
        content=content, tags="", json=True, sync_mode=False, scope=None,
        shared_with=None, kind=None, replaces=None,
    )
    values.update(extra)
    return Namespace(**values)


def test_rerunning_the_same_remember_stores_it_once(engine_with_mock_deps, capsys) -> None:
    from superlocalmemory.cli.commands import cmd_remember

    bodies: list[dict] = []
    with _client(engine_with_mock_deps) as client:

        def forward(_method, path, body, **_kwargs):
            # Only the CLI's remember requests are under test. The same patched function also
            # carries the embedder proxy's POST /api/v3/embed when an earlier test in the worker
            # left the proxy embedder in place; that body has no idempotency key by design.
            if path.startswith("/remember"):
                bodies.append(dict(body))
            response = client.post(path, json=body)
            assert response.status_code in (200, 202), response.text
            return response.json()

        with (
            patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True),
            patch("superlocalmemory.cli.daemon.daemon_request", side_effect=forward),
        ):
            same = "The night ferry to Kos leaves at 23:40 from pier 4."
            cmd_remember(_args(same))
            cmd_remember(_args(same))
            cmd_remember(_args("The day ferry to Kos leaves at 09:10 from pier 2."))
            cmd_remember(_args(same, tags="travel"))
    capsys.readouterr()

    keys = [body.get("idempotency_key") for body in bodies]
    assert all(keys), "the CLI must send an idempotency key"
    assert keys[0] == keys[1], "a re-run of the same command is the same request"
    assert len({keys[0], keys[2], keys[3]}) == 3, "a different request gets its own key"
    operations = engine_with_mock_deps._db.execute("SELECT * FROM ingestion_operations")
    assert len(operations) == 3
