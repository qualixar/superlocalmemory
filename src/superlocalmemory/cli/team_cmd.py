# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm team`` — the workspace's login policy, from the terminal.

    slm team status
    slm team policy --require-login on|off

When a workspace requires login, the dashboard's install token cannot change
the policy: it is handed to any program on this computer. This command sends the
daemon capability instead, a private file that only this user can read, so the
owner can always switch the requirement off, even with every administrator
locked out.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from typing import Any, NoReturn

from superlocalmemory.cli.daemon import DaemonRefused, daemon_request

_BASE = "/api/rbac"
_NOT_RUNNING = "The SLM daemon is not running. Start it with: slm serve"
_REFUSED = ("The daemon refused this. Run it as the same user that runs SLM, "
            "with the daemon running from this user's data folder.")


def register_team_parser(sub: Any) -> None:
    """Attach the ``team`` parser. Called from cli/main.py."""
    p = sub.add_parser("team", help="Workspace login policy: status, require-login on/off")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    tsub = p.add_subparsers(dest="team_command", title="team subcommands")
    tsub.add_parser("status", help="is login required, and how many users exist")
    pol = tsub.add_parser("policy", help="require login for this workspace, or stop requiring it")
    pol.add_argument("--require-login", choices=["on", "off"], required=True,
                     help="on: every user signs in; off: the machine owner is the user")


def _fail(args: Namespace, command: str, message: str) -> NoReturn:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(command, error={"message": message})
    else:
        print(message)
    sys.exit(1)


def _request(args: Namespace, command: str, method: str, path: str, body: dict | None = None) -> dict:
    try:
        result = daemon_request(method, _BASE + path, body)
    except DaemonRefused:
        _fail(args, command, _REFUSED)
    if result is None:
        _fail(args, command, _NOT_RUNNING)
    return result


def _emit(args: Namespace, command: str, data: dict, text: str) -> None:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(command, data=data)
    else:
        print(text)


def cmd_team(args: Namespace) -> None:
    sub = getattr(args, "team_command", None) or "status"
    command = f"team {sub}"
    if sub == "policy":
        wanted = args.require_login == "on"
        done = _request(args, command, "POST", "/policy", {"require_login": wanted})
        _emit(args, command, done, f"Require login is now {'on' if wanted else 'off'} for this workspace.")
        return
    status = _request(args, command, "GET", "/status")
    state = "on" if status.get("require_login") else "off"
    _emit(args, command, status, f"Require login: {state}. Users: {status.get('user_count', 0)}.")
