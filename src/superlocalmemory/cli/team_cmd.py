# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm team`` — the workspace's login policy, from the terminal.

    slm team status
    slm team policy --require-login on|off
    slm team admin <username> [--profile NAME]
    slm team reset-password <username>

When a workspace requires login, the dashboard's install token cannot change
the policy: it is handed to any program on this computer. This command sends the
daemon capability instead, a private file that only this user can read, so the
owner can always recover, even with every administrator locked out: make an
account an administrator again, or set a new password for one (4.1.25 audit
UX-20 / UX-21: the dashboard pointed at a reset control that did not exist).
"""

from __future__ import annotations

import sys
from argparse import SUPPRESS, Namespace
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
    leaf_json = {"action": "store_true", "dest": "json", "default": SUPPRESS,
                 "help": "machine-readable output (same as `slm team --json ...`)"}
    st = tsub.add_parser("status", help="is login required, and how many users exist")
    st.add_argument("--json", **leaf_json)
    pol = tsub.add_parser("policy", help="require login for this workspace, or stop requiring it")
    pol.add_argument("--json", **leaf_json)
    pol.add_argument("--require-login", choices=["on", "off"], required=True,
                     help="on: every user signs in; off: the machine owner is the user")
    adm = tsub.add_parser("admin", help="make an account an administrator of a workspace")
    adm.add_argument("username")
    adm.add_argument("--profile", default=None, help="workspace (default: the active one)")
    adm.add_argument("--json", **leaf_json)
    rst = tsub.add_parser("reset-password", help="set a new password for an account")
    rst.add_argument("username")
    rst.add_argument("--password-stdin", action="store_true",
                     help="read the new password from standard input (for scripts)")
    rst.add_argument("--json", **{**leaf_json, "help": "machine-readable output; needs "
                                                       "--password-stdin (JSON never prompts)"})


def _fail(args: Namespace, command: str, message: str, code: str = "failed",
          **fields: Any) -> NoReturn:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(command, error={"code": code, "message": message, **fields})
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


def _user_id(args: Namespace, command: str, username: str) -> str:
    users = _request(args, command, "GET", "/users").get("users", [])
    for user in users:
        if user.get("username") == username:
            return str(user["user_id"])
    _fail(args, command, f"No account named '{username}'.")


def _new_password(args: Namespace, command: str) -> str:
    if getattr(args, "password_stdin", False):
        password = sys.stdin.readline().rstrip("\n")
    elif getattr(args, "json", False):
        _fail(args, command, "JSON never prompts: pass the new password on standard input "
              "with --password-stdin.", code="input_required",
              missing_flags=["--password-stdin"])
    else:
        import getpass

        password = getpass.getpass("New password (min 8 characters): ")
        if password != getpass.getpass("Repeat it: "):
            _fail(args, command, "The two passwords differ. Nothing was changed.")
    if len(password) < 8:
        _fail(args, command, "The password must be at least 8 characters. Nothing was changed.")
    return password


def _recover(args: Namespace, sub: str, command: str) -> None:
    uid = _user_id(args, command, args.username)
    if sub == "admin":
        body = {"user_id": uid, "role": "admin"}
        if args.profile:
            body["profile_id"] = args.profile
        done = _request(args, command, "POST", "/members", body)
        membership = done.get("membership") or {}
        where = membership.get("profile_id", args.profile or "the active workspace")
        _emit(args, command, done, f"{args.username} is now an administrator of {where}.")
        return
    password = _new_password(args, command)
    done = _request(args, command, "PATCH", f"/users/{uid}", {"password": password})
    _emit(args, command, {"ok": bool(done.get("ok"))},
          f"New password set for {args.username}. Their old sessions are signed out.")


def cmd_team(args: Namespace) -> None:
    sub = getattr(args, "team_command", None) or "status"
    command = f"team {sub}"
    if sub in ("admin", "reset-password"):
        _recover(args, sub, command)
        return
    if sub == "policy":
        wanted = args.require_login == "on"
        done = _request(args, command, "POST", "/policy", {"require_login": wanted})
        _emit(args, command, done, f"Require login is now {'on' if wanted else 'off'} for this workspace.")
        return
    status = _request(args, command, "GET", "/status")
    state = "on" if status.get("require_login") else "off"
    _emit(args, command, status, f"Require login: {state}. Users: {status.get('user_count', 0)}.")
