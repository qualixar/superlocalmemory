# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""``slm embedder`` — change the embedding model without stopping SLM.

    slm embedder switch MODEL [--dimension N] [--provider P] [--endpoint URL]
    slm embedder status
    slm embedder rollback
    slm embedder cancel
    slm embedder forget-previous
    slm embedder upgrade [--yes]

A switch re-indexes every memory in the background inside the running daemon:
recall and remember keep working on the current model until the new one is
ready, then both change in one step. The previous vectors are kept until the
next switch so ``rollback`` can return to them; ``forget-previous`` frees
them. Keys are never taken on the command line: a hosted model's key is set
in the dashboard or config.json. ``--json`` everywhere.
"""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from typing import Any, NoReturn

from superlocalmemory.cli.daemon import DaemonConflict, DaemonUnprocessable, daemon_request

_BASE = "/api/v3/embedding/reindex"
_NOT_RUNNING = "The SLM daemon is not running. Start it with: slm serve"
#: A rollback starts and tests the previous model inside the request (9.9-22 s
#: measured on a 22k-fact store under load); a short timeout reported "not
#: running" for a rollback the daemon went on to start.
_ROLLBACK_TIMEOUT_S = 300.0


def _out(args: Namespace, command: str, data: dict, text: str) -> None:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(f"embedder {command}", data=data)
    else:
        print(text)


def _fail(args: Namespace, command: str, message: str, code: str = "REFUSED") -> NoReturn:
    if getattr(args, "json", False):
        from superlocalmemory.cli.json_output import json_print

        json_print(f"embedder {command}", error={"code": code, "message": message})
    else:
        print(message, file=sys.stderr)
    sys.exit(1)


def _request(args: Namespace, command: str, method: str, path: str = "",
             body: dict | None = None, timeout: float = 60.0) -> dict:
    try:
        result = daemon_request(method, _BASE + path, body, preserve_conflict=True,
                                preserve_unprocessable=True, timeout_seconds=timeout)
    except DaemonConflict as exc:
        _fail(args, command, exc.detail, "CONFLICT")
    except DaemonUnprocessable as exc:
        _fail(args, command, exc.message, exc.code or "INVALID")
    if result is None:
        _fail(args, command, _NOT_RUNNING, "DAEMON_UNAVAILABLE")
    if result.get("error"):
        _fail(args, command, str(result.get("detail") or result["error"]), str(result["error"]))
    return result


def describe(job: dict | None) -> str:
    if not job:
        return "No embedding model switch has run on this store."
    line = f"Job {job['job_id']} ({job['kind']}): {job['from']} -> {job['to']}: {job['state']}"
    if job["state"] in ("queued", "running", "catching_up", "ready"):
        line += f", {job['done']}/{job['total']} memories"
        if job.get("eta_seconds") is not None:
            line += f", about {int(job['eta_seconds'])} s left"
    if job.get("error"):
        line += f"\n  Reason: {job['error']}"
    return line


def _status_text(data: dict) -> str:
    lines = [f"Embedding model in use: {data.get('live') or 'not recorded yet'}",
             describe(data.get("job"))]
    if data.get("previous_vectors_kept"):
        lines.append(f"Previous model kept for rollback: {data.get('previous')} "
                     "(free it with: slm embedder forget-previous)")
    return "\n".join(lines)


def _switch(args: Namespace) -> None:
    body: dict[str, Any] = {"model_name": args.model}
    if args.dimension:
        body["dimension"] = args.dimension
    if args.provider:
        from superlocalmemory.core.embedding_providers import validate_embedding_provider

        try:
            body["provider"] = validate_embedding_provider(args.provider)
        except ValueError as exc:
            _fail(args, "switch", str(exc), "INVALID")
    if args.endpoint:
        body["api_endpoint"] = args.endpoint
    data = _request(args, "switch", "POST", "", body)
    job = data["job"]
    if not getattr(args, "json", False) and not args.no_wait:
        job = _wait_for_start(job)
    _out(args, "switch", {**data, "job": job},
         f"{data.get('detail', '')}\n{describe(job)}")
    if job.get("state") == "failed":
        sys.exit(1)


def _wait_for_start(job: dict, seconds: float = 20.0) -> dict:
    """A failing model fails in its first seconds; say so before returning."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and job.get("state") == "queued":
        time.sleep(0.5)
        status = daemon_request("GET", _BASE) or {}
        latest = status.get("job") or {}
        if latest.get("job_id") == job.get("job_id"):
            job = latest
    return job


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _size_text(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def _plan_text(plan: dict) -> str:
    old, new = plan["from"], plan["to"]
    return "\n".join([
        "Upgrade memory engine",
        f"  Now:       {old['model']}",
        f"  Upgrade to: {new['model']} (the model that also reads pictures and documents)",
        f"  Memories:  {plan['memories']}",
        f"  Memory (RAM) while it works: {_size_text(plan['ram_mb'])}",
        f"  Disk: about {_size_text(plan['disk_mb'])} extra, because the previous engine's notes "
        "are kept until you free them",
        f"  Time: {plan['minutes_label']}",
        "",
        plan["explain"],
        "Undo with: slm embedder rollback",
    ])


def _confirm_upgrade(args: Namespace) -> bool:
    """--yes, or a yes at a terminal. Off a terminal (or with --json) it never starts unasked."""
    if getattr(args, "yes", False):
        return True
    if getattr(args, "json", False) or not _is_tty():
        return False
    try:
        answer = input("Upgrade now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _upgrade(args: Namespace) -> None:
    plan = _request(args, "upgrade", "GET", "/upgrade")
    if not plan.get("available"):
        if plan.get("already"):
            _out(args, "upgrade", {"plan": plan, "started": False}, plan["reason"])
            return
        _fail(args, "upgrade", plan.get("reason") or "The upgrade is not available right now.",
              "NOT_AVAILABLE")
    if not getattr(args, "json", False):
        print(_plan_text(plan))
    if not _confirm_upgrade(args):
        hint = "" if _is_tty() and not getattr(args, "json", False) else "Run again with --yes to start it."
        if getattr(args, "json", False):
            _out(args, "upgrade", {"plan": plan, "started": False}, "")
        else:
            print(f"{hint}\nNothing changed.".strip())
        return
    data = _request(args, "upgrade", "POST", "/upgrade", {})
    job = data["job"]
    if not getattr(args, "json", False) and not getattr(args, "no_wait", False):
        job = _wait_for_start(job)
    _out(args, "upgrade", {"plan": plan, "started": True, **data, "job": job},
         f"{data.get('detail', '')}\n{describe(job)}")
    if job.get("state") == "failed":
        sys.exit(1)


def _status(args: Namespace) -> None:
    data = _request(args, "status", "GET")
    _out(args, "status", data, _status_text(data))


def _simple(command: str, path: str, timeout: float = 60.0):
    def handler(args: Namespace) -> None:
        data = _request(args, command, "POST", path, {}, timeout=timeout)
        text = data.get("detail") or ""
        if "job" in data:
            text = f"{text}\n{describe(data['job'])}".strip()
        elif "freed_vectors" in data:
            text = f"Previous embedding space freed ({data['freed_vectors']} vectors)."
        _out(args, command, data, text)
    return handler


_HANDLERS = {"switch": _switch, "upgrade": _upgrade, "status": _status,
             "rollback": _simple("rollback", "/rollback", timeout=_ROLLBACK_TIMEOUT_S),
             "cancel": _simple("cancel", "/cancel"),
             "forget-previous": _simple("forget-previous", "/forget-previous")}


def cmd_embedder(args: Namespace) -> None:
    handler = _HANDLERS.get(getattr(args, "embedder_command", None) or "")
    if handler is None:
        print(__doc__.split("\n\n")[1])
        return
    handler(args)


def status_line() -> str:
    """One line for ``slm status`` while a switch is pending; empty otherwise."""
    try:
        data = daemon_request("GET", _BASE, timeout_seconds=5.0) or {}
    except Exception:
        return ""
    job = data.get("job") or {}
    if job.get("state") in ("queued", "running", "catching_up", "ready", "failed"):
        return f"  Embedding switch: {describe(job)}\n"
    return ""


def text_provider_note(config: Any) -> str:
    """One line for ``slm status`` when the daemon is down and text vectors come from it."""
    if getattr(getattr(config, "embedding", None), "provider", "") != "slm-media":
        return ""
    from superlocalmemory.core.daemon_text_embedder import NEEDS_SERVICE

    return f"  {NEEDS_SERVICE}\n"


def _json_flag(parser: Any) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")


def register_embedder_parser(sub: Any) -> None:
    p = sub.add_parser("embedder", help="Switch the embedding model in the background")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    esub = p.add_subparsers(dest="embedder_command", title="embedder subcommands")
    s = esub.add_parser("switch", help="re-index every memory with another model")
    s.add_argument("model", help="model name, e.g. nomic-ai/nomic-embed-text-v1.5")
    s.add_argument("--dimension", type=int, default=0, help="the model's vector size")
    s.add_argument("--provider", default="",
                   help="sentence-transformers | ollama | openai | slm-media (default: current)")
    s.add_argument("--endpoint", default="", help="OpenAI-compatible endpoint URL")
    s.add_argument("--no-wait", action="store_true", help="return as soon as it is queued")
    _json_flag(s)
    u = esub.add_parser("upgrade", help="move your memories to the newer memory engine (rolls back)")
    u.add_argument("--yes", "-y", action="store_true", help="do not ask")
    u.add_argument("--no-wait", action="store_true", help="return as soon as it is queued")
    _json_flag(u)
    for name, text in (("status", "progress of the current or last switch"),
                       ("rollback", "go back to the previous model (re-indexes)"),
                       ("cancel", "stop a running switch; the current model stays"),
                       ("forget-previous", "free the previous model's vectors")):
        _json_flag(esub.add_parser(name, help=text))


__all__ = ["cmd_embedder", "describe", "register_embedder_parser", "status_line", "text_provider_note"]
