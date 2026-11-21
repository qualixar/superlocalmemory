# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""``slm tool``: every MCP tool from the command line.

    slm tool list [--json] [--tool-set NAME]
    slm tool describe NAME [--json] [--tool-set NAME]
    slm tool NAME [--json '{...}'] [--args-file PATH] [-] [--key value ...]

For an agent that can run commands but has no SuperLocalMemory MCP server.
A call prints the tool's own JSON result on stdout. Exit codes: 0 the tool
succeeded, 1 the tool answered with an error (a refusal included), 2 the call
could not be made (unknown tool, bad arguments, a tool outside the tool set).
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from superlocalmemory.cli import tool_args, tool_host

EXIT_OK, EXIT_TOOL_ERROR, EXIT_USAGE = 0, 1, 2
#: ``slm tool NAME``'s own options, and whether each takes a value. A tool argument never
#: has one of these names (``RESERVED_OPTIONS``; a test checks every tool).
_CALL_OPTIONS = {"json": True, "args_file": True, "tool_set": True, "session": True,
                 "image_dir": True, "pretty": False}
#: Options of ``slm tool NAME`` itself; a tool argument with one of these names
#: is passed with ``--json`` (none of today's tools has one; a test says so).
RESERVED_OPTIONS = frozenset(_CALL_OPTIONS)

_EPILOG = """\
examples:
  slm tool list
  slm tool describe recall
  slm tool recall --query "which database does project X use" --limit 5
  slm tool remember --json '{"content": "Project X uses PostgreSQL 16", "tags": "project-x"}'
  echo '{"to": "broadcast", "message": "starting the auth refactor"}' | slm tool mesh_send -

exit codes: 0 success, 1 the tool returned an error, 2 the call could not be made
"""


def add_tool_parser(sub: Any) -> None:
    """Register ``slm tool`` on the top-level subparsers."""
    parser = sub.add_parser(
        "tool",
        help="List, describe or call any MCP tool from the command line",
        description="Call SuperLocalMemory's MCP tools from a shell, with the same "
                    "arguments, checks and results as the MCP server.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", dest="json",
                        help="JSON for list and describe; a call always prints JSON, so it "
                             "changes nothing there. After a tool name, --json '{...}' "
                             "passes the tool's arguments instead.")
    parser.add_argument("target", metavar="list|describe|TOOL",
                        help="list, describe, or the name of a tool to call")
    parser.add_argument("rest", nargs=argparse.REMAINDER, metavar="...",
                        help="options for list/describe, or the tool's arguments")


def _emit(payload: Any, pretty: bool = False) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str,
                                indent=2 if pretty else None) + "\n")
    sys.stdout.flush()


def _fail(code: str, message: str, *, tool: str = "", details: list[str] | None = None,
          exit_code: int = EXIT_USAGE) -> NoReturn:
    body: dict[str, Any] = {"success": False, "code": code, "error": message}
    if tool:
        body["tool"] = tool
    if details:
        body["details"] = details
    _emit(body)
    sys.exit(exit_code)


class _JsonErrorParser(argparse.ArgumentParser):
    """A usage error is a JSON failure on stdout, like every other failure of ``slm tool``."""

    def error(self, message: str) -> NoReturn:
        _fail("usage", f"{self.prog}: {message}")


def _sub_parser(prog: str) -> argparse.ArgumentParser:
    parser = _JsonErrorParser(prog=prog, allow_abbrev=False, add_help=True)
    parser.add_argument("--tool-set", default=None,
                        help="core, code, full, power, mesh, whole (default) or mcp "
                             "(what an MCP connection on this computer sees now)")
    return parser


def _host(tool_set: str | None) -> Any:
    try:
        return tool_host.build_host(tool_set)
    except tool_host.UnknownToolSet as exc:
        _fail("unknown_tool_set", str(exc))


def _cmd_list(rest: list[str]) -> None:
    parser = _sub_parser("slm tool list")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    opts = parser.parse_args(rest)
    host = _host(opts.tool_set)
    tools = sorted(tool_host.listed_tools(host), key=lambda t: t.name)
    shown = (opts.tool_set or tool_host.DEFAULT_TOOL_SET).lower()
    if opts.json:
        _emit({"success": True, "tool_set": shown, "count": len(tools), "tools": [
            {"name": t.name, "title": t.title or "", "summary": tool_host.one_line(t.description),
             "cli": tool_host.cli_form(t.name, t.input_schema)} for t in tools]})
        return
    print(f"{len(tools)} tools (tool set: {shown}). Details: slm tool describe <name>")
    width = max((len(t.name) for t in tools), default=0)
    for t in tools:
        print(f"  {t.name:<{width}}  {tool_host.one_line(t.description)[:100]}")


def _find(host: Any, name: str) -> Any:
    """The registered tool ``name`` that the tool set allows, or a usage failure."""
    tools = {t.name: t for t in tool_host.registered_tools(host)}
    if name not in tools:
        close = difflib.get_close_matches(name, tools, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else " See: slm tool list"
        _fail("unknown_tool", f"There is no tool named {name!r}.{hint}", tool=name)
    allowed = getattr(host, "_slm_allowed", None)
    if allowed is not None and name not in allowed:
        _fail("not_in_tool_set",
              f"{name} is not in this tool set. Leave out --tool-set to reach every tool.",
              tool=name)
    return tools[name]


def _cmd_describe(rest: list[str]) -> None:
    parser = _sub_parser("slm tool describe")
    parser.add_argument("name")
    parser.add_argument("--json", action="store_true", help="Output structured JSON")
    opts = parser.parse_args(rest)
    host = _host(opts.tool_set)
    tool = _find(host, opts.name)
    listed = {t.name: t for t in tool_host.listed_tools(host)}
    mcp_tool = listed.get(opts.name)
    record = tool_host.tool_record(mcp_tool) if mcp_tool is not None else {
        **tool_host.tool_record(_as_listed(tool)), "listed": False,
        "note": "Not in the MCP tool list right now (pictures and documents are off); "
                "a call still answers, with the reason."}
    if opts.json:
        _emit({"success": True, **record})
        return
    print(f"{record['name']}  {record['title']}".rstrip())
    print()
    print(record["description"].strip())
    print()
    print(f"call: {record['cli']}  (or --json '{{...}}')")
    props = (record["input_schema"] or {}).get("properties") or {}
    required = set((record["input_schema"] or {}).get("required") or [])
    for arg, spec in props.items():
        mark = "required" if arg in required else f"default {json.dumps(spec.get('default'))}"
        print(f"  --{arg.replace('_', '-'):<24} {_type_label(spec):<16} {mark}")


def _as_listed(tool: Any) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(name=tool.name, title=tool.title, description=tool.description,
                           input_schema=tool.parameters, annotations=tool.annotations)


def _type_label(spec: dict[str, Any]) -> str:
    kinds = sorted(tool_args._types(spec) - {"null"})
    return "/".join(kinds) or "any"


_HELP = ("-h", "--help")


@dataclass(frozen=True)
class _Call:
    """``slm tool NAME`` tokens, split into slm tool's own options and the tool's arguments."""

    options: dict[str, Any]
    tool_tokens: list[str]
    use_stdin: bool
    wants_help: bool


def _split_call(rest: list[str]) -> _Call:
    """Read the tokens after the tool name, left to right.

    The token after a tool argument (``--value``) is always that argument's value, whatever
    it looks like: ``--value -h``, ``--value --json`` and ``--value -`` send those strings to
    the tool. Only a token in an option's position is one of slm tool's own options, a lone
    ``-`` (arguments on stdin), or ``-h``/``--help`` (the tool's description).
    """
    options: dict[str, Any] = {}
    tool_tokens: list[str] = []
    use_stdin = wants_help = False
    i = 0
    while i < len(rest):
        token = rest[i]
        key, sep, value = token[2:].partition("=") if token.startswith("--") else ("", "", "")
        name = key.replace("-", "_")
        if token in _HELP:
            wants_help = True
        elif token == "-":
            use_stdin = True
        elif name in _CALL_OPTIONS:
            if not _CALL_OPTIONS[name]:
                if sep:
                    raise tool_args.ArgumentProblem(f"--{key} takes no value")
                options[name] = True
            elif sep:
                options[name] = value
            elif i + 1 < len(rest):
                i += 1
                options[name] = rest[i]
            else:
                raise tool_args.ArgumentProblem(f"--{key} needs a value")
        elif key and not sep and i + 1 < len(rest):
            tool_tokens += [token, rest[i + 1]]  # a tool argument and its value
            i += 1
        else:
            tool_tokens.append(token)  # --key=value, or a stray token flag_pairs reports
        i += 1
    return _Call(options, tool_tokens, use_stdin, wants_help)


def _cmd_call(name: str, rest: list[str]) -> None:
    try:
        call = _split_call(rest)
    except tool_args.ArgumentProblem as exc:
        _fail("invalid_arguments", str(exc), tool=name)
    opts = call.options
    if call.wants_help:  # a tool's help is its description
        tool_set = opts.get("tool_set")
        _cmd_describe([name, *(["--tool-set", tool_set] if tool_set else [])])
        return
    host = _host(opts.get("tool_set"))
    tool = _find(host, name)
    try:
        arguments = tool_args.gather(
            json_text=opts.get("json"), args_file=opts.get("args_file"),
            use_stdin=call.use_stdin, pairs=tool_args.flag_pairs(call.tool_tokens),
            schema=tool.parameters)
        tool_args.validate(arguments, tool.parameters)
    except tool_args.ArgumentProblem as exc:
        _fail("invalid_arguments", str(exc), tool=name, details=exc.details)
    # The daemon first: a daemon this call starts must never see the session bound below.
    _ensure_daemon()
    if name.startswith("mesh_"):
        tool_host.bind_mesh_session(opts.get("session"))
    result = _run(host, name, arguments)
    image_dir = opts.get("image_dir")
    payload, is_error = tool_host.result_payload(result, Path(image_dir) if image_dir else None)
    _emit(payload, bool(opts.get("pretty")))
    if is_error:
        sys.exit(EXIT_TOOL_ERROR)


def _ensure_daemon() -> None:
    """Start the daemon if it is down, as the stdio MCP server does when it starts."""
    import os

    if os.environ.get("SLM_DISABLE_WARMUP_SIDE_EFFECTS") == "1":  # harnesses: never start one
        return
    try:
        from superlocalmemory.cli.daemon import ensure_daemon

        ensure_daemon()
    except Exception:  # noqa: BLE001 - the tool itself reports an unavailable daemon
        pass


def _run(host: Any, name: str, arguments: dict[str, Any]) -> Any:
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    try:
        return asyncio.run(host.call_tool(name, arguments))
    except UnexpectedToolError as exc:
        cause = exc.__cause__ or exc
        _fail("tool_failed", f"{name} failed: {type(cause).__name__}: {cause}"[:500],
              tool=name, exit_code=EXIT_TOOL_ERROR)
    except ToolError as exc:
        _fail("tool_error", str(exc)[:500], tool=name)
    except ValueError as exc:  # SLMFastMCP: not enabled for this connection
        _fail("not_in_tool_set", str(exc), tool=name)


def _quiet_logs() -> None:
    """Progress logs go nowhere: stdout carries the result, stderr only real problems."""
    import logging

    for name in ("", "superlocalmemory"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_tool(args: argparse.Namespace) -> None:
    """Dispatch ``slm tool list|describe|NAME``."""
    _quiet_logs()
    target = args.target
    rest = list(args.rest or [])
    # `slm --json tool list` / `slm tool --json describe X`: list and describe print
    # text unless asked; a call always prints JSON, so it needs nothing.
    if getattr(args, "json", False) and target in ("list", "describe") and "--json" not in rest:
        rest.append("--json")
    if target == "list":
        _cmd_list(rest)
    elif target == "describe":
        _cmd_describe(rest)
    else:
        _cmd_call(target, rest)


__all__ = ["EXIT_OK", "EXIT_TOOL_ERROR", "EXIT_USAGE", "RESERVED_OPTIONS",
           "add_tool_parser", "cmd_tool"]
