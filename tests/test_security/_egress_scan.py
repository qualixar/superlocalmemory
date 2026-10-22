# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Find every place a module opens an outbound network connection.

An AST walk, not a grep: aliases are resolved (``import httpx as _hx``,
``import urllib.request as _urq``, ``from httpx import AsyncClient``), so
``_hx.post(...)`` is found as ``httpx.post``. Each site is reported as
``(file, enclosing function, canonical call)``.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

#: Calls that open a connection (or build a client that will).
NETWORK_CALLS = frozenset({
    *(f"httpx.{m}" for m in ("post", "get", "put", "patch", "delete", "head",
                             "options", "request", "stream", "Client", "AsyncClient")),
    "urllib.request.urlopen", "urllib.request.build_opener",
    "urllib.request.OpenerDirector",
    *(f"requests.{m}" for m in ("post", "get", "put", "patch", "delete", "head",
                                "request", "Session")),
    "aiohttp.ClientSession", "aiohttp.request",
    "http.client.HTTPConnection", "http.client.HTTPSConnection",
    "socket.create_connection", "asyncio.open_connection",
    "openai.OpenAI", "openai.AsyncOpenAI", "openai.AzureOpenAI", "openai.AsyncAzureOpenAI",
    "anthropic.Anthropic", "anthropic.AsyncAnthropic",
    "imaplib.IMAP4", "imaplib.IMAP4_SSL", "smtplib.SMTP", "smtplib.SMTP_SSL",
    "googleapiclient.discovery.build",
    "websockets.connect", "websocket.create_connection",
})

#: argv[0] values that make a subprocess an HTTP client.
NETWORK_BINARIES = frozenset({"curl", "wget", "http", "https", "nc", "ncat"})
_SUBPROCESS_CALLS = frozenset({
    "subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_call",
    "subprocess.check_output", "asyncio.create_subprocess_exec",
})

Site = tuple[str, str, str]


class _Scanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.stack: list[str] = []
        self.found: list[tuple[str, str]] = []

    # Aliases are collected in a first pass over the whole module, so an
    # import inside one function still resolves a call in another.
    def collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.asname:
                        self.aliases[name.asname] = name.name
                    else:
                        head = name.name.split(".")[0]
                        self.aliases.setdefault(head, head)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for name in node.names:
                    self.aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def _qualname(self) -> str:
        return ".".join(self.stack) or "<module>"

    def visit_FunctionDef(self, node: ast.AST) -> None:
        self.stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = self._resolve(node.func)
        if name in NETWORK_CALLS:
            self.found.append((self._qualname(), name))
        elif name in _SUBPROCESS_CALLS and node.args:
            argv = node.args[0]
            if isinstance(argv, (ast.List, ast.Tuple)) and argv.elts:
                first = argv.elts[0]
                if isinstance(first, ast.Constant) and first.value in NETWORK_BINARIES:
                    self.found.append((self._qualname(), f"subprocess:{first.value}"))
        self.generic_visit(node)

    def _resolve(self, func: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name):
            return ""
        head = self.aliases.get(func.id, func.id)
        return ".".join([head, *reversed(parts)])


def scan_source(source: str, rel: str = "<string>") -> list[Site]:
    tree = ast.parse(source)
    scanner = _Scanner()
    scanner.collect(tree)
    scanner.visit(tree)
    return [(rel, qual, call) for qual, call in scanner.found]


def scan_tree(root: Path, base: Path) -> Counter[Site]:
    sites: Counter[Site] = Counter()
    for path in sorted(root.rglob("*.py")):
        if any(part in {"node_modules", "__pycache__", "tests", ".venv", ".worktrees"} for part in path.parts):
            continue
        rel = path.relative_to(base).as_posix()
        sites.update(scan_source(path.read_text(encoding="utf-8"), rel))
    return sites
