# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Asking the same thing through a different door should not change how much
comes back.

``recall`` returned twenty and ``search`` returned ten, for the same store and
the same question, because one bound its default to the shared constant and the
other carried a literal. The constant's own documentation claimed every surface
bound to it, which had stopped being true in four places.

These read the live signatures rather than the source text, so a default that
moves after this file was written is still caught.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from superlocalmemory.core.config import (
    BROWSE_PAGE_SIZE,
    CANONICAL_LIST_LIMIT,
    CANONICAL_RECALL_LIMIT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default(func, name: str = "limit"):
    param = inspect.signature(func).parameters[name]
    assert param.default is not inspect.Parameter.empty, (
        f"{func.__qualname__} has no default for {name!r}"
    )
    return param.default


def _mcp_tools() -> dict:
    """Every MCP tool function, keyed by name, without starting a server."""
    from superlocalmemory.mcp import tools_core

    collected: dict = {}

    class _Collector:
        def tool(self, *args, **kwargs):
            def decorator(func):
                collected[func.__name__] = func
                return func
            return decorator

    tools_core.register_core_tools(_Collector(), lambda: None)
    return collected


# --- retrieval: one answer size -------------------------------------------

@pytest.mark.parametrize("tool_name", ["recall", "search"])
def test_a_retrieval_tool_returns_the_canonical_number(tool_name: str) -> None:
    tool = _mcp_tools()[tool_name]
    assert _default(tool) == CANONICAL_RECALL_LIMIT, (
        f"MCP {tool_name} defaults to {_default(tool)}; recall and search answer "
        f"the same question and must return the same many"
    )


def test_the_engine_agrees_with_the_surfaces() -> None:
    from superlocalmemory.retrieval.engine import RetrievalEngine

    assert _default(RetrievalEngine.recall) == CANONICAL_RECALL_LIMIT


def test_the_http_search_body_agrees() -> None:
    from superlocalmemory.server.api import SearchRequest

    assert SearchRequest.model_fields["limit"].default == CANONICAL_RECALL_LIMIT


# --- listing: its own number, used consistently ----------------------------

def _cli_help(command: str) -> str:
    """The shipped CLI's own help for one subcommand.

    The parser is built inside ``main()``, so there is no factory to call; the
    only honest way to read the CLI's contract is to run the CLI. Slower than
    an import, and it tests the program a user actually invokes.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "superlocalmemory.cli.main", command, "--help"],
        capture_output=True, text=True, timeout=120,
        cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert result.returncode == 0, (
        f"`slm {command} --help` exited {result.returncode}: {result.stderr[-400:]}"
    )
    return result.stdout


def test_listing_the_newest_agrees_across_mcp_and_cli() -> None:
    assert _default(_mcp_tools()["list_recent"]) == CANONICAL_LIST_LIMIT
    assert f"default {CANONICAL_LIST_LIMIT}" in _cli_help("list"), (
        f"slm list does not advertise a default of {CANONICAL_LIST_LIMIT}; "
        f"MCP list_recent returns that many"
    )


@pytest.mark.parametrize("command", ["recall", "search"])
def test_recall_and_search_agree_on_the_cli_too(command: str) -> None:
    assert f"default {CANONICAL_RECALL_LIMIT}" in _cli_help(command), (
        f"slm {command} does not advertise a default of {CANONICAL_RECALL_LIMIT}"
    )


# --- browsing: deliberately its own, and deliberately documented -----------

def test_the_paged_browser_uses_the_page_size() -> None:
    """A paged table is not an answer, so it keeps its own number.

    This is the documented divergence. It is pinned rather than left implicit
    so that if someone changes it, they change it on purpose.
    """
    from superlocalmemory.server.routes import memories

    for name in ("get_memories", "get_cluster_detail"):
        default = _default(getattr(memories, name)).default  # fastapi Query
        assert default == BROWSE_PAGE_SIZE, (
            f"{name} pages at {default}, not {BROWSE_PAGE_SIZE}"
        )


def test_no_surface_carries_a_bare_literal() -> None:
    """The constants only help if the surfaces actually reference them.

    Each of these files carried its own number at some point; the shared
    constant's docstring claimed otherwise for four releases.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
    bound = {
        "mcp/tools_core.py": ("CANONICAL_RECALL_LIMIT", "CANONICAL_LIST_LIMIT"),
        "retrieval/engine.py": ("CANONICAL_RECALL_LIMIT",),
        "server/api.py": ("CANONICAL_RECALL_LIMIT",),
        "server/routes/memories.py": ("BROWSE_PAGE_SIZE",),
        "cli/main.py": ("CANONICAL_RECALL_LIMIT", "CANONICAL_LIST_LIMIT"),
    }
    for rel, names in bound.items():
        text = (root / rel).read_text(encoding="utf-8")
        for name in names:
            assert name in text, f"{rel} no longer references {name}"
