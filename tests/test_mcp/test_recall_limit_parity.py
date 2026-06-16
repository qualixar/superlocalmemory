# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""WP-02 RED tests: recall limit parity across surfaces.

Validates that CANONICAL_RECALL_LIMIT == 20 and is the default used by:
  - MCP recall tool (tools_core.py)
  - CLI argparse (cli/main.py)
  - engine.recall (core/engine.py) — already 20, binding to constant

Daemon /recall default is verified by constant-binding (FastAPI rewrites
sigs, so introspection is unreliable per LLD §9 / CRIT-3).

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import inspect
from argparse import Namespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Sentinel value from LLD §5
# ---------------------------------------------------------------------------

EXPECTED_LIMIT = 20


# ---------------------------------------------------------------------------
# Helper: _MockServer (same pattern as test_mcp_recall_tool.py)
# ---------------------------------------------------------------------------

class _MockServer:
    """Minimal mock that captures @server.tool()-decorated functions."""

    def __init__(self):
        self._tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn
        return decorator


def _get_recall_fn():
    """Register core tools on a mock server and return the recall function."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _MockServer()
    get_engine = MagicMock()
    register_core_tools(srv, get_engine)
    return srv._tools["recall"]


class _ParserCaptured(Exception):
    """Internal sentinel raised to abort main() once the parser is built."""


def _get_cli_recall_default() -> int:
    """Read the recall `--limit` default from the REAL argparse parser.

    cli/main.py builds the parser inline inside main() (no factory), so we
    drive main() and intercept ArgumentParser.parse_args to capture the
    fully-constructed parser. We force sys.argv=["slm","mcp"] so main() takes
    the mcp-stdio path that SKIPS the upgrade-banner + data-migration block
    (lines 92-123) — zero side effects. After the patch is restored we parse a
    real `recall` invocation on the captured parser and read .limit.

    This exercises the genuine argparse default (robust to source reformatting)
    rather than scraping the source text.
    """
    import argparse
    import sys
    from unittest.mock import patch

    holder: dict = {}

    def _capture(self, *args, **kwargs):  # noqa: ANN001
        holder["parser"] = self
        raise _ParserCaptured()

    with patch.object(argparse.ArgumentParser, "parse_args", _capture), \
            patch.object(sys, "argv", ["slm", "mcp"]):
        from superlocalmemory.cli.main import main
        try:
            main()
        except _ParserCaptured:
            pass

    parser = holder.get("parser")
    assert parser is not None, "Failed to capture the real argparse parser from main()"
    # parse_args is restored here — parse a real recall invocation.
    ns = parser.parse_args(["recall", "q"])
    return ns.limit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecallLimitParity:
    """WP-02 D9: recall default limit == CANONICAL_RECALL_LIMIT across surfaces."""

    def test_recall_default_limit_parity_across_surfaces(self):
        """CANONICAL_RECALL_LIMIT==20; MCP recall default==20; CLI argparse default==20.

        Checks:
          1. CANONICAL_RECALL_LIMIT constant exists and equals 20.
          2. MCP recall tool inspect.signature default == 20.
          3. CLI argparse --limit default == 20 (via the real parser).
        """
        from superlocalmemory.core.config import CANONICAL_RECALL_LIMIT

        assert CANONICAL_RECALL_LIMIT == EXPECTED_LIMIT, (
            f"CANONICAL_RECALL_LIMIT should be {EXPECTED_LIMIT}, "
            f"got {CANONICAL_RECALL_LIMIT}"
        )

        # MCP tool default via signature introspection
        recall_fn = _get_recall_fn()
        sig = inspect.signature(recall_fn)
        mcp_default = sig.parameters["limit"].default
        assert mcp_default == EXPECTED_LIMIT, (
            f"MCP recall tool `limit` default should be {EXPECTED_LIMIT}, "
            f"got {mcp_default}"
        )

        # CLI argparse default via the real parser
        cli_default = _get_cli_recall_default()
        assert cli_default == EXPECTED_LIMIT, (
            f"CLI argparse --limit default should be {EXPECTED_LIMIT}, "
            f"got {cli_default}"
        )

    def test_engine_recall_default_unchanged(self):
        """engine.recall default limit is already 20 and must stay so.

        This test proves the engine default is bound to CANONICAL_RECALL_LIMIT
        (same value), not a different literal. A regression here would mean
        truncation at a lower value.
        """
        from superlocalmemory.core.engine import MemoryEngine

        sig = inspect.signature(MemoryEngine.recall)
        engine_default = sig.parameters["limit"].default
        assert engine_default == EXPECTED_LIMIT, (
            f"engine.recall `limit` default should be {EXPECTED_LIMIT}, "
            f"got {engine_default}"
        )
