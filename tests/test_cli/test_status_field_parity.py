# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""WP-02 RED tests: CLI↔MCP status field parity.

Validates that cli/commands.py cmd_status --json and mcp/tools_core.py
get_status return the same canonical KEY SET (not value-case).

Canonical field set (LLD §5):
    mode, provider, profile, base_dir, db_path, db_size_mb,
    fact_count, entity_count, edge_count

Envelope keys (success, next_actions) are excluded from the key-set test.
v3.6.12 fields that must NOT be removed: the fields already present
before WP-02.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from argparse import Namespace
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Canonical definition (mirrors LLD §5)
# ---------------------------------------------------------------------------

CANONICAL_STATUS_FIELDS = frozenset({
    "mode",
    "provider",
    "profile",
    "base_dir",
    "db_path",
    "db_size_mb",
    "fact_count",
    "entity_count",
    "edge_count",
})

# Fields that were present in v3.6.12 CLI json output — must never be removed.
# db_size_mb was emitted (conditionally, when the db existed) in v3.6.12 and is
# now unconditional; anchored here so a future refactor can't silently drop it.
_CLI_V3612_FIELDS = frozenset({"mode", "provider", "base_dir", "db_path", "db_size_mb"})

# Fields that were present in v3.6.12 MCP get_status — must never be removed.
_MCP_V3612_FIELDS = frozenset({"mode", "profile", "fact_count", "entity_count",
                                "edge_count", "db_size_mb"})

# Envelope-level keys excluded from the field-set equality test.
_ENVELOPE_EXCLUDE = frozenset({"success", "next_actions"})


# ---------------------------------------------------------------------------
# Helpers
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


def _build_mock_engine(db_path: Path, mode_value: str = "b") -> MagicMock:
    """Build a fully-configured mock MemoryEngine for get_status tests."""
    mock_engine = MagicMock()
    mock_engine.profile_id = "default"
    mock_engine._db.get_fact_count.return_value = 5
    # execute() is called twice: entities + edges counts
    mock_engine._db.execute.return_value = [{"c": 10}]
    mock_engine._db.db_path = db_path
    mock_engine._config.mode.value = mode_value
    return mock_engine


def _run_mcp_get_status(db_path: Path, mode_value: str = "b") -> dict:
    """Register core tools and invoke get_status via a _MockServer."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    srv = _MockServer()
    get_engine_mock = MagicMock()
    mock_engine = _build_mock_engine(db_path, mode_value)
    get_engine_mock.return_value = mock_engine

    register_core_tools(srv, get_engine_mock)
    get_status_fn = srv._tools["get_status"]

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(get_status_fn())
    finally:
        loop.close()


def _run_cli_status_json(
    tmpdir: Path,
    *,
    mock_engine_counts: tuple[int, int, int] = (3, 2, 4),
) -> dict:
    """Run cmd_status --json; return parsed data dict (the envelope's 'data' key).

    Patches SLMConfig to avoid real I/O. After GREEN, cmd_status json branch
    opens a MemoryEngine read-only for counts — we patch MemoryEngine too.

    SLMConfig and MemoryEngine are imported inline inside cmd_status, so we
    patch at their source module paths (not the commands module namespace).
    """
    from superlocalmemory.cli.commands import cmd_status

    args = Namespace(json=True, verbose=False)
    captured = StringIO()

    with patch("superlocalmemory.core.config.SLMConfig.load") as mock_load:
        mock_config = MagicMock()
        mock_config.mode.value = "b"
        mock_config.active_profile = "default"
        mock_config.llm.provider = "ollama"
        mock_config.base_dir = tmpdir
        mock_config.db_path = tmpdir / "memory.db"
        mock_load.return_value = mock_config

        # After GREEN, cmd_status json branch opens MemoryEngine (LIGHT mode)
        # for counts. Patch at the module where it's defined so the inline
        # import inside cmd_status picks it up.
        with patch(
            "superlocalmemory.core.engine.MemoryEngine",
        ) as mock_engine_cls:
            mock_eng = MagicMock()
            fact_count, entity_count, edge_count = mock_engine_counts
            mock_eng._db.get_fact_count.return_value = fact_count
            mock_eng._db.execute.return_value = [{"c": entity_count}]
            mock_engine_cls.return_value = mock_eng

            with patch("builtins.print", side_effect=lambda s: captured.write(str(s) + "\n")):
                cmd_status(args)

    output = captured.getvalue().strip()
    envelope = json.loads(output)
    return envelope.get("data", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStatusFieldParity:
    """WP-02 D8: CLI json and MCP get_status must expose the same key set."""

    def test_cli_mcp_status_canonical_fields_equal(self):
        """cli_keys == CANONICAL == mcp_keys (key sets; NOT value equality).

        Mode value-case intentionally differs (CLI .upper(), MCP lower) —
        see LLD §STAGE-5 Decision A. Only key presence is asserted.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "memory.db"
            db_path.touch()

            mcp_result = _run_mcp_get_status(db_path)
            mcp_keys = set(mcp_result.keys()) - _ENVELOPE_EXCLUDE

            cli_data = _run_cli_status_json(tmpdir_path)
            cli_keys = set(cli_data.keys())

        assert cli_keys == CANONICAL_STATUS_FIELDS, (
            f"CLI keys differ from canonical.\n"
            f"  CLI:       {sorted(cli_keys)}\n"
            f"  CANONICAL: {sorted(CANONICAL_STATUS_FIELDS)}\n"
            f"  Missing:   {sorted(CANONICAL_STATUS_FIELDS - cli_keys)}\n"
            f"  Extra:     {sorted(cli_keys - CANONICAL_STATUS_FIELDS)}"
        )
        assert mcp_keys == CANONICAL_STATUS_FIELDS, (
            f"MCP keys differ from canonical.\n"
            f"  MCP:       {sorted(mcp_keys)}\n"
            f"  CANONICAL: {sorted(CANONICAL_STATUS_FIELDS)}\n"
            f"  Missing:   {sorted(CANONICAL_STATUS_FIELDS - mcp_keys)}\n"
            f"  Extra:     {sorted(mcp_keys - CANONICAL_STATUS_FIELDS)}"
        )
        assert cli_keys == mcp_keys, (
            f"CLI and MCP key sets differ.\n"
            f"  CLI only: {sorted(cli_keys - mcp_keys)}\n"
            f"  MCP only: {sorted(mcp_keys - cli_keys)}"
        )

    def test_cli_status_no_v3612_field_removed(self):
        """CLI json output must retain all fields that existed in v3.6.12.

        v3.6.12 CLI fields: mode, provider, base_dir, db_path.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cli_data = _run_cli_status_json(tmpdir_path)

        missing = _CLI_V3612_FIELDS - set(cli_data.keys())
        assert not missing, (
            f"v3.6.12 backward-compat BROKEN: CLI json lost fields: {sorted(missing)}"
        )

    def test_mcp_status_no_v3612_field_removed(self):
        """MCP get_status must retain all fields that existed in v3.6.12.

        v3.6.12 MCP fields: mode, profile, fact_count, entity_count,
        edge_count, db_size_mb.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "memory.db"
            db_path.touch()

            mcp_result = _run_mcp_get_status(db_path)

        mcp_keys = set(mcp_result.keys()) - _ENVELOPE_EXCLUDE
        missing = _MCP_V3612_FIELDS - mcp_keys
        assert not missing, (
            f"v3.6.12 backward-compat BROKEN: MCP get_status lost fields: "
            f"{sorted(missing)}"
        )

    def test_cli_status_json_does_not_create_db_on_fresh_install(self):
        """REGRESSION (WP-02 review BLOCKER): `slm status --json` must stay
        observational. On a fresh install (no db yet) it must NOT open the
        engine — which would mkdir/connect/migrate the db as a side effect.

        Crucially this does NOT patch MemoryEngine: if the db-existence guard
        were removed, cmd_status would import + initialize the REAL engine and
        create the db, failing the post-condition.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "memory.db"
            assert not db_path.exists()  # precondition: fresh install

            args = Namespace(json=True, verbose=False)
            captured = StringIO()
            with patch("superlocalmemory.core.config.SLMConfig.load") as mock_load:
                mock_config = MagicMock()
                mock_config.mode.value = "b"
                mock_config.active_profile = "default"
                mock_config.llm.provider = "ollama"
                mock_config.base_dir = tmpdir_path
                mock_config.db_path = db_path
                mock_load.return_value = mock_config
                from superlocalmemory.cli.commands import cmd_status
                with patch("builtins.print",
                           side_effect=lambda s: captured.write(str(s) + "\n")):
                    cmd_status(args)

            # POST-CONDITION: no write side-effect — db was NOT created.
            assert not db_path.exists(), (
                "BLOCKER REGRESSION: `slm status --json` created the database "
                "on a fresh install (engine opened without a db-existence guard)."
            )
            data = json.loads(captured.getvalue().strip()).get("data", {})
            # Full canonical key set still present, counts fail-open to 0.
            assert set(data.keys()) == CANONICAL_STATUS_FIELDS
            assert data["fact_count"] == 0
            assert data["entity_count"] == 0
            assert data["edge_count"] == 0

    def test_cli_status_json_fails_open_with_canonical_keys(self):
        """When the db exists but engine init RAISES, status must not crash and
        must still emit the full canonical key set with counts = 0 (fail-open).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            db_path = tmpdir_path / "memory.db"
            db_path.touch()  # db present → guard lets the engine open

            args = Namespace(json=True, verbose=False)
            captured = StringIO()
            with patch("superlocalmemory.core.config.SLMConfig.load") as mock_load:
                mock_config = MagicMock()
                mock_config.mode.value = "b"
                mock_config.active_profile = "default"
                mock_config.llm.provider = "ollama"
                mock_config.base_dir = tmpdir_path
                mock_config.db_path = db_path
                mock_load.return_value = mock_config
                from superlocalmemory.cli.commands import cmd_status
                with patch("superlocalmemory.core.engine.MemoryEngine") as mock_cls:
                    mock_cls.return_value.initialize.side_effect = RuntimeError("boom")
                    with patch("builtins.print",
                               side_effect=lambda s: captured.write(str(s) + "\n")):
                        cmd_status(args)  # must NOT raise

            data = json.loads(captured.getvalue().strip()).get("data", {})
            assert set(data.keys()) == CANONICAL_STATUS_FIELDS, (
                "Fail-open path must emit the SAME canonical key set."
            )
            assert data["fact_count"] == 0
            assert data["entity_count"] == 0
            assert data["edge_count"] == 0
