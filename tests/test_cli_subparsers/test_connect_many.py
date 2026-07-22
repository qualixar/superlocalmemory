# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-08 connect_many tests

"""TDD tests for hooks.portable_kit.connect_many.

Coverage:
  (b) connect_many merges WITHOUT clobbering a pre-existing unrelated MCP server
      (assert servers_preserved); uses EXISTING non-destructive merge.
  (c) connect_many is idempotent on re-run.
  (d) connect_many returns per-IDE result list.
  (e) connect_many on empty list returns [].
  (f) connect_many error on one IDE does not abort others.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from superlocalmemory.hooks.portable_kit import (  # noqa: E402
    IDE_MATRIX,
    connect_many,
    connect_ide,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pre_populate_with_other_server(config_path: Path, fmt: str, server_key: str) -> None:
    """Write a pre-existing 'othersrv' entry into the config file."""
    if fmt == "json":
        data = {
            server_key: {
                "othersrv": {
                    "command": "other-cmd",
                    "args": ["--flag"],
                    "type": "stdio",
                }
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    elif fmt == "toml":
        import tomli_w
        data = {
            server_key: {
                "othersrv": {"command": "other-cmd", "args": ["--flag"]}
            }
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(tomli_w.dumps(data).encode())

    elif fmt == "yaml":
        import yaml
        data = {
            server_key: [
                {
                    "name": "mcp",
                    "params": {
                        "serverName": "othersrv",
                        "command": "other-cmd",
                        "args": ["--flag"],
                    },
                }
            ]
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _get_preserved_count(results: list[dict[str, Any]]) -> int:
    """Sum servers_preserved across all results."""
    return sum(r.get("servers_preserved", 0) for r in results)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConnectMany:
    def test_empty_list_returns_empty(self, fake_home: Path) -> None:
        results = connect_many([], home=fake_home)
        assert results == []

    def test_returns_one_result_per_ide(self, fake_home: Path) -> None:
        ids = ["cursor", "codex"]
        results = connect_many(ids, home=fake_home)
        assert len(results) == 2
        assert results[0]["ide"] == "cursor"
        assert results[1]["ide"] == "codex"

    def test_each_result_has_required_keys(self, fake_home: Path) -> None:
        results = connect_many(["cursor"], home=fake_home)
        r = results[0]
        assert "ide" in r
        assert "mcp_config" in r
        assert "mcp_path" in r
        assert "agents_md" in r
        assert "servers_preserved" in r
        assert "error" in r

    def test_does_not_clobber_pre_existing_server_json(
        self, fake_home: Path
    ) -> None:
        """CRITICAL: pre-existing othersrv must survive; servers_preserved >= 1."""
        ide_id = "cursor"
        desc = IDE_MATRIX[ide_id]
        config_path = fake_home / desc.mcp_path_global
        _pre_populate_with_other_server(config_path, desc.fmt, desc.server_key)

        results = connect_many([ide_id], home=fake_home)
        assert results[0]["error"] is None
        assert results[0]["servers_preserved"] >= 1

        # Verify othersrv is still in the file
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "othersrv" in data[desc.server_key]
        assert "superlocalmemory" in data[desc.server_key]

    def test_does_not_clobber_pre_existing_server_toml(
        self, fake_home: Path
    ) -> None:
        """TOML target: othersrv must survive alongside new SLM entry."""
        import tomllib
        ide_id = "codex"
        desc = IDE_MATRIX[ide_id]
        config_path = fake_home / desc.mcp_path_global
        _pre_populate_with_other_server(config_path, desc.fmt, desc.server_key)

        results = connect_many([ide_id], home=fake_home)
        assert results[0]["error"] is None
        assert results[0]["servers_preserved"] >= 1

        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        assert "othersrv" in data[desc.server_key]
        assert "superlocalmemory" in data[desc.server_key]

    def test_does_not_clobber_pre_existing_server_yaml(
        self, fake_home: Path
    ) -> None:
        """YAML target: othersrv list entry must survive alongside new SLM entry."""
        import yaml
        ide_id = "continue"
        desc = IDE_MATRIX[ide_id]
        config_path = fake_home / desc.mcp_path_global
        _pre_populate_with_other_server(config_path, desc.fmt, desc.server_key)

        results = connect_many([ide_id], home=fake_home)
        assert results[0]["error"] is None
        assert results[0]["servers_preserved"] >= 1

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        providers = data.get(desc.server_key, [])
        server_names = [
            p.get("params", {}).get("serverName") for p in providers
        ]
        assert "othersrv" in server_names
        assert "superlocalmemory" in server_names

    def test_idempotent_second_run(self, fake_home: Path) -> None:
        """Second call produces byte-identical files; all statuses == unchanged."""
        ids = ["cursor", "codex"]
        connect_many(ids, home=fake_home)

        # Capture file content after first run
        snapshots = {}
        for ide_id in ids:
            path = fake_home / IDE_MATRIX[ide_id].mcp_path_global
            snapshots[ide_id] = path.read_bytes()

        # Second run
        results2 = connect_many(ids, home=fake_home)

        for ide_id in ids:
            path = fake_home / IDE_MATRIX[ide_id].mcp_path_global
            assert path.read_bytes() == snapshots[ide_id], (
                f"File changed on second run for {ide_id}"
            )
        for r in results2:
            assert r["mcp_config"] == "unchanged", (
                f"{r['ide']} was not 'unchanged' on second run: {r['mcp_config']}"
            )

    def test_error_on_one_does_not_abort_others(
        self, fake_home: Path
    ) -> None:
        """An unknown IDE in the list errors; valid IDEs still succeed."""
        ids = ["cursor", "BOGUS_IDE", "codex"]
        results = connect_many(ids, home=fake_home)
        assert len(results) == 3

        cursor_r = next(r for r in results if r["ide"] == "cursor")
        bogus_r = next(r for r in results if r["ide"] == "BOGUS_IDE")
        codex_r = next(r for r in results if r["ide"] == "codex")

        assert cursor_r["error"] is None
        assert bogus_r["error"] is not None
        assert codex_r["error"] is None

    def test_multi_ide_all_succeed(self, fake_home: Path) -> None:
        """All valid IDEs in a batch connect without errors."""
        ids = ["cursor", "windsurf", "codex", "continue"]
        results = connect_many(ids, home=fake_home)
        for r in results:
            assert r["error"] is None, f"{r['ide']} errored: {r['error']}"
            assert r["mcp_config"] in ("wrote", "merged", "unchanged")

    def test_profile_propagated_to_all(self, fake_home: Path) -> None:
        """Profile kwarg is passed through to every IDE."""
        ids = ["cursor", "windsurf"]
        connect_many(ids, home=fake_home, profile="myprofile")

        for ide_id in ids:
            desc = IDE_MATRIX[ide_id]
            config_path = fake_home / desc.mcp_path_global
            data = json.loads(config_path.read_text(encoding="utf-8"))
            block = data[desc.server_key]["superlocalmemory"]
            assert block.get("env", {}).get("SLM_MCP_PROFILE") == "myprofile"

    def test_many_with_pre_existing_multi_server_preserves_all(
        self, fake_home: Path
    ) -> None:
        """Multiple pre-existing servers are ALL preserved after connect_many."""
        ide_id = "cursor"
        desc = IDE_MATRIX[ide_id]
        config_path = fake_home / desc.mcp_path_global
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Write 3 pre-existing servers
        pre_data = {
            desc.server_key: {
                "server_a": {"command": "cmd-a"},
                "server_b": {"command": "cmd-b", "args": ["--x"]},
                "server_c": {"command": "cmd-c"},
            }
        }
        config_path.write_text(json.dumps(pre_data, indent=2), encoding="utf-8")

        results = connect_many([ide_id], home=fake_home)
        assert results[0]["servers_preserved"] == 3

        data = json.loads(config_path.read_text(encoding="utf-8"))
        servers = data[desc.server_key]
        assert "server_a" in servers
        assert "server_b" in servers
        assert "server_c" in servers
        assert "superlocalmemory" in servers
        # Pre-existing values are byte-for-byte preserved
        assert servers["server_a"] == {"command": "cmd-a"}
        assert servers["server_b"] == {"command": "cmd-b", "args": ["--x"]}
