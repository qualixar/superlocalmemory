# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-08 portable-kit tests

"""TDD tests for hooks.portable_kit — RED first, then GREEN.

Coverage target: 100% branch coverage of connect_ide().
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from superlocalmemory.hooks.portable_kit import (  # noqa: E402
    IDE_MATRIX,
    connect_ide,
    resolve_descriptor,
    supported_ides,
)

# IDEs that use the json format (server_key → dict)
JSON_IDES = [
    ide_id
    for ide_id, d in IDE_MATRIX.items()
    if d.fmt == "json" and ide_id not in ("claude-code",)
]
TOML_IDES = [
    ide_id for ide_id, d in IDE_MATRIX.items() if d.fmt == "toml"
]
YAML_IDES = [
    ide_id for ide_id, d in IDE_MATRIX.items() if d.fmt == "yaml"
]

# All MCP-capable IDEs (not claude-code which is OUT)
MCP_IDES = [ide_id for ide_id, d in IDE_MATRIX.items() if d.fmt != ""]

# For merge tests — we need at least json/toml/yaml representatives
MERGE_IDES = JSON_IDES[:1] + TOML_IDES[:1] + YAML_IDES[:1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_home(tmp_path: Path) -> Path:
    """Return a tmp_path acting as a fake HOME directory."""
    return tmp_path


@pytest.fixture()
def fake_agents_md_source(tmp_path: Path):
    """Factory that returns a callable returning AGENTS.md content."""
    content = "# SuperLocalMemory — Agent Rules\nUse recall + remember."
    agents_file = tmp_path / "_agents_src" / "AGENTS.md"
    agents_file.parent.mkdir(parents=True, exist_ok=True)
    agents_file.write_text(content)

    def _source() -> str:
        return agents_file.read_text()

    return _source


# ---------------------------------------------------------------------------
# test_connect_writes_correct_key[ide]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ide_id", [ide_id for ide_id in MCP_IDES if ide_id != "vscode-copilot"],
)
def test_connect_writes_correct_key(ide_id: str, fake_home: Path):
    """SLM block appears under the exact server_key; command == 'slm'."""
    desc = IDE_MATRIX[ide_id]
    result = connect_ide(ide_id, home=fake_home)

    assert result["ide"] == ide_id
    assert result["error"] is None, f"Unexpected error: {result['error']}"
    assert result["mcp_config"] in ("wrote", "merged", "unchanged")

    path = fake_home / desc.mcp_path_global
    assert path.exists(), f"Config not written for {ide_id}"

    _assert_slm_block_present(path, desc)


def _assert_slm_block_present(path: Path, desc: Any) -> None:
    """Parse config and assert SLM block exists under server_key."""
    if desc.fmt == "json":
        data = json.loads(path.read_text())
        if desc.ide_id == "opencode":
            # opencode uses top-level "mcp" → nested dict
            servers = data[desc.server_key]
        else:
            servers = data[desc.server_key]
        assert "superlocalmemory" in servers
        block = servers["superlocalmemory"]
        assert block["command"] == "slm"
    elif desc.fmt == "toml":
        import tomllib
        data = tomllib.loads(path.read_text())
        servers = data[desc.server_key]
        assert "superlocalmemory" in servers
        assert servers["superlocalmemory"]["command"] == "slm"
    elif desc.fmt == "yaml":
        data = yaml.safe_load(path.read_text()) or {}
        # continue uses contextProviders list
        providers = data.get(desc.server_key, [])
        names = [p.get("params", {}).get("serverName") for p in providers]
        assert "superlocalmemory" in names


# ---------------------------------------------------------------------------
# test_merge_preserves_existing_servers[ide] — CRIT-3 deep equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ide_id", MERGE_IDES)
def test_merge_preserves_existing_servers(ide_id: str, fake_home: Path):
    """Pre-existing servers must survive byte-for-byte; no clobber."""
    desc = IDE_MATRIX[ide_id]
    config_path = fake_home / desc.mcp_path_global
    config_path.parent.mkdir(parents=True, exist_ok=True)

    other_server_snapshot = _write_other_server(config_path, desc)

    result = connect_ide(ide_id, home=fake_home)

    assert result["error"] is None
    assert result["servers_preserved"] >= 1

    # Deep-equality: the OTHER server must be exactly what we wrote
    _assert_other_server_deep_equal(config_path, desc, other_server_snapshot)


def _write_other_server(config_path: Path, desc: Any) -> Any:
    """Write a pre-existing 'othersrv' server into the config. Return snapshot."""
    if desc.fmt == "json":
        other = {
            "othersrv": {
                "command": "other-cmd",
                "args": ["--flag"],
                "type": "stdio",
                "extra_nested": {"key": "value", "num": 42},
            }
        }
        if desc.ide_id == "opencode":
            data = {desc.server_key: copy.deepcopy(other)}
        else:
            data = {desc.server_key: copy.deepcopy(other)}
        config_path.write_text(json.dumps(data, indent=2))
        return copy.deepcopy(other["othersrv"])

    elif desc.fmt == "toml":
        import tomli_w
        other = {
            "othersrv": {
                "command": "other-cmd",
                "args": ["--flag"],
            }
        }
        data = {desc.server_key: copy.deepcopy(other)}
        config_path.write_bytes(tomli_w.dumps(data).encode())
        return copy.deepcopy(other["othersrv"])

    elif desc.fmt == "yaml":
        # continue contextProviders is a list
        other_entry = {
            "name": "mcp",
            "params": {
                "serverName": "othersrv",
                "command": "other-cmd",
                "args": ["--flag"],
                "description": "some other provider",
            },
        }
        data = {desc.server_key: [copy.deepcopy(other_entry)]}
        config_path.write_text(yaml.safe_dump(data))
        return copy.deepcopy(other_entry)

    raise ValueError(f"Unknown fmt: {desc.fmt}")


def _assert_other_server_deep_equal(
    config_path: Path, desc: Any, snapshot: Any
) -> None:
    """Assert the non-SLM server in the config is DEEP-equal to snapshot."""
    if desc.fmt == "json":
        data = json.loads(config_path.read_text())
        servers = data[desc.server_key]
        assert "othersrv" in servers, "othersrv was deleted!"
        assert servers["othersrv"] == snapshot, (
            f"Deep equality failed. Expected: {snapshot}, Got: {servers['othersrv']}"
        )

    elif desc.fmt == "toml":
        import tomllib
        data = tomllib.loads(config_path.read_text())
        servers = data[desc.server_key]
        assert "othersrv" in servers
        assert servers["othersrv"] == snapshot

    elif desc.fmt == "yaml":
        data = yaml.safe_load(config_path.read_text()) or {}
        providers = data.get(desc.server_key, [])
        others = [p for p in providers if p.get("params", {}).get("serverName") != "superlocalmemory"]
        assert len(others) == 1, f"Expected 1 other provider, got {len(others)}"
        assert others[0] == snapshot, (
            f"Deep equality failed. Expected: {snapshot}, Got: {others[0]}"
        )


# ---------------------------------------------------------------------------
# test_idempotent[ide] — 2nd run byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ide_id", [ide_id for ide_id in MCP_IDES if ide_id != "vscode-copilot"],
)
def test_idempotent(ide_id: str, fake_home: Path):
    """Second call produces byte-identical file; status==unchanged."""
    desc = IDE_MATRIX[ide_id]
    connect_ide(ide_id, home=fake_home)
    config_path = fake_home / desc.mcp_path_global
    content_after_first = config_path.read_bytes()

    result2 = connect_ide(ide_id, home=fake_home)
    content_after_second = config_path.read_bytes()

    assert content_after_first == content_after_second, (
        f"Idempotency failed for {ide_id}: file changed on 2nd run"
    )
    assert result2["mcp_config"] == "unchanged"


# ---------------------------------------------------------------------------
# test_profile_env_injected
# ---------------------------------------------------------------------------


def test_profile_env_injected(fake_home: Path):
    """--profile injects SLM_MCP_PROFILE into server_block env."""
    ide_id = "cursor"
    connect_ide(ide_id, home=fake_home, profile="research")

    desc = IDE_MATRIX[ide_id]
    config_path = fake_home / desc.mcp_path_global
    data = json.loads(config_path.read_text())
    block = data[desc.server_key]["superlocalmemory"]
    assert "env" in block
    assert block["env"]["SLM_MCP_PROFILE"] == "research"


# ---------------------------------------------------------------------------
# test_here_uses_project_scope
# ---------------------------------------------------------------------------


def test_here_uses_project_scope(fake_home: Path, tmp_path: Path):
    """--here writes to project-relative path (mcp_path_project)."""
    ide_id = "cursor"
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()

    desc = IDE_MATRIX[ide_id]
    result = connect_ide(ide_id, home=fake_home, here=True, project=project_dir)

    assert result["error"] is None
    # Should have written to project dir, not home
    if desc.mcp_path_project:
        expected = project_dir / desc.mcp_path_project
        assert expected.exists(), f"Project-scoped config not found at {expected}"


# ---------------------------------------------------------------------------
# test_agents_md_placed — append under SLM-START/END, never overwrite
# ---------------------------------------------------------------------------


def test_agents_md_placed(fake_home: Path, fake_agents_md_source):
    """AGENTS.md is appended under <!-- SLM-START/END --> markers."""
    ide_id = "cursor"  # cursor supports agents_md
    desc = IDE_MATRIX[ide_id]
    assert desc.agents_md_path is not None, "cursor should have agents_md_path"

    # Pre-populate with user rules (must survive)
    agents_path = fake_home / desc.agents_md_path
    agents_path.parent.mkdir(parents=True, exist_ok=True)
    user_content = "# My existing rules\nDo not touch this.\n"
    agents_path.write_text(user_content)

    connect_ide(ide_id, home=fake_home, agents_md_source=fake_agents_md_source)

    final = agents_path.read_text()
    assert "My existing rules" in final, "User content was clobbered!"
    assert "<!-- SLM-START -->" in final
    assert "<!-- SLM-END -->" in final
    assert "SuperLocalMemory" in final


def test_agents_md_append_idempotent(fake_home: Path, fake_agents_md_source):
    """Second call with agents_md_source should not duplicate the SLM section."""
    ide_id = "cursor"
    connect_ide(ide_id, home=fake_home, agents_md_source=fake_agents_md_source)
    connect_ide(ide_id, home=fake_home, agents_md_source=fake_agents_md_source)

    desc = IDE_MATRIX[ide_id]
    agents_path = fake_home / desc.agents_md_path
    content = agents_path.read_text()
    assert content.count("<!-- SLM-START -->") == 1, "SLM section duplicated!"


# ---------------------------------------------------------------------------
# test_agents_md_skipped_when_unsupported (zed)
# ---------------------------------------------------------------------------


def test_agents_md_skipped_when_unsupported(fake_home: Path, fake_agents_md_source):
    """Zed has no rules surface; agents_md status should be 'skipped'."""
    ide_id = "zed"
    result = connect_ide(
        ide_id, home=fake_home, agents_md_source=fake_agents_md_source
    )
    assert result["agents_md"].startswith("skipped")


# ---------------------------------------------------------------------------
# test_unknown_ide_errors
# ---------------------------------------------------------------------------


def test_unknown_ide_errors(fake_home: Path, capsys):
    """Unknown ide_id returns non-zero + lists supported IDEs."""
    result = connect_ide("nonexistent-ide-xyz", home=fake_home)
    assert result["error"] is not None
    assert result["mcp_config"] == "error"
    # resolve_descriptor should return None for unknown
    assert resolve_descriptor("nonexistent-ide-xyz") is None


def test_supported_ides_lists_matrix():
    """supported_ides() returns exactly the keys from IDE_MATRIX."""
    assert set(supported_ides()) == set(IDE_MATRIX.keys())


# ---------------------------------------------------------------------------
# test_malformed_config_aborts (file BYTE-unchanged)
# ---------------------------------------------------------------------------


_MALFORMED_BY_FMT: dict[str, bytes] = {
    "json": b"{ this is : not valid json [[[",
    "toml": b"[broken toml\nkey = ??? INVALID",
    "yaml": b"key: [\n  - bad:\n    indent: [unclosed",
}


@pytest.mark.parametrize("ide_id", ["cursor", "codex", "continue"])
def test_malformed_config_aborts(ide_id: str, fake_home: Path):
    """If existing config is malformed, abort with no write; file unchanged."""
    desc = IDE_MATRIX[ide_id]
    config_path = fake_home / desc.mcp_path_global
    config_path.parent.mkdir(parents=True, exist_ok=True)

    malformed = _MALFORMED_BY_FMT[desc.fmt]
    config_path.write_bytes(malformed)

    result = connect_ide(ide_id, home=fake_home)

    assert result["mcp_config"] == "error"
    assert result["error"] is not None
    # File must be BYTE-unchanged
    assert config_path.read_bytes() == malformed, (
        f"Malformed file was modified for {ide_id}!"
    )


# ---------------------------------------------------------------------------
# test_claude_code_skips_mcp (no config, plugin pointer)
# ---------------------------------------------------------------------------


def test_claude_code_skips_mcp(fake_home: Path, capsys):
    """slm connect claude-code writes NO config, exits 0 with WP-06 pointer."""
    result = connect_ide("claude-code", home=fake_home)

    assert result["mcp_config"] == "skipped"
    assert result["error"] is None
    # No config file should be written
    desc = resolve_descriptor("claude-code")
    # claude-code has no mcp_path_global
    assert desc is not None
    assert desc.mcp_path_global == ""


# ---------------------------------------------------------------------------
# test_continue_list_dedupe
# ---------------------------------------------------------------------------


def test_continue_list_dedupe(fake_home: Path):
    """Running connect_ide for continue twice doesn't duplicate the entry."""
    connect_ide("continue", home=fake_home)
    connect_ide("continue", home=fake_home)

    desc = IDE_MATRIX["continue"]
    config_path = fake_home / desc.mcp_path_global
    data = yaml.safe_load(config_path.read_text()) or {}
    providers = data.get(desc.server_key, [])
    slm_entries = [
        p for p in providers
        if p.get("params", {}).get("serverName") == "superlocalmemory"
    ]
    assert len(slm_entries) == 1, f"Duplicate entries found: {slm_entries}"


# ---------------------------------------------------------------------------
# test_toml_codex_roundtrip
# ---------------------------------------------------------------------------


def test_toml_codex_roundtrip(fake_home: Path):
    """TOML round-trip for codex preserves the SLM block correctly."""
    import tomllib

    ide_id = "codex"
    connect_ide(ide_id, home=fake_home)

    desc = IDE_MATRIX[ide_id]
    config_path = fake_home / desc.mcp_path_global
    assert config_path.exists()

    data = tomllib.loads(config_path.read_text())
    assert "superlocalmemory" in data[desc.server_key]
    block = data[desc.server_key]["superlocalmemory"]
    assert block["command"] == "slm"
    assert block["args"] == ["mcp"]


# ---------------------------------------------------------------------------
# test_here_requires_project
# ---------------------------------------------------------------------------


def test_here_no_project_errors(fake_home: Path):
    """--here without --project should return an error."""
    result = connect_ide("cursor", home=fake_home, here=True, project=None)
    assert result["error"] is not None
    assert result["mcp_config"] == "error"


def test_vscode_copilot_requires_project_scope(fake_home: Path) -> None:
    result = connect_ide("vscode-copilot", home=fake_home)
    assert result["error"] is not None
    assert "--here" in result["error"]
    assert result["mcp_config"] == "error"
    assert not (fake_home / ".vscode" / "mcp.json").exists()


def test_vscode_copilot_here_writes_supported_project_files(
    fake_home: Path,
    fake_agents_md_source,
    tmp_path: Path,
) -> None:
    project = tmp_path / "copilot-project"
    project.mkdir()
    result = connect_ide(
        "vscode-copilot",
        home=fake_home,
        here=True,
        project=project,
        agents_md_source=fake_agents_md_source,
    )
    assert result["error"] is None
    assert (project / ".vscode" / "mcp.json").is_file()
    assert (project / ".github" / "copilot-instructions.md").is_file()


# ---------------------------------------------------------------------------
# Additional branch coverage tests
# ---------------------------------------------------------------------------


def test_merged_status_when_slm_block_changes(fake_home: Path):
    """When SLM block exists but differs, status should be 'merged'."""
    ide_id = "cursor"
    desc = IDE_MATRIX[ide_id]

    # Write SLM block with a different value
    config_path = fake_home / desc.mcp_path_global
    config_path.parent.mkdir(parents=True, exist_ok=True)
    old_data = {
        desc.server_key: {
            "superlocalmemory": {
                "command": "slm-old",
                "args": ["mcp"],
                "type": "stdio",
            }
        }
    }
    config_path.write_text(json.dumps(old_data, indent=2))

    result = connect_ide(ide_id, home=fake_home)
    # Old block existed but differs → merged
    assert result["mcp_config"] == "merged"


def test_continue_yaml_profile_injected(fake_home: Path):
    """Profile injection into yaml list block (continue)."""
    result = connect_ide("continue", home=fake_home, profile="work")
    assert result["error"] is None

    desc = IDE_MATRIX["continue"]
    config_path = fake_home / desc.mcp_path_global
    data = yaml.safe_load(config_path.read_text()) or {}
    providers = data.get(desc.server_key, [])
    slm = next(
        p for p in providers if p.get("params", {}).get("serverName") == "superlocalmemory"
    )
    assert "env" in slm.get("params", {})
    assert slm["params"]["env"]["SLM_MCP_PROFILE"] == "work"


def test_continue_yaml_merge_update(fake_home: Path):
    """When continue SLM entry exists but differs, returns 'merged'."""
    import yaml as _yaml

    desc = IDE_MATRIX["continue"]
    config_path = fake_home / desc.mcp_path_global
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write SLM entry with old args
    old_data = {
        desc.server_key: [
            {
                "name": "mcp",
                "params": {
                    "serverName": "superlocalmemory",
                    "command": "slm-old",
                    "args": ["mcp"],
                },
            }
        ]
    }
    config_path.write_text(_yaml.safe_dump(old_data))

    result = connect_ide("continue", home=fake_home)
    assert result["mcp_config"] == "merged"


def test_continue_yaml_non_list_providers_recovered(fake_home: Path):
    """If contextProviders is a dict (malformed), recover to list."""
    import yaml as _yaml

    desc = IDE_MATRIX["continue"]
    config_path = fake_home / desc.mcp_path_global
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # contextProviders is a dict instead of list — edge case
    bad_data = {desc.server_key: {"bad": "value"}}
    config_path.write_text(_yaml.safe_dump(bad_data))

    result = connect_ide("continue", home=fake_home)
    assert result["error"] is None
    assert result["mcp_config"] in ("wrote", "merged")


def test_agents_md_source_error_skips_gracefully(fake_home: Path):
    """If agents_md_source() raises, AGENTS.md is skipped without failing MCP write."""
    def _bad_source():
        raise RuntimeError("disk error")

    result = connect_ide("cursor", home=fake_home, agents_md_source=_bad_source)
    assert result["error"] is None  # MCP write succeeded
    assert result["agents_md"] == "skipped(source-error)"


def test_agents_md_skipped_when_no_source(fake_home: Path):
    """No agents_md_source → AGENTS.md skipped but MCP write proceeds."""
    result = connect_ide("cursor", home=fake_home, agents_md_source=None)
    assert result["error"] is None
    assert result["agents_md"] == "skipped(no-source)"


def test_codex_dry_run_never_writes_config_or_agents(fake_home: Path, fake_agents_md_source):
    """The exact ``slm connect codex --dry-run`` path is read-only."""
    result = connect_ide(
        "codex",
        home=fake_home,
        agents_md_source=fake_agents_md_source,
        dry_run=True,
    )

    assert result["error"] is None
    assert result["mcp_config"] == "would_write"
    assert not (fake_home / ".codex" / "config.toml").exists()
    assert not (fake_home / "AGENTS.md").exists()


def test_codex_dry_run_fails_when_packaged_agents_asset_is_missing(fake_home: Path):
    """Dry-run must prove the packaged rule asset can actually be loaded."""
    def missing_asset():
        raise FileNotFoundError("plugin-src/rules/AGENTS.md")

    result = connect_ide(
        "codex",
        home=fake_home,
        agents_md_source=missing_asset,
        dry_run=True,
    )

    assert result["agents_md"] == "error(source-unavailable)"
    assert "asset lookup failed" in result["error"]
    assert not (fake_home / ".codex" / "config.toml").exists()
    assert not (fake_home / "AGENTS.md").exists()


def test_yaml_load_config_none_result(fake_home: Path, tmp_path: Path):
    """YAML file that parses to None (empty file) is treated as empty dict."""
    import yaml as _yaml
    from superlocalmemory.hooks.portable_kit import _load_config

    empty_yaml = tmp_path / "config.yaml"
    empty_yaml.write_text("")  # empty file → yaml.safe_load returns None

    result = _load_config(empty_yaml, "yaml")
    assert result == {}


def test_yaml_load_config_non_dict_result(tmp_path: Path):
    """YAML file that parses to a scalar returns empty dict."""
    import yaml as _yaml
    from superlocalmemory.hooks.portable_kit import _load_config

    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("just a string\n")

    result = _load_config(scalar_yaml, "yaml")
    assert result == {}


def test_load_config_unknown_fmt_returns_empty(tmp_path: Path):
    """_load_config with an unknown fmt should return {}."""
    from superlocalmemory.hooks.portable_kit import _load_config

    dummy = tmp_path / "dummy.xyz"
    dummy.write_text("anything")

    result = _load_config(dummy, "unknown-fmt")
    assert result == {}


def test_atomic_write_cleans_up_tmp_on_failure(tmp_path: Path, monkeypatch):
    """If serialization fails in _atomic_write, tmp file is cleaned up."""
    import superlocalmemory.hooks.portable_kit as pk
    from superlocalmemory.hooks.portable_kit import _atomic_write

    config_path = tmp_path / "sub" / "config.json"

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pk.json, "dumps", _boom)

    with pytest.raises(RuntimeError, match="boom"):
        _atomic_write(config_path, {"key": "val"}, "json")

    # No .tmp should remain
    assert not list(tmp_path.rglob("*.tmp")), "tmp file was not cleaned up"


def test_atomic_write_unknown_format_raises(tmp_path: Path):
    """_atomic_write raises ValueError for unknown format; no tmp file left."""
    from superlocalmemory.hooks.portable_kit import _atomic_write

    config_path = tmp_path / "config.xyz"
    with pytest.raises(ValueError, match="Unknown format"):
        _atomic_write(config_path, {}, "xyz")


def test_connect_ide_write_failure_returns_error(fake_home: Path, monkeypatch):
    """If _atomic_write raises inside connect_ide, mcp_config == 'error'."""
    import superlocalmemory.hooks.portable_kit as pk

    def _failing_write(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(pk, "_atomic_write", _failing_write)

    result = connect_ide("cursor", home=fake_home)
    assert result["mcp_config"] == "error"
    assert result["error"] is not None
    assert "disk full" in result["error"]


def test_atomic_write_tmp_cleanup_on_os_replace_failure(tmp_path: Path, monkeypatch):
    """If os.replace fails after writing .tmp, the .tmp file is removed."""
    import superlocalmemory.hooks.portable_kit as pk
    from superlocalmemory.hooks.portable_kit import _atomic_write

    config_path = tmp_path / "config.json"
    real_replace = os.replace

    def _fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(pk.os, "replace", _fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _atomic_write(config_path, {"key": "val"}, "json")

    # tmp file should have been cleaned up
    assert not list(tmp_path.rglob("*.tmp")), "tmp file not cleaned up after os.replace failure"
