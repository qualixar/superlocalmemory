# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""WP-06 plugin manifest tests — validates DOC-CORRECT plugin/ layout.

DOC-CORRECT layout (v3.6.14):
  .claude-plugin/marketplace.json   (repo root; source:"./plugin")
  plugin/                           (PLUGIN ROOT)
    .claude-plugin/plugin.json      (manifest; version ONLY here)
    skills/<7>/SKILL.md
    agents/*.md
    .mcp.json                       (SLM_MCP_PROFILE=code)
    hooks/hooks.json
    scripts/ensure-venv.sh          (+x)
    settings.json · requirements.txt · CLAUDE.md

Tests verify:
  - plugin.json: full §5 schema (name, version, description, author{Qualixar}, repo, keywords,
    mcpServers pointer, hooks pointer) — lives at plugin/.claude-plugin/plugin.json
  - .mcp.json: command ends with /venv/bin/slm, contains ${CLAUDE_PLUGIN_DATA}, args==["mcp"],
    NEVER bare "slm"/"python"/"superlocalmemory.mcp", env SLM_MCP_PROFILE=code, SLM_DATA_DIR set
  - marketplace.json: at repo root .claude-plugin/, NO version key in plugin entry, source="./plugin"
  - hooks.json: SessionStart fires ensure-venv.sh with ${CLAUDE_PLUGIN_ROOT}
  - settings.json: does NOT auto-allow mutating tools (forget/update_memory/delete_memory)
  - Generator: after build, all WP-06 files exist in plugin/ matching plugin-src sources
  - 7 skills present in plugin/skills/, no commands/ dir
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths — DOC-CORRECT layout
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
# plugin root (= plugin/ subdir, NOT the repo root)
PLUGIN_ROOT = REPO / "plugin"
# plugin manifest location: plugin/.claude-plugin/plugin.json
PLUGIN_CLAUDE_DIR = PLUGIN_ROOT / ".claude-plugin"
# marketplace.json at repo root .claude-plugin/ (NOT plugin root)
REPO_CLAUDE_DIR = REPO / ".claude-plugin"
PLUGIN_SRC = REPO / "plugin-src"

# 7 skills in v3.6.14
EXPECTED_SKILLS = [
    "slm-cache",
    "slm-compress",
    "slm-graph",
    "slm-recall",
    "slm-remember",
    "slm-session",
    "slm-status",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    """Load JSON; fail with clear message if missing or invalid."""
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T1 — plugin.json: at plugin/.claude-plugin/plugin.json
# ---------------------------------------------------------------------------
class TestPluginJson:
    def test_plugin_json_exists_and_parses(self) -> None:
        p = PLUGIN_CLAUDE_DIR / "plugin.json"
        assert p.exists(), f"plugin.json not found at {p} (must be at plugin/.claude-plugin/plugin.json)"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_plugin_json_has_name(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "name" in data, "plugin.json missing 'name'"
        assert data["name"] == "superlocalmemory"

    def test_plugin_json_has_version(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "version" in data, "plugin.json missing 'version'"
        assert data["version"] == "3.6.15"

    def test_plugin_json_has_description(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "description" in data, "plugin.json missing 'description'"
        assert isinstance(data["description"], str) and data["description"].strip()

    def test_plugin_json_version_matches_manifest(self) -> None:
        """Version in plugin.json must match plugin-src/manifest.json."""
        manifest = _load_json(PLUGIN_SRC / "manifest.json")
        plugin = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert plugin["version"] == manifest["version"], (
            f"plugin.json version {plugin['version']!r} != "
            f"manifest.json version {manifest['version']!r}"
        )

    def test_plugin_json_author_is_qualixar(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "author" in data, "plugin.json missing 'author'"
        author = data["author"]
        assert isinstance(author, dict), "author must be an object"
        assert author.get("name") == "Qualixar", (
            f"author.name must be 'Qualixar', got {author.get('name')!r}"
        )

    def test_plugin_json_author_has_url(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        author = data.get("author", {})
        assert "url" in author, "plugin.json author missing 'url'"
        assert author["url"].startswith("https://"), "author.url must be https://"

    def test_plugin_json_has_repository(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "repository" in data, "plugin.json missing 'repository'"
        assert "qualixar/superlocalmemory" in data["repository"], (
            f"repository must reference qualixar/superlocalmemory, got {data['repository']!r}"
        )

    def test_plugin_json_has_keywords(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "keywords" in data, "plugin.json missing 'keywords'"
        assert isinstance(data["keywords"], list), "keywords must be an array"
        assert len(data["keywords"]) > 0, "keywords must be non-empty"

    def test_plugin_json_has_mcp_servers_pointer(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "mcpServers" in data, "plugin.json missing 'mcpServers' pointer"
        mcp_ptr = str(data["mcpServers"])
        assert ".mcp.json" in mcp_ptr, (
            f"mcpServers must point to .mcp.json, got {data['mcpServers']!r}"
        )

    def test_plugin_json_has_hooks_pointer(self) -> None:
        data = _load_json(PLUGIN_CLAUDE_DIR / "plugin.json")
        assert "hooks" in data, "plugin.json missing 'hooks' pointer"
        hooks_ptr = str(data["hooks"])
        assert "hooks.json" in hooks_ptr, (
            f"hooks must reference hooks.json, got {data['hooks']!r}"
        )


# ---------------------------------------------------------------------------
# T2 — .mcp.json: at plugin/.mcp.json; SLM_MCP_PROFILE=code
# ---------------------------------------------------------------------------
class TestMcpJson:
    def _get_server(self) -> dict:
        data = _load_json(PLUGIN_ROOT / ".mcp.json")
        assert "mcpServers" in data, ".mcp.json missing 'mcpServers'"
        servers = data["mcpServers"]
        assert len(servers) == 1, f"Expected 1 server, got {len(servers)}"
        return next(iter(servers.values()))

    def test_mcp_json_exists(self) -> None:
        assert (PLUGIN_ROOT / ".mcp.json").exists(), (
            ".mcp.json not found at plugin/.mcp.json (must be at plugin root)"
        )

    def test_mcp_command_resolves_slm_launcher(self) -> None:
        """WP-F cross-platform fix: command must reference the slm-launch wrapper
        (not the hardcoded POSIX-only venv/bin/slm path).
        The launcher resolves venv/bin/slm on POSIX and venv\\Scripts\\slm.exe on Windows."""
        server = self._get_server()
        assert "command" in server, ".mcp.json server missing 'command'"
        cmd = server["command"]
        # Must reference the cross-platform launcher, not the POSIX-only venv/bin/slm
        assert "slm-launch" in cmd or cmd.endswith("/venv/bin/slm"), (
            f"command must reference slm-launch launcher or venv/bin/slm, got {cmd!r}"
        )
        # Must NOT be the bare POSIX venv path (that is the defect we fixed)
        assert cmd != "${CLAUDE_PLUGIN_DATA}/venv/bin/slm", (
            "command must not be the bare POSIX-only path ${CLAUDE_PLUGIN_DATA}/venv/bin/slm "
            "(Windows uses venv\\Scripts\\slm.exe; use slm-launch wrapper instead)"
        )

    def test_mcp_command_contains_claude_plugin_root_or_data(self) -> None:
        """Command must contain a plugin runtime env variable for portability."""
        server = self._get_server()
        cmd = server["command"]
        has_env_var = "${CLAUDE_PLUGIN_DATA}" in cmd or "${CLAUDE_PLUGIN_ROOT}" in cmd
        assert has_env_var, (
            f"command must contain ${{CLAUDE_PLUGIN_DATA}} or ${{CLAUDE_PLUGIN_ROOT}} variable, got {cmd!r}"
        )

    def test_mcp_args_empty_or_mcp(self) -> None:
        """WP-F: With the slm-launch wrapper, the launcher passes 'mcp' internally,
        so args can be [] or ['mcp']. Both are valid; bare 'slm' needs 'mcp' somewhere."""
        server = self._get_server()
        args = server.get("args", [])
        cmd = server.get("command", "")
        # If command is a launcher wrapper, args can be empty
        # If command is the direct slm binary, args must include 'mcp'
        if "slm-launch" in cmd:
            # Launcher passes mcp internally — args should be empty
            assert args == [] or args == ["mcp"], (
                f"With slm-launch wrapper, args must be [] or ['mcp'], got {args!r}"
            )
        else:
            # Direct binary — must pass 'mcp'
            assert args == ["mcp"], (
                f"args must be ['mcp'] when command is direct slm binary, got {args!r}"
            )

    def test_mcp_command_is_not_bare_slm(self) -> None:
        """Command must NOT be the bare string 'slm' (must be absolute venv path)."""
        server = self._get_server()
        cmd = server["command"]
        assert cmd != "slm", "command must not be bare 'slm'"
        assert "python" not in cmd.lower(), (
            f"command must not use python directly, got {cmd!r}"
        )

    def test_mcp_command_never_superlocalmemory_mcp(self) -> None:
        """Must not reference superlocalmemory.mcp module directly (non-runnable)."""
        server = self._get_server()
        cmd = server["command"]
        args = server.get("args", [])
        full = " ".join([cmd] + args)
        assert "superlocalmemory.mcp" not in full, (
            f"command must not reference superlocalmemory.mcp (non-runnable), got {full!r}"
        )

    def test_mcp_env_slm_mcp_profile_is_code(self) -> None:
        """v3.6.14: profile must be 'code' (20 tools, includes graph intelligence)."""
        server = self._get_server()
        env = server.get("env", {})
        assert env.get("SLM_MCP_PROFILE") == "code", (
            f"SLM_MCP_PROFILE must be 'code', got {env.get('SLM_MCP_PROFILE')!r}"
        )

    def test_mcp_env_slm_data_dir_is_set(self) -> None:
        server = self._get_server()
        env = server.get("env", {})
        assert "SLM_DATA_DIR" in env, "env missing SLM_DATA_DIR"
        assert "${CLAUDE_PLUGIN_DATA}" in env["SLM_DATA_DIR"], (
            f"SLM_DATA_DIR must reference ${{CLAUDE_PLUGIN_DATA}}, got {env['SLM_DATA_DIR']!r}"
        )


# ---------------------------------------------------------------------------
# T3 — marketplace.json: at repo root .claude-plugin/; source="./plugin"
# ---------------------------------------------------------------------------
class TestMarketplaceJson:
    def test_marketplace_json_parses(self) -> None:
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        assert isinstance(data, dict)

    def test_marketplace_has_name(self) -> None:
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        assert "name" in data, "marketplace.json missing 'name'"

    def test_marketplace_has_plugins(self) -> None:
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        assert "plugins" in data, "marketplace.json missing 'plugins'"
        assert isinstance(data["plugins"], list), "plugins must be an array"
        assert len(data["plugins"]) >= 1

    def test_marketplace_plugin_source_is_dot_slash_plugin(self) -> None:
        """DOC-CORRECT: source must be './plugin' (not './')."""
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        plugin = data["plugins"][0]
        assert "source" in plugin, "plugin entry missing 'source'"
        assert plugin["source"] == "./plugin", (
            f"source must be './plugin' (DOC-CORRECT), got {plugin['source']!r}"
        )

    def test_marketplace_plugin_entry_has_no_version_key(self) -> None:
        """Per LLD §5 AC-3: version is ONLY in plugin.json; marketplace plugin entry has NO version."""
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        plugin = data["plugins"][0]
        assert "version" not in plugin, (
            "marketplace plugin entry must NOT have 'version' key "
            "(version lives only in plugin.json)"
        )

    def test_marketplace_owner_is_qualixar(self) -> None:
        data = _load_json(REPO_CLAUDE_DIR / "marketplace.json")
        owner = data.get("owner", {})
        assert owner.get("name") == "Qualixar", (
            f"marketplace owner.name must be 'Qualixar', got {owner.get('name')!r}"
        )


# ---------------------------------------------------------------------------
# T4 — hooks.json: at plugin/hooks/hooks.json
# ---------------------------------------------------------------------------
class TestHooksJson:
    def _get_hooks_json(self) -> dict:
        p = PLUGIN_ROOT / "hooks" / "hooks.json"
        return _load_json(p)

    def test_hooks_json_exists(self) -> None:
        p = PLUGIN_ROOT / "hooks" / "hooks.json"
        assert p.exists(), f"hooks/hooks.json not found at {p} (must be at plugin/hooks/hooks.json)"

    def test_hooks_has_session_start(self) -> None:
        data = self._get_hooks_json()
        hooks = data.get("hooks", {})
        assert "SessionStart" in hooks, "hooks.json missing 'SessionStart' event"

    def test_session_start_calls_ensure_venv(self) -> None:
        data = self._get_hooks_json()
        hooks = data.get("hooks", {})
        session_start = hooks.get("SessionStart", [])
        assert len(session_start) >= 1, "SessionStart must have at least one hook group"

        all_commands = []
        for group in session_start:
            for h in group.get("hooks", []):
                if h.get("type") == "command":
                    all_commands.append(h.get("command", ""))

        assert any("ensure-venv.sh" in cmd for cmd in all_commands), (
            f"SessionStart must call ensure-venv.sh, found commands: {all_commands}"
        )

    def test_session_start_uses_claude_plugin_root(self) -> None:
        data = self._get_hooks_json()
        hooks = data.get("hooks", {})
        session_start = hooks.get("SessionStart", [])

        all_commands = []
        for group in session_start:
            for h in group.get("hooks", []):
                if h.get("type") == "command":
                    all_commands.append(h.get("command", ""))

        assert any("${CLAUDE_PLUGIN_ROOT}" in cmd for cmd in all_commands), (
            f"SessionStart command must reference ${{CLAUDE_PLUGIN_ROOT}}, got: {all_commands}"
        )


# ---------------------------------------------------------------------------
# T5 — settings.json: at plugin/settings.json; mutating tools NOT auto-allowed
# ---------------------------------------------------------------------------
class TestSettingsJson:
    MUTATING_TOOLS = {"forget", "update_memory", "delete_memory"}

    def test_settings_json_exists(self) -> None:
        p = PLUGIN_ROOT / "settings.json"
        assert p.exists(), f"settings.json not found at {p} (must be at plugin/settings.json)"

    def test_settings_json_parses(self) -> None:
        data = _load_json(PLUGIN_ROOT / "settings.json")
        assert isinstance(data, dict)

    def test_settings_does_not_auto_allow_mutating_tools(self) -> None:
        data = _load_json(PLUGIN_ROOT / "settings.json")
        permissions = data.get("permissions", {})
        allowed = permissions.get("allow", [])
        allowed_str = json.dumps(allowed)
        for tool in self.MUTATING_TOOLS:
            assert tool not in allowed_str, (
                f"settings.json must NOT auto-allow mutating tool '{tool}' "
                f"(user must confirm). Found in permissions.allow: {allowed}"
            )


# ---------------------------------------------------------------------------
# T6 — Skills layout: 7 skills in plugin/skills/, no commands/
# ---------------------------------------------------------------------------
class TestSkillsLayout:
    def test_all_7_skills_exist(self) -> None:
        for skill in EXPECTED_SKILLS:
            p = PLUGIN_ROOT / "skills" / skill / "SKILL.md"
            assert p.exists(), (
                f"plugin/skills/{skill}/SKILL.md not found — expected 7 skills in v3.6.14"
            )

    def test_no_commands_dir_in_plugin(self) -> None:
        """Commands are folded into skills in v3.6.14; no plugin/commands/ dir."""
        commands_dir = PLUGIN_ROOT / "commands"
        assert not commands_dir.exists(), (
            f"plugin/commands/ must not exist — commands folded into skills in v3.6.14"
        )

    def test_no_old_skills_in_plugin(self) -> None:
        """Old skills (slm-build-graph, slm-list-recent, etc.) must not exist."""
        retired = [
            "slm-build-graph", "slm-list-recent", "slm-show-patterns",
            "slm-switch-profile", "slm-optimize",
        ]
        for skill in retired:
            p = PLUGIN_ROOT / "skills" / skill
            assert not p.exists(), (
                f"plugin/skills/{skill}/ found but this skill is retired in v3.6.14"
            )


# ---------------------------------------------------------------------------
# T7 — Generator round-trip: plugin/ files match plugin-src sources
# ---------------------------------------------------------------------------
class TestGeneratorRoundTrip:
    """After running the build generator, plugin-src files must appear in plugin/."""

    def test_build_generator_copies_mcp_json(self) -> None:
        assert (PLUGIN_ROOT / ".mcp.json").exists(), (
            "plugin/.mcp.json must exist after build"
        )

    def test_build_generator_copies_settings_json(self) -> None:
        assert (PLUGIN_ROOT / "settings.json").exists(), (
            "plugin/settings.json must exist after build"
        )

    def test_build_generator_copies_requirements_txt(self) -> None:
        assert (PLUGIN_ROOT / "requirements.txt").exists(), (
            "plugin/requirements.txt must exist after build"
        )

    def test_build_generator_copies_hooks_json(self) -> None:
        assert (PLUGIN_ROOT / "hooks" / "hooks.json").exists(), (
            "plugin/hooks/hooks.json must exist after build"
        )

    def test_build_generator_copies_ensure_venv_sh(self) -> None:
        assert (PLUGIN_ROOT / "scripts" / "ensure-venv.sh").exists(), (
            "plugin/scripts/ensure-venv.sh must exist after build"
        )

    def test_ensure_venv_sh_is_executable(self) -> None:
        p = PLUGIN_ROOT / "scripts" / "ensure-venv.sh"
        assert p.exists(), "plugin/scripts/ensure-venv.sh must exist"
        import stat
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, (
            f"ensure-venv.sh must be user-executable (+x), mode={oct(mode)}"
        )

    def test_build_generator_emits_claude_md(self) -> None:
        """Generator must emit plugin/CLAUDE.md from rules/CLAUDE.md.fragment."""
        assert (PLUGIN_ROOT / "CLAUDE.md").exists(), (
            "plugin/CLAUDE.md must exist after build (from rules/CLAUDE.md.fragment)"
        )

    def test_claude_md_matches_fragment(self) -> None:
        """CLAUDE.md content must match the fragment source."""
        fragment = PLUGIN_SRC / "rules" / "CLAUDE.md.fragment"
        claude_md = PLUGIN_ROOT / "CLAUDE.md"
        assert fragment.exists(), f"Source fragment missing: {fragment}"
        assert claude_md.exists(), f"plugin/CLAUDE.md missing"
        assert fragment.read_text(encoding="utf-8") == claude_md.read_text(encoding="utf-8"), (
            "plugin/CLAUDE.md content does not match plugin-src/rules/CLAUDE.md.fragment"
        )

    def test_mcp_json_matches_source(self) -> None:
        src = PLUGIN_SRC / ".mcp.json"
        out = PLUGIN_ROOT / ".mcp.json"
        assert src.exists(), f"Source .mcp.json missing: {src}"
        assert out.exists(), f"plugin/.mcp.json missing"
        assert src.read_text(encoding="utf-8") == out.read_text(encoding="utf-8"), (
            "plugin/.mcp.json must match plugin-src/.mcp.json byte-for-byte"
        )

    def test_settings_json_matches_source(self) -> None:
        src = PLUGIN_SRC / "settings.json"
        out = PLUGIN_ROOT / "settings.json"
        assert src.exists(), f"Source settings.json missing: {src}"
        assert out.exists(), f"plugin/settings.json missing"
        assert src.read_text(encoding="utf-8") == out.read_text(encoding="utf-8"), (
            "plugin/settings.json must match plugin-src/settings.json byte-for-byte"
        )

    def test_requirements_txt_matches_source(self) -> None:
        src = PLUGIN_SRC / "requirements.txt"
        out = PLUGIN_ROOT / "requirements.txt"
        assert src.exists(), f"Source requirements.txt missing: {src}"
        assert out.exists(), f"plugin/requirements.txt missing"
        assert src.read_text(encoding="utf-8") == out.read_text(encoding="utf-8"), (
            "plugin/requirements.txt must match plugin-src/requirements.txt byte-for-byte"
        )
