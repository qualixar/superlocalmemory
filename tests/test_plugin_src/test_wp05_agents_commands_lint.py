# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Lint test — verifies plugin-src agents + rules frontmatter + allowlists.

Tests are parametrized over agent + rules md files. plugin-src/commands/ was
removed in v3.6.18 (superseded by the skills system). Uses stdlib + PyYAML only.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
SRC = REPO / "plugin-src"

# ---------------------------------------------------------------------------
# §5.0 VERIFIED REAL ARTIFACTS (single source of truth)
# ---------------------------------------------------------------------------
CORE_TOOLS: frozenset[str] = frozenset(
    {
        "remember",
        "recall",
        "search",
        "fetch",
        "list_recent",
        "update_memory",
        "forget",
        "session_init",
        "close_session",
        "slm_compress",
        "slm_retrieve",
        "slm_cache_set",
        "slm_cache_get",
        "slm_optimize_stats",
    }
)

BUILTINS: frozenset[str] = frozenset({"Read", "Bash", "Grep", "Glob", "Write", "Edit"})

ALLOWED_TOOLS: frozenset[str] = CORE_TOOLS | BUILTINS

REAL_SKILLS: frozenset[str] = frozenset(
    {
        # v3.6.14 skills (7)
        "slm-cache",
        "slm-compress",
        "slm-graph",
        "slm-recall",
        "slm-remember",
        "slm-session",
        "slm-status",
        # v3.8.0 new skills (5)
        "slm-scope",
        "slm-profile",
        "slm-governance",
        "slm-mesh",
        "slm-loop",
        # retired skills (kept for backward-compat lint of older files)
        "slm-optimize",
        "slm-build-graph",
        "slm-list-recent",
        "slm-show-patterns",
        "slm-switch-profile",
        # Skills the rules and fragment now point at (shipped in plugin-src/skills/).
        "slm-bot-memory",
        "slm-getting-started-bot",
        "slm-web-access",
        "slm-media",  # 4.1.25: pictures, PDFs and folders
    }
) | frozenset(d.name for d in (REPO / "plugin-src" / "skills").iterdir() if (d / "SKILL.md").exists())
# ^ every skill that ships counts as real (slm-cli, 4.2.32), so the list cannot go stale.

# Subagent names referenced in skill/advisor cross-refs
ADVISOR_NAMES: frozenset[str] = frozenset(
    {
        "slm-memory-advisor",
        "slm-optimize-advisor",
        "slm-governance-advisor",
        "slm-loop-runner",
    }
)

# CLI top-level verbs (from §5.0: slm list|remember|recall|search|forget|status
# plus optimize sub-commands, cache/compress, and delete)
CLI_FIRST_VERBS: frozenset[str] = frozenset(
    {
        "list",
        "remember",
        "recall",
        "search",
        "forget",
        "delete",   # slm delete <fact_id> — verified in main.py
        "status",
        "optimize",
        "cache",
        "compress",
        "loop",   # slm loop demo|history|show — bounded loops (v3.8.0)
        # 4.2.14 store upkeep, all real top-level commands (slm <cmd> --help):
        "corrections",  # overtaken | restore-overtaken | dismiss
        "cues",  # list (4.2.30)
        "ops",          # list | resolve | repair-saves | status
        "config",       # get | set
        "restart",
        "update",  # slm update <fact_id> <content> — proposes a reviewed correction
        "gdpr",    # slm gdpr status|export|erase|verify — subject-rights CLI
        "media",   # slm media enable|disable|status|gc|repair (4.1.25)
        "sources", # slm sources add|list|report|rescan|remove|forget-empty (4.1.25)
        "team",    # slm team status|policy (4.1.25)
        "token",   # slm token show (4.1.25)
        "remote",  # slm remote keys allow|disallow web-<id> mesh|media
        "mesh",    # slm mesh status|peers (the only mesh commands)
        # 4.2.30 truth layer, real top-level commands (slm <cmd> --help):
        "disputes",  # slm disputes list|settle|still-true
        "audit",     # slm audit [--count]: asks a person at a terminal
        "proposals", # slm proposals list|show|apply|reject|rollback: apply, reject, rollback ask a person
        "wiki",      # slm wiki status|diff|build|rebuild|sync|dir: rebuild and sync ask a person
        "consent",   # slm consent list|give|withdraw: give asks a person at a terminal
        "why",       # slm why FACT_ID: everything that explains one memory (read-only, WS-5)
        "signing",   # slm signing status|resign: signed memories; resign asks a person at a terminal
        "measurement",  # slm measurement status|on|off: what SLM measures, the off switch (D-28)
        "trust",     # slm trust pending|status|approve: approve asks a person at a terminal (WS-5)
        "health",    # slm health --memory: the Memory Health report (read-only, W3)
        "flags",     # slm flags [set FLAG on|off]: the 4.2.30 switches and the kill switch
        "import",    # slm import preview|run|status|list|cancel|undo: other apps' exports (WS-9)
    }
)


def _real_top_level_commands() -> frozenset[str]:
    """Every top-level command the real ``slm`` parser accepts (``slm tool``, ``slm help``, ...)."""
    import argparse  # noqa: PLC0415

    from superlocalmemory.cli.main import build_parser  # noqa: PLC0415

    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse has no public accessor
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return frozenset(action.choices)
    return frozenset()


# A verb is real when the parser has it; the hand-kept set above documents the original contract.
CLI_FIRST_VERBS = CLI_FIRST_VERBS | _real_top_level_commands()

# Banned MCP tool names (outside core, must never appear as MCP calls)
BANNED_MCP_TOOLS: frozenset[str] = frozenset({"get_status", "observe"})

# ---------------------------------------------------------------------------
# Deliverables — agents + rules only (commands/ removed in v3.6.18)
# ---------------------------------------------------------------------------
AGENT_FILES = [
    SRC / "agents" / "slm-memory-advisor.md",
    SRC / "agents" / "slm-optimize-advisor.md",
    SRC / "agents" / "slm-governance-advisor.md",
]

RULE_FILES = [
    SRC / "rules" / "AGENTS.md",
    SRC / "rules" / "CLAUDE.md.fragment",
]

ALL_MD_FILES = AGENT_FILES + RULE_FILES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_frontmatter(path: Path) -> dict:
    """Parse YAML front matter (between first two --- lines). Returns {} if none."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    # Find closing ---
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    yaml_block = text[3:end].strip()
    return yaml.safe_load(yaml_block) or {}


def _parse_tools_field(value) -> list[str]:
    """Normalise a tools/allowed-tools frontmatter value to list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    # comma or space separated string
    parts = re.split(r"[,\s]+", str(value).strip())
    return [p for p in parts if p]


def _body_text(path: Path) -> str:
    """Return the body text after the frontmatter block."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :]


# ---------------------------------------------------------------------------
# T1-A: file existence (parametrized over all 8)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", ALL_MD_FILES, ids=lambda p: p.name)
def test_file_exists(md_path: Path) -> None:
    assert md_path.exists(), f"Missing deliverable: {md_path}"


# ---------------------------------------------------------------------------
# T1-B: agent frontmatter (name==stem, has description)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", AGENT_FILES, ids=lambda p: p.name)
def test_agent_frontmatter(md_path: Path) -> None:
    fm = _parse_frontmatter(md_path)
    assert "name" in fm, f"{md_path.name}: missing 'name' in frontmatter"
    assert fm["name"] == md_path.stem, (
        f"{md_path.name}: name '{fm['name']}' != stem '{md_path.stem}'"
    )
    assert "description" in fm, f"{md_path.name}: missing 'description' in frontmatter"
    assert isinstance(fm["description"], str) and fm["description"].strip(), (
        f"{md_path.name}: 'description' must be a non-empty string"
    )


# ---------------------------------------------------------------------------
# T1-D: agent tools ⊆ CORE_TOOLS ∪ BUILTINS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", AGENT_FILES, ids=lambda p: p.name)
def test_agent_tools_are_real(md_path: Path) -> None:
    fm = _parse_frontmatter(md_path)
    tools = _parse_tools_field(fm.get("tools"))
    unknown = [t for t in tools if t not in ALLOWED_TOOLS]
    assert not unknown, (
        f"{md_path.name}: 'tools' contains unknown items: {unknown}. "
        f"Allowed: CORE_TOOLS ∪ BUILTINS"
    )


# ---------------------------------------------------------------------------
# T1-F: no invented MCP tool in body prose
# Backticked tokens that are slm_-prefixed OR in BANNED_MCP_TOOLS must be in CORE_TOOLS.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "md_path", AGENT_FILES, ids=lambda p: p.name
)
def test_no_invented_mcp_tool(md_path: Path) -> None:
    body = _body_text(md_path)
    # Extract all backtick-enclosed tokens
    backtick_tokens = re.findall(r"`([^`]+)`", body)
    # Pull out the bare name (before first '(' or space)
    violations = []
    for token in backtick_tokens:
        bare = re.split(r"[\s(]", token)[0].strip()
        if bare in BANNED_MCP_TOOLS:
            violations.append(bare)
        elif bare.startswith("slm_") and bare not in CORE_TOOLS:
            violations.append(bare)
    assert not violations, (
        f"{md_path.name}: backtick-referenced MCP tools not in core: {violations}"
    )


# ---------------------------------------------------------------------------
# T1-G: CLI verbs are real — `slm <verb>` mentions ⊆ CLI_FIRST_VERBS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "md_path", AGENT_FILES + RULE_FILES, ids=lambda p: p.name
)
def test_cli_verbs_are_real(md_path: Path) -> None:
    body = md_path.read_text(encoding="utf-8")
    # Match: `slm <verb>` or slm <verb> (backtick-wrapped or bare in code blocks)
    pattern = re.compile(r"(?:^|\s|`|\"|\')slm\s+([a-z_]+)", re.MULTILINE)
    found_verbs = {m.group(1) for m in pattern.finditer(body)}
    # Remove sub-command verbs (e.g. optimize status, optimize on/off/savings)
    # Only test top-level verb (first word after slm)
    unknown = [v for v in found_verbs if v not in CLI_FIRST_VERBS]
    assert not unknown, (
        f"{md_path.name}: unknown `slm` CLI verbs: {unknown}. "
        f"Allowed: {sorted(CLI_FIRST_VERBS)}"
    )


# ---------------------------------------------------------------------------
# T1-H: skill refs in body prose ∈ REAL_SKILLS ∪ ADVISOR_NAMES
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "md_path", AGENT_FILES + RULE_FILES, ids=lambda p: p.name
)
def test_skill_refs_are_real(md_path: Path) -> None:
    body = md_path.read_text(encoding="utf-8")
    # Match slm-<word> patterns (hyphenated skill names)
    pattern = re.compile(r"\bslm-[a-z][a-z0-9-]+")
    refs = {m.group(0) for m in pattern.finditer(body)}
    # Remove the file's own name if it appears (agents reference themselves)
    refs.discard(md_path.stem)
    unknown = [r for r in refs if r not in REAL_SKILLS and r not in ADVISOR_NAMES]
    assert not unknown, (
        f"{md_path.name}: references unknown skill/advisor names: {unknown}. "
        f"Real skills: {sorted(REAL_SKILLS)}, Advisors: {sorted(ADVISOR_NAMES)}"
    )


# ---------------------------------------------------------------------------
# T1-I: attribution line present in all files
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", ALL_MD_FILES, ids=lambda p: p.name)
def test_attribution_line(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    # Version-agnostic: the build stamps the current version into the footer, so
    # assert the attribution PATTERN (any semver) is present rather than pinning
    # a specific version that drifts every release.
    assert re.search(r"SuperLocalMemory v\d+\.\d+\.\d+", text) and "Qualixar" in text, (
        f"{md_path.name}: missing attribution line 'SuperLocalMemory vX.Y.Z · Qualixar'"
    )
