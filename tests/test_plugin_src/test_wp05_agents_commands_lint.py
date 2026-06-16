# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""WP-05 lint test — verifies plugin-src agents + commands frontmatter + allowlists.

Tests are parametrized over the 8 deliverable md files. Run RED (missing files)
before authoring, GREEN after. Uses stdlib + PyYAML only.
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
        "slm-recall",
        "slm-remember",
        "slm-status",
        "slm-optimize",
        "slm-build-graph",
        "slm-list-recent",
        "slm-show-patterns",
        "slm-switch-profile",
    }
)

# Subagent names referenced in skill/advisor cross-refs
ADVISOR_NAMES: frozenset[str] = frozenset(
    {
        "slm-memory-advisor",
        "slm-optimize-advisor",
    }
)

# CLI top-level verbs (from §5.0: slm list|remember|recall|search|forget|status
# plus optimize sub-commands and cache/compress)
CLI_FIRST_VERBS: frozenset[str] = frozenset(
    {
        "list",
        "remember",
        "recall",
        "search",
        "forget",
        "status",
        "optimize",
        "cache",
        "compress",
    }
)

# Banned MCP tool names (outside core, must never appear as MCP calls)
BANNED_MCP_TOOLS: frozenset[str] = frozenset({"get_status", "observe"})

# ---------------------------------------------------------------------------
# Deliverables (8 md files)
# ---------------------------------------------------------------------------
AGENT_FILES = [
    SRC / "agents" / "slm-memory-advisor.md",
    SRC / "agents" / "slm-optimize-advisor.md",
]

COMMAND_FILES = [
    SRC / "commands" / "slm-recall.md",
    SRC / "commands" / "slm-remember.md",
    SRC / "commands" / "slm-optimize.md",
    SRC / "commands" / "slm-status.md",
]

RULE_FILES = [
    SRC / "rules" / "AGENTS.md",
    SRC / "rules" / "CLAUDE.md.fragment",
]

ALL_MD_FILES = AGENT_FILES + COMMAND_FILES + RULE_FILES


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
# T1-C: command frontmatter (description is non-empty string)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", COMMAND_FILES, ids=lambda p: p.name)
def test_command_frontmatter(md_path: Path) -> None:
    fm = _parse_frontmatter(md_path)
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
# T1-E: command allowed-tools ⊆ CORE_TOOLS ∪ BUILTINS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("md_path", COMMAND_FILES, ids=lambda p: p.name)
def test_command_allowed_tools_are_real(md_path: Path) -> None:
    fm = _parse_frontmatter(md_path)
    tools = _parse_tools_field(fm.get("allowed-tools"))
    unknown = [t for t in tools if t not in ALLOWED_TOOLS]
    assert not unknown, (
        f"{md_path.name}: 'allowed-tools' contains unknown items: {unknown}. "
        f"Allowed: CORE_TOOLS ∪ BUILTINS"
    )


# ---------------------------------------------------------------------------
# T1-F: no invented MCP tool in body prose
# Backticked tokens that are slm_-prefixed OR in BANNED_MCP_TOOLS must be in CORE_TOOLS.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "md_path", AGENT_FILES + COMMAND_FILES, ids=lambda p: p.name
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
    "md_path", AGENT_FILES + COMMAND_FILES + RULE_FILES, ids=lambda p: p.name
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
    "md_path", AGENT_FILES + COMMAND_FILES + RULE_FILES, ids=lambda p: p.name
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
    assert "SuperLocalMemory v3.6.14" in text and "Qualixar" in text, (
        f"{md_path.name}: missing attribution line 'SuperLocalMemory v3.6.14 · Qualixar'"
    )
