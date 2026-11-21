# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""Every SLM skill, in every harness, must teach the shell route to memory.

Hosts that block local MCP servers, and bots or agent frameworks that never had
them, read the same skills. A skill that only names MCP tools leaves them with no
way in. One fragment (``plugin-src/fragments/no-mcp-cli.md``) is written into every
source skill by ``scripts/sync-skill-fallback.mjs`` (the first build step) and into
``slm-cli`` by ``scripts/build_cli_skill.py``; every distribution copies the skills.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRAGMENT = REPO / "plugin-src" / "fragments" / "no-mcp-cli.md"
START = "<!-- slm:no-mcp-cli:start -->"
END = "<!-- slm:no-mcp-cli:end -->"
SOURCE_SKILLS = sorted((REPO / "plugin-src" / "skills").glob("*/SKILL.md"))

# Every generated tree that ships SLM's skills to a host, with the file pattern it uses.
DISTRIBUTIONS = (
    ("plugin", "skills/*/SKILL.md"),
    ("cursor-plugin", "cursor-skills/*/SKILL.md"),
    ("codex-plugin", "skills/*/SKILL.md"),
    ("antigravity-plugin", "skills/*/SKILL.md"),
    ("hermes-plugin", "skills/*/SKILL.md"),
    ("muse-plugin", "skills/*/SKILL.md"),
    ("vscode-agent-plugin", "skills/*/SKILL.md"),
    ("copilot-plugin", ".github/prompts/slm-*.prompt.md"),
)


def _block(text: str) -> str:
    match = re.search(re.escape(START) + r"\n(.*?)\n" + re.escape(END), text, re.S)
    return match.group(1) if match else ""


def test_there_are_seventeen_source_skills():
    assert len(SOURCE_SKILLS) == 17, [p.parent.name for p in SOURCE_SKILLS]


@pytest.mark.parametrize("skill", SOURCE_SKILLS, ids=lambda p: p.parent.name)
def test_source_skill_carries_the_fragment_exactly(skill):
    fragment = FRAGMENT.read_text(encoding="utf-8").strip("\n")
    assert _block(skill.read_text(encoding="utf-8")) == fragment, (
        f"{skill.parent.name}: run node scripts/sync-skill-fallback.mjs")


@pytest.mark.parametrize("skill", SOURCE_SKILLS, ids=lambda p: p.parent.name)
def test_source_skill_may_run_slm(skill):
    head = skill.read_text(encoding="utf-8").split("\n---\n", 1)[0]
    line = next(ln for ln in head.splitlines() if ln.startswith("allowed-tools:"))
    assert "Bash(slm:*)" in line, f"{skill.parent.name} cannot run the slm CLI"


def test_fragment_names_the_whole_route():
    text = FRAGMENT.read_text(encoding="utf-8")
    for needed in ("slm status --json", "slm tool list", "slm tool describe <name>",
                   "slm tool <name> --json", "slm --json", "SLM_JSON=1", "slm help --json",
                   "slm recall", "slm remember", "slm-cli"):
        assert needed in text, needed


@pytest.mark.parametrize(("tree", "pattern"), DISTRIBUTIONS)
def test_every_distribution_ships_the_fragment(tree, pattern):
    files = sorted((REPO / tree).glob(pattern))
    assert len(files) >= 15, (tree, len(files))
    missing = [f.name if f.name != "SKILL.md" else f.parent.name
               for f in files if "When SLM's MCP tools are not in your tool list" not in
               f.read_text(encoding="utf-8")]
    assert missing == [], f"{tree} ships skills without the shell route: {missing}"


RULE_DOCS = (
    "plugin-src/rules/CLAUDE.md.fragment", "plugin-src/rules/AGENTS.md", "codex-plugin/AGENTS.md",
    "copilot-plugin/.github/copilot-instructions.md",
)


@pytest.mark.parametrize("doc", RULE_DOCS)
def test_rules_documents_teach_the_shell_route(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    assert "### When SLM's MCP tools are not in your tool list" in text, doc
    assert "slm tool <name> --json" in text and "SLM_JSON=1" in text, doc
