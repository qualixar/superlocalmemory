# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The instructions we ship to AI tools must name tools that exist.

WHAT WAS WRONG

Two shipped artifacts declared access to tools named ``slm_recall`` and
``slm_remember``. Neither has ever existed: the tools are ``recall`` and
``remember``. The names sat in the ``tools:`` line of an agent definition and the
``allowed-tools:`` line of a skill, which is where a host reads what an agent may
call — so the agent was granted two names that resolve to nothing, and its
memory step could only ever fail. It shipped that way in three built plugins.

Four other counts were wrong at the same time. The profile table told users
``core`` had 14 tools, ``code`` 29, ``full`` 47 and ``power`` 59; the real
numbers are 18, 34, 50 and 62. A recall skill said every recall runs "all seven
channels" when the implementation registers four always and two conditionally.

None of that is the sort of thing review catches by reading, because each claim
is plausible and the truth is a count in another file. So it is a test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SHIPPED_MARKDOWN = sorted(
    p for p in (REPO / "plugin-src").rglob("*.md")
) + sorted(
    p for p in (REPO / "plugin").rglob("*.md")
) if (REPO / "plugin-src").exists() else []


def _real_tool_names() -> set[str]:
    """Every name registered with @server.tool(), read from the source.

    Read from source rather than by importing and registering, because the
    registration functions need a live server object and half of them import
    optional backends.
    """
    names: set[str] = set()
    mcp_dir = REPO / "src" / "superlocalmemory" / "mcp"
    for path in mcp_dir.glob("tools_*.py"):
        pending = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("@server.tool"):
                # The decorator's own arguments can nest parentheses
                # (`@server.tool(annotations=ToolAnnotations(...))`), so this
                # tracks state line by line rather than trying to balance them
                # in one regular expression — which is how the first version of
                # this scanner missed `recall` and reported the plugins clean.
                pending = True
                continue
            if pending and stripped.startswith("@"):
                continue  # another decorator, e.g. @admits(...)
            if pending:
                match = re.match(r"(?:async )?def (\w+)\(", stripped)
                if match:
                    names.add(match.group(1))
                    pending = False
                elif stripped.startswith(("def ", "async def ", "class ")):
                    pending = False
                # Anything else is the second line of a decorator that spans
                # lines, e.g. `meta=FILE_PARAMS_META)` after
                # `@server.tool(annotations=...,` in tools_media.py.
    return names


@pytest.mark.skipif(not SHIPPED_MARKDOWN, reason="plugin sources not present")
class TestEveryToolWeTellAnAgentToUseExists:
    def test_declared_tool_access_names_real_tools(self) -> None:
        real = _real_tool_names()
        assert real, "no MCP tools found — the scanner is broken, not the plugins"

        # Host-provided tools an agent may also be granted. These are not ours
        # and their absence from our registry says nothing.
        host_tools = {
            "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Task", "WebFetch",
            "WebSearch", "TodoWrite", "NotebookEdit", "SlashCommand", "Skill",
        }
        problems: list[str] = []
        for path in SHIPPED_MARKDOWN:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                for key in ("tools:", "allowed-tools:"):
                    if not stripped.startswith(key):
                        continue
                    declared = [
                        t.strip() for t in stripped[len(key):].split(",") if t.strip()
                    ]
                    for tool in declared:
                        if tool in host_tools or tool in real:
                            continue
                        problems.append(
                            f"{path.relative_to(REPO)}: declares {tool!r}, "
                            f"which is not a tool this server registers"
                        )
        assert not problems, "\n".join(problems)

    def test_the_scanner_would_notice_a_made_up_name(self) -> None:
        """A test that cannot fail is worse than no test."""
        real = _real_tool_names()
        assert "recall" in real and "remember" in real
        assert "slm_recall" not in real, (
            "if this name ever becomes real, the check above stops meaning anything"
        )
        assert "slm_remember" not in real


@pytest.mark.skipif(not SHIPPED_MARKDOWN, reason="plugin sources not present")
class TestTheNumbersWeQuoteAreTheNumbersWeHave:
    def test_the_profile_table_matches_the_profiles(self) -> None:
        from superlocalmemory.mcp import profiles

        skill = REPO / "plugin-src" / "skills" / "slm-profile" / "SKILL.md"
        if not skill.exists():
            pytest.skip("slm-profile skill not present")
        text = skill.read_text(encoding="utf-8")

        wrong: list[str] = []
        for name in ("core", "code", "full", "power", "mesh"):
            attribute = getattr(profiles, f"_PROFILE_{name.upper()}", None)
            if attribute is None:
                continue
            actual = len(attribute)
            match = re.search(r"\|\s*`" + name + r"`\s*\|\s*(\d+)\s*tools", text)
            if match and int(match.group(1)) != actual:
                wrong.append(f"{name}: table says {match.group(1)}, really {actual}")
        assert not wrong, "; ".join(wrong)

    def test_no_skill_claims_more_retrieval_channels_than_exist(self) -> None:
        """Four register always, two more when their prerequisites are met."""
        wiring = (
            REPO / "src" / "superlocalmemory" / "core" / "engine_wiring.py"
        ).read_text(encoding="utf-8")
        assert '"spreading_activation"' in wiring and '"hopfield"' in wiring

        offenders = [
            str(p.relative_to(REPO))
            for p in SHIPPED_MARKDOWN
            if re.search(r"all seven channels|seven retrieval channels", p.read_text(encoding="utf-8"))
        ]
        assert not offenders, (
            f"these claim seven retrieval channels; there are at most six: {offenders}"
        )


@pytest.mark.skipif(not SHIPPED_MARKDOWN, reason="plugin sources not present")
class TestTheVersionWeStampIsTheVersionWeAre:
    def test_shipped_markdown_carries_the_current_version(self) -> None:
        """Packaging copies these verbatim, so a stale footer ships as-is.

        The build stamps the built plugins but never writes back to the source,
        and the source is what the Python package installs — so the footers sat
        six releases behind for every user who installed that way.
        """
        version = tomllib.loads(
            (REPO / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]

        stale: list[str] = []
        for path in sorted((REPO / "plugin-src").rglob("*.md")):
            for found in re.findall(r"SuperLocalMemory v(\d+\.\d+\.\d+)", path.read_text(encoding="utf-8")):
                if found != version:
                    stale.append(f"{path.relative_to(REPO)}: says v{found}")
        assert not stale, (
            f"package version is {version}; these disagree: " + "; ".join(stale)
        )
