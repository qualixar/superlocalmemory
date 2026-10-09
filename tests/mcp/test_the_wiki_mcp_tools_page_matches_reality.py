# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""wiki-content/MCP-Tools.md is published to the GitHub wiki, not built from
any source — nothing compared its summary line or its table to the real
profile sets, so both drifted to core 16 / code 31 / full 49 / power 61 /
whole 94 (real: 18 / 38 / 54 / 66 / 101).
"""

from __future__ import annotations

import re
from pathlib import Path

from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_PAGE = REPO_ROOT / "wiki-content" / "MCP-Tools.md"

# "whole" is deliberately absent from `_PROFILE_DEFINITIONS` (raw server, all
# tools). Pinned by tests/test_mcp/test_mcp_exposure_contract.py
# (`test_registration_exposure_is_exact_and_duplicate_free`, exposure
# "whole", expected_count 106).
_WHOLE_TOOLS_COUNT = 106


def _real_count(name: str) -> int:
    return _WHOLE_TOOLS_COUNT if name == "whole" else len(_PROFILE_DEFINITIONS[name])


# "`core` **16**, `code` **31** ... `whole` **94** ... `mesh` **8**."
_SUMMARY_CLAIM = re.compile(r"`(\w+)`\s*\*\*(\d+)\*\*")

# "| `core` | **16** |"
_TABLE_ROW = re.compile(r"^\| `(\w+)` \| \*\*(\d+)\*\* \|", flags=re.MULTILINE)


def test_the_summary_line_matches_the_real_profile_counts() -> None:
    text = WIKI_PAGE.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "Current V4 profile counts" in ln or (
        ln.startswith(">") and "`core`" in ln
    )]
    assert lines, f"{WIKI_PAGE.name} no longer has a 'Current V4 profile counts' summary line"
    summary = "\n".join(lines)
    claims = [
        (name, int(count))
        for name, count in _SUMMARY_CLAIM.findall(summary)
        if name in _PROFILE_DEFINITIONS or name == "whole"
    ]
    assert claims, (
        f"{WIKI_PAGE.name}'s summary line no longer states a tool count per "
        f"profile; if that was deliberate, delete this test rather than "
        f"leaving it passing vacuously"
    )
    for name, advertised in claims:
        real = _real_count(name)
        assert advertised == real, (
            f"{WIKI_PAGE.name} summary says {name!r} is {advertised} tools; "
            f"it actually holds {real}"
        )


def test_the_profile_table_matches_the_real_profile_counts() -> None:
    text = WIKI_PAGE.read_text(encoding="utf-8")
    rows = _TABLE_ROW.findall(text)
    assert rows, (
        f"{WIKI_PAGE.name}'s profile table no longer states a tool count per "
        f"profile; if that was deliberate, delete this test rather than "
        f"leaving it passing vacuously"
    )
    for name, advertised in rows:
        real = _real_count(name)
        assert int(advertised) == real, (
            f"{WIKI_PAGE.name} table says {name!r} is {advertised} tools; "
            f"it actually holds {real}"
        )


def test_every_named_profile_plus_whole_appears_in_the_table() -> None:
    text = WIKI_PAGE.read_text(encoding="utf-8")
    named = {name for name, _ in _TABLE_ROW.findall(text)}
    expected = set(_PROFILE_DEFINITIONS) | {"whole"}
    assert named == expected, f"table lists {sorted(named)}, expected {sorted(expected)}"
