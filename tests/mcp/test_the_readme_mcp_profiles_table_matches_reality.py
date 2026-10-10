# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""README.md's MCP tool-profiles table is a markdown table, not prose, so
none of the sibling count-checking tests parse it — it drifted to
core 16 / code 31 / mesh 8 / full 49 / power 61 / whole 94 (real:
18 / 38 / 8 / 54 / 66 / 101) with nothing catching it.

The README rewrite renamed the section ("### MCP Profiles" became
"## MCP memory server: tool profiles") and wrote the default row as
"`full` (and unset)". The section is found by what it is -- a heading naming
MCP and profiles -- not by one exact title, so the next rename does not turn
this check off.
"""

from __future__ import annotations

import re
from pathlib import Path

from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# "whole" is deliberately absent from `_PROFILE_DEFINITIONS` (it means the
# raw server, all tools). Pinned by
# tests/test_mcp/test_mcp_exposure_contract.py
# (`test_registration_exposure_is_exact_and_duplicate_free`, exposure
# "whole", expected_count 108).
_WHOLE_TOOLS_COUNT = 108

# "| `core` | 18 |" and "| `full` (and unset) | 54 |".
_ROW = re.compile(r"^\| `(\w+)`[^|`]*\| (\d+) \|", flags=re.MULTILINE)
_SECTION = re.compile(r"^#{2,3} [^\n]*\bMCP\b[^\n]*\bprofiles?\b[^\n]*$",
                      flags=re.MULTILINE | re.IGNORECASE)


def _table_text() -> str:
    text = README.read_text(encoding="utf-8")
    headings = list(_SECTION.finditer(text))
    assert len(headings) == 1, (
        "README.md must have exactly one heading naming MCP and profiles "
        f"(found {[m.group(0) for m in headings]})"
    )
    # Table ends at the next heading.
    return text[headings[0].end():].split("\n#", 1)[0]


def test_every_row_in_the_mcp_profiles_table_matches_the_real_count() -> None:
    table = _table_text()
    rows = _ROW.findall(table)
    assert rows, (
        "README.md's MCP Profiles table no longer states a tool count per "
        "profile; if that was deliberate, delete this test rather than "
        "leaving it passing vacuously"
    )

    for name, advertised in rows:
        real = _WHOLE_TOOLS_COUNT if name == "whole" else len(_PROFILE_DEFINITIONS[name])
        assert int(advertised) == real, (
            f"README.md's MCP Profiles table says {name!r} is {advertised} "
            f"tools; it actually holds {real}"
        )


def test_every_named_profile_plus_whole_appears_in_the_table() -> None:
    table = _table_text()
    named = {name for name, _ in _ROW.findall(table)}
    expected = set(_PROFILE_DEFINITIONS) | {"whole"}
    assert named == expected, (
        f"table lists {sorted(named)}, expected {sorted(expected)}"
    )


def test_the_row_for_an_unset_profile_is_full() -> None:
    """Unset registers the same 57 tools as ``full`` (pinned by
    tests/test_mcp/test_mcp_exposure_contract.py, "essential", "", 54)."""
    table = _table_text()
    unset_rows = [line for line in table.splitlines()
                  if line.startswith("|") and "unset" in line.lower()]
    assert len(unset_rows) == 1 and unset_rows[0].startswith("| `full`"), unset_rows
