# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-02 invariant: MCP mode surface must not transmit regulatory compliance claims.

An operating mode is a capability selector. EU AI Act / GDPR compliance is a
deployment assessment — never a wire-level claim from set_mode or
_mode_description.
"""

from __future__ import annotations

import inspect
import re

import pytest

from superlocalmemory.mcp import tools_v3

_COMPLIANCE_CLAIM = re.compile(
    r"ai act|compliance|compliant|gdpr compliant",
    re.IGNORECASE,
)


def test_mode_description_has_no_compliance_claim_for_any_mode() -> None:
    for mode in ("a", "b", "c", "z", ""):
        text = tools_v3._mode_description(mode)
        assert isinstance(text, str)
        assert _COMPLIANCE_CLAIM.search(text) is None, (
            f"_mode_description({mode!r}) still ships a compliance claim: {text!r}"
        )


def test_set_mode_docstring_has_no_compliance_claim() -> None:
    """set_mode is nested inside register_v3_tools; inspect source of the module site."""
    src = inspect.getsource(tools_v3.register_v3_tools)
    # Extract the set_mode docstring body from the function source.
    # The tool description sent on the MCP wire is this docstring.
    assert "async def set_mode" in src
    # Slice from set_mode through the next tool definition (get_mode or similar).
    start = src.index("async def set_mode")
    # Docstring is the first triple-quoted string after the def line.
    after = src[start:]
    q3 = after.find('"""')
    assert q3 >= 0
    q3_end = after.find('"""', q3 + 3)
    assert q3_end >= 0
    docstring = after[q3 + 3 : q3_end]
    for line in docstring.splitlines():
        assert _COMPLIANCE_CLAIM.search(line) is None, (
            f"set_mode docstring still ships a compliance claim: {line!r}"
        )


def test_mode_description_states_capability_not_legal_status() -> None:
    """Capability-only descriptions still identify the modes."""
    assert "Local Guardian" in tools_v3._mode_description("a")
    assert "zero LLM" in tools_v3._mode_description("a").lower() or "local" in tools_v3._mode_description("a").lower()
    assert "Smart Local" in tools_v3._mode_description("b")
    assert "Full Power" in tools_v3._mode_description("c")
    assert "Unknown" in tools_v3._mode_description("z")
