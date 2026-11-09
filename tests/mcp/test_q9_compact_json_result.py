# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | mcp 2.0.0 Q9 regression

"""Q9 (2026-10-06): an MCP tool result must not pay for pretty-print whitespace.

mcp==2.0.0's call-tool path serializes every dict/list tool return with
``pydantic_core.to_json(result, indent=2)`` — ``http_transport.install_compact_mcp_json``
wraps that one chokepoint so the wire text is compact instead. This exercises
the REAL composed path: a real ``SLMFastMCP`` server, the official ``Client``,
a real ``tools/call`` round trip — not a call to the tool function directly
(which never goes through the vendor's content-serialization at all).
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("SLM_MCP_EMBEDDED", "1")
os.environ.setdefault("SLM_DISABLE_WARMUP_SIDE_EFFECTS", "1")


def _big_payload() -> dict:
    """Shaped like the Q9 evidence: one big bounded list plus ordinary fields."""
    return {
        "success": True,
        "community_id": 0,
        "summary": "A very large community.",
        "member_count": 3816,
        "sample_ids": [f"fact-{i:05d}" for i in range(10)],
        "nested": {"a": 1, "b": [1, 2, 3], "c": None, "d": True},
    }


def _build_server_with_big_tool():
    from superlocalmemory.mcp.http_transport import SLMFastMCP

    server = SLMFastMCP("slm-q9-compact-json-e2e")

    @server.tool()
    def big_result() -> dict:
        """Returns a payload shaped like a Q9 community_context response."""
        return _big_payload()

    return server


@pytest.mark.asyncio
async def test_tool_result_text_has_no_pretty_print_whitespace():
    from mcp.client import Client

    server = _build_server_with_big_tool()
    async with Client(server, mode="auto") as client:
        result = await client.call_tool("big_result", {})
        assert result is not None and result.content, "no content blocks returned"
        text = result.content[0].text

    # No vendor pretty-print indentation (2-space continuation lines).
    assert "\n  " not in text, f"result text still pretty-printed: {text[:200]!r}"
    assert "\n" not in text, "compact JSON is a single line"


@pytest.mark.asyncio
async def test_tool_result_data_is_byte_identical_only_envelope_shrinks():
    """Quality guard: compacting formatting must not touch a single value."""
    from mcp.client import Client

    expected = _big_payload()
    server = _build_server_with_big_tool()
    async with Client(server, mode="auto") as client:
        result = await client.call_tool("big_result", {})
        text = result.content[0].text

    assert json.loads(text) == expected, (
        "compact serialization changed the data, not just the whitespace"
    )


@pytest.mark.asyncio
async def test_compact_result_is_meaningfully_smaller_than_pretty_printed():
    """The whole point: fewer bytes for the exact same data (Q9)."""
    from mcp.client import Client

    expected = _big_payload()
    pretty_size = len(json.dumps(expected, indent=2))

    server = _build_server_with_big_tool()
    async with Client(server, mode="auto") as client:
        result = await client.call_tool("big_result", {})
        compact_size = len(result.content[0].text)

    assert compact_size < pretty_size
    # Indentation alone roughly doubles a nested payload like this one.
    assert compact_size < pretty_size * 0.7


@pytest.mark.asyncio
async def test_a_tool_returning_a_plain_string_is_untouched():
    """Non-JSON unstructured content (a tool that already returns prose) must
    pass through unchanged — the wrapper only ever touches JSON blocks."""
    from mcp.client import Client

    from superlocalmemory.mcp.http_transport import SLMFastMCP

    server = SLMFastMCP("slm-q9-compact-json-str-e2e")

    @server.tool()
    def greeting() -> str:
        return "hello   world\nwith   odd   spacing"

    async with Client(server, mode="auto") as client:
        result = await client.call_tool("greeting", {})
        text = result.content[0].text

    assert text == "hello   world\nwith   odd   spacing"
