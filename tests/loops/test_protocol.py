"""Guards that bounded loops are discoverable through the cross-agent layer.

The shared ``memory_protocol_markdown()`` is embedded by the Cursor, Copilot,
and Antigravity adapters, so a bounded-loop block there reaches every one of
those front-ends automatically. These tests pin that wiring and keep the block
free of internal/competitor branding.
"""

from superlocalmemory.hooks.memory_protocol import (
    loop_protocol_markdown,
    memory_protocol_markdown,
)


def test_loop_block_is_wired_into_shared_protocol():
    protocol = memory_protocol_markdown()
    assert "bounded-loop protocol" in protocol
    assert loop_protocol_markdown() in protocol


def test_loop_block_teaches_the_core_rule_and_surface():
    block = loop_protocol_markdown()
    assert "INDEPENDENT gate" in block
    assert "advisory" in block
    assert "slm loop demo" in block
    assert "slm loop history" in block
    for status in ("DONE", "HALT", "PAUSE", "KILLED", "ERROR"):
        assert status in block


def test_loop_block_has_no_internal_or_competitor_branding():
    block = loop_protocol_markdown().lower()
    for banned in ("headroom", "omnicache", "llmlingua", "todo", "fixme"):
        assert banned not in block
