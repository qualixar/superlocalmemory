"""A one-time upload link is the one URL a remote answer must carry whole."""

from __future__ import annotations

from superlocalmemory.server import remote_redaction

TOKEN = "Ab_-" * 10 + "Ab_"
LINK = "https://mcp.superlocalmemory.com/u/" + "a" * 32 + "/" + TOKEN


def test_the_link_is_never_rewritten_even_when_a_host_string_occurs_inside_it(monkeypatch):
    monkeypatch.setattr(remote_redaction, "_host_strings", lambda: ("b_-A", "mcp"))
    text = f"Open this link: {LINK} It works once."
    assert remote_redaction.redact_text(text) == text


def test_text_around_the_link_is_still_redacted(monkeypatch):
    monkeypatch.setattr(remote_redaction, "_host_strings", lambda: ("secretuser",))
    out = remote_redaction.redact_text(f"secretuser /Users/secretuser/a/b.png {LINK} secretuser")
    assert LINK in out and "secretuser" not in out.replace(LINK, "") and "/Users" not in out


def test_other_links_and_malformed_tokens_are_not_exempt(monkeypatch):
    monkeypatch.setattr(remote_redaction, "_host_strings", lambda: ("Ab_-",))
    other = "https://mcp.superlocalmemory.com/u/" + "a" * 32 + "/" + "Ab_-" * 3
    assert "Ab_-" not in remote_redaction.redact_text(other).split("/")[-1]
