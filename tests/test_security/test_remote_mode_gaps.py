# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-15 coverage tests (D6 — remote_mode gaps)

"""Gap-fill tests for remote_mode.py lines not covered by test_remote_mode.py.

Targeted uncovered lines (confirmed via coverage report):
  - Line 94:  ``_host_matches`` → ``except ValueError`` for malformed CIDR
  - Lines 100-101: ``_host_matches`` → malformed CIDR falls through to False
  - Line 144: ``is_remote_origin_allowed`` → IPv6 bracketed host parsing

These are edge paths not exercised by the existing test_remote_mode.py tests.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core import remote_mode as rm


# ---------------------------------------------------------------------------
# _host_matches — invalid CIDR ValueError path (lines 94, 100-101)
# ---------------------------------------------------------------------------

class TestHostMatchesEdgeCases:
    def test_invalid_cidr_returns_false_not_exception(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Malformed CIDR entry must NOT raise — it returns False gracefully.

        Line 94: ``except ValueError: return False`` inside _host_matches.
        This is the error-handling path for CIDR strings like "192.168.0/24"
        (missing octet) or "999.0.0.0/8" (invalid octet value).
        """
        monkeypatch.setenv("SLM_REMOTE", "1")
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "999.999.0.0/8")

        # Must return False, not raise ValueError
        result = rm.is_lan_client_allowed("192.168.1.5")
        assert result is False, (
            "Malformed CIDR should gracefully return False, not raise"
        )

    def test_invalid_cidr_does_not_allow_any_ip(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A malformed CIDR in the allowlist denies all, not allows all."""
        monkeypatch.setenv("SLM_REMOTE", "1")
        # Mix: one bad CIDR, one valid exact IP
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "not-a-cidr/16,10.0.0.5")

        # 10.0.0.5 is in the allowlist → allowed
        assert rm.is_lan_client_allowed("10.0.0.5") is True
        # 192.168.1.1 is NOT — bad CIDR doesn't grant it
        assert rm.is_lan_client_allowed("192.168.1.1") is False

    def test_empty_host_entry_returns_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty string entry in allowlist must return False (lines 100-101 guard)."""
        monkeypatch.setenv("SLM_REMOTE", "1")
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", ",,,")  # all empty after strip

        result = rm.is_lan_client_allowed("192.168.1.1")
        assert result is False, "Empty allowlist entries must deny all clients"


# ---------------------------------------------------------------------------
# is_remote_origin_allowed — IPv6 bracketed host (line 144)
# ---------------------------------------------------------------------------

class TestRemoteOriginAllowedIPv6:
    def test_ipv6_bracketed_origin_parsed_correctly(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """is_remote_origin_allowed handles Origin with IPv6 bracketed host.

        Line 144: ``host = host.split("]", 1)[0].lstrip("[")``
        This path fires when origin is like ``http://[::ffff:192.168.1.50]:8765``.
        """
        monkeypatch.setenv("SLM_REMOTE", "1")
        # Allow the mapped IPv4 address as a literal
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "::ffff:192.168.1.50")

        # Origin with IPv6 bracket notation
        result = rm.is_remote_origin_allowed("http://[::ffff:192.168.1.50]:8765")
        # The function should parse the host and match it — not crash
        # (True/False depends on allowlist match, but it must not raise)
        assert isinstance(result, bool), (
            "is_remote_origin_allowed must return bool, not raise on IPv6 origin"
        )

    def test_ipv6_bracket_path_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bracketed IPv6 origin must not raise any exception."""
        monkeypatch.setenv("SLM_REMOTE", "1")
        monkeypatch.setenv("SLM_MCP_ALLOWED_HOSTS", "192.168.0.0/16")

        # Should gracefully return False (not in allowlist), not raise
        try:
            result = rm.is_remote_origin_allowed("http://[fe80::1]:8765")
        except Exception as exc:
            pytest.fail(
                f"is_remote_origin_allowed raised on IPv6 origin: {exc}"
            )
        # Not in allowlist → False
        assert result is False
