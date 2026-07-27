# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Unit tests for the centralized is_loopback helper (Workstream B / issue #90).

These tests MUST fail before loopback.py is created (ImportError → RED),
and pass after implementation (GREEN).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# Parametrized unit tests for is_loopback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host,expected", [
    # ---- Standard IPv4 loopback ----
    ("127.0.0.1",           True),
    ("127.0.0.2",           True),   # full 127.0.0.0/8 range is loopback per RFC 5735
    ("127.255.255.255",     True),   # edge of /8 range

    # ---- IPv6 loopback ----
    ("::1",                 True),

    # ---- IPv4-mapped IPv6 loopback — the #90 case ----
    ("::ffff:127.0.0.1",    True),   # primary bug case
    ("::ffff:127.0.0.2",    True),   # other /8 addresses via mapped form
    ("::ffff:7f00:1",       True),   # hex form of ::ffff:127.0.0.1

    # ---- Hostname alias ----
    ("localhost",           True),
    ("LOCALHOST",           True),   # case-insensitive

    # ---- Negative: SEC-L-02 — empty/missing host must never be trusted ----
    ("",                    False),

    # ---- Negative: private IPs are NOT loopback ----
    ("192.168.1.1",         False),
    ("10.0.0.1",            False),
    ("172.16.0.1",          False),

    # ---- Negative: public IPs ----
    ("8.8.8.8",             False),
    ("1.1.1.1",             False),

    # ---- Negative: IPv4-mapped private — MUST NOT be accepted as loopback ----
    ("::ffff:192.168.1.1",  False),  # private, not loopback
    ("::ffff:8.8.8.8",      False),  # public, not loopback

    # ---- Negative: special values that must NOT bypass ----
    ("testclient",          False),  # test bypass must NOT flow through is_loopback
    ("0.0.0.0",             False),  # bind address, not a peer address

    # ---- Negative: bracketed IPv6 form is not a valid ip_address input ----
    ("[::1]",               False),
])
def test_is_loopback(host: str, expected: bool) -> None:
    """Parametrized is_loopback correctness check."""
    from superlocalmemory.server.loopback import is_loopback

    result = is_loopback(host)
    assert result == expected, (
        f"is_loopback({host!r}) returned {result!r}, expected {expected!r}"
    )


def test_is_loopback_rejects_none() -> None:
    """None input must not raise; returns False per SEC-L-02."""
    from superlocalmemory.server.loopback import is_loopback

    # None is not a str — should return False without raising
    assert is_loopback(None) is False  # type: ignore[arg-type]


def test_is_loopback_normalizes_mapped_ipv4_when_stdlib_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapped IPv4 loopback remains local on Python versions that misreport it.

    Some supported CPython builds report ``False`` for the IPv6 address's
    ``is_loopback`` property even though its embedded IPv4 address is in
    127.0.0.0/8.  The application contract, rather than that implementation
    detail, decides whether a peer is local.
    """
    from superlocalmemory.server import loopback as lb

    mapped = SimpleNamespace(
        is_loopback=False,
        ipv4_mapped=SimpleNamespace(is_loopback=True),
    )
    monkeypatch.setattr(lb._ipa, "ip_address", lambda _host: mapped)

    assert lb.is_loopback("::ffff:127.0.0.1") is True


def test_module_exports_loopback_predicate() -> None:
    """The module exports a callable loopback predicate without import-time guards."""
    from superlocalmemory.server import loopback as lb

    assert hasattr(lb, "is_loopback"), "is_loopback must be exported"
    assert callable(lb.is_loopback), "is_loopback must be callable"
