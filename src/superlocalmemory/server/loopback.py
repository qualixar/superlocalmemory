# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Centralized loopback-address predicate for the SLM server.

Replaces every ``frozenset({"127.0.0.1", "::1", "localhost"})`` auth check
with a semantic helper that correctly handles IPv4-mapped IPv6 addresses
(``::ffff:127.0.0.1``) — the root cause of issue #90.

Issue #90 root cause
--------------------
When the daemon is started with ``SLM_DAEMON_HOST=0.0.0.0`` on a dual-stack
Linux host (common in LXC/Docker containers), the OS creates an IPv6 socket
that accepts IPv4 connections via the IPv4-mapped IPv6 address mechanism
(RFC 4291 §2.5.5.2). A client connecting to ``localhost`` on such a host
may have its peer address reported as ``::ffff:127.0.0.1`` by
uvicorn/Starlette. The literal set ``("127.0.0.1", "::1", "localhost")``
does not include this form, causing a spurious 403 for install-token and
uncredentialed-loopback callers.

``ipaddress.ip_address("::ffff:127.0.0.1").is_loopback`` already returns
``True`` in CPython. This module uses that fact.

SECURITY INVARIANTS (non-negotiable):
  - Empty/None host → False. SEC-L-02: a missing peer is never trusted.
  - No proxy header trust. Callers MUST pass ``request.client.host`` only,
    never X-Forwarded-For or any other spoofable header.
  - All 127.0.0.0/8 is loopback per RFC 5735 (includes 127.0.0.2, etc.).
  - ``"localhost"`` is accepted as a hostname alias. Callers that need the
    stricter no-hostname check (``prewarm_auth``) keep their own predicate.
  - ``"testclient"`` is NOT in the loopback set — that bypass must be wired
    explicitly alongside ``_TEST_ISOLATION_ALLOWED`` so it cannot appear in
    production paths.
  - ``"0.0.0.0"`` is NOT loopback — it is a bind address, not a peer address.
"""

from __future__ import annotations

import ipaddress as _ipa


def is_loopback(host: str) -> bool:
    """Return ``True`` iff ``host`` is a loopback address in any standard form.

    Handles:
    * ``"127.0.0.1"`` and the full 127.0.0.0/8 range (RFC 5735).
    * ``"::1"`` (IPv6 loopback).
    * ``"::ffff:127.0.0.1"`` and all ``::ffff:127.x.x.x`` (IPv4-mapped IPv6,
      fixes issue #90).
    * ``"localhost"`` (hostname alias, case-insensitive).

    Returns ``False`` for:
    * Empty string or ``None`` (SEC-L-02: missing peer is never trusted).
    * Any non-loopback IP (192.168.x.x, 10.x.x.x, public IPs, etc.).
    * ``"::ffff:192.168.x.x"`` and other IPv4-mapped non-loopback addresses.
    * ``"testclient"`` — callers that need the test-client exemption must
      wire it explicitly alongside ``_TEST_ISOLATION_ALLOWED``.
    * ``"0.0.0.0"`` — bind address, not a peer address.

    Args:
        host: The ``request.client.host`` string from an incoming HTTP
            request. Must be the TCP-observed peer address only.

    Returns:
        ``True`` if the host is any recognised loopback form; ``False``
        otherwise.
    """
    if not isinstance(host, str) or not host:
        return False  # SEC-L-02
    if host.lower() == "localhost":
        return True
    try:
        ip = _ipa.ip_address(host)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    # CPython's handling of IPv4-mapped IPv6 changed across supported
    # runtimes. Normalize through the embedded IPv4 address so a dual-stack
    # socket reporting ::ffff:127.x.x.x retains the equivalent IPv4 decision.
    mapped_ipv4 = getattr(ip, "ipv4_mapped", None)
    return bool(mapped_ipv4 is not None and mapped_ipv4.is_loopback)


__all__ = ("is_loopback",)
