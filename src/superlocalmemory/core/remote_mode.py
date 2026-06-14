# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Distributed / LAN deployment mode — the single ``SLM_REMOTE`` switch.

SuperLocalMemory historically assumes every dashboard browser, MCP client,
and API caller lives on ``127.0.0.1``. That assumption breaks three things
for users who deploy SLM on a server and reach it across a LAN (issue #39):

  1. ``/internal/token`` refuses any non-loopback client → Brain page can't
     fetch the install token → "Couldn't load Brain".
  2. The MCP Streamable-HTTP transport is **stateful** — every call must
     replay the ``Mcp-Session-Id`` from the ``initialize`` handshake. A
     gateway/hub that forwards a tool call without replaying it gets
     ``-32600 Session not found``.
  3. Dashboard CSRF origin checks only accept loopback origins.

``SLM_REMOTE=1`` flips all three assumptions at once, **default OFF** so the
loopback-only security posture is unchanged for the 99% local case. LAN
access is still gated by an explicit IP allowlist (``SLM_MCP_ALLOWED_HOSTS``)
— remote mode alone does not throw the doors open.

Granular overrides (each implied by ``SLM_REMOTE=1`` but usable alone):
  * ``SLM_MCP_STATELESS=1`` — stateless MCP transport only (gateway fix),
    without opening the dashboard token endpoint.

Security note (WORSTCASE): stateless MCP drops per-session isolation, and
serving the install token to a LAN host lets any allowlisted machine read
the brain. Keep the allowlist specific (never blanket ``*`` unless the
network is fully trusted) — see ``docs/distributed-deployment.md``.
"""

from __future__ import annotations

import ipaddress
import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _is_truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


def is_remote_mode() -> bool:
    """True iff ``SLM_REMOTE`` opts this daemon into LAN/distributed mode."""
    return _is_truthy(os.environ.get("SLM_REMOTE"))


def mcp_stateless() -> bool:
    """True iff the MCP transport should run stateless (no session id required).

    Enabled by ``SLM_REMOTE=1`` (umbrella) or ``SLM_MCP_STATELESS=1`` (granular).
    Stateless mode lets any gateway/hub forward ``tools/call`` without replaying
    the ``Mcp-Session-Id`` handshake — the fix for issue #39 Issue 3.
    """
    return is_remote_mode() or _is_truthy(os.environ.get("SLM_MCP_STATELESS"))


def _allowlist_entries() -> list[str]:
    """Trusted-client allowlist, from ``SLM_MCP_ALLOWED_HOSTS``.

    Reuses the existing LAN allowlist the user already sets for MCP DNS-rebinding
    protection so there is ONE place to configure trusted hosts. Entries are
    comma-separated and may be: ``*`` (any), an exact IP, a CIDR block
    (``192.168.1.0/24``), or a prefix wildcard (``192.168.*``). A trailing
    ``:port`` / ``:*`` (host-header style) is ignored for client-IP matching.
    """
    raw = os.environ.get("SLM_MCP_ALLOWED_HOSTS", "").strip()
    return [e.strip() for e in raw.split(",") if e.strip()]


def _strip_port(entry: str) -> str:
    """Drop a trailing ``:port`` / ``:*`` host-header suffix.

    Handles plain ``host[:port]`` and CIDR ``a.b.c.d/n[:port]`` (v3.6.12 lan-1:
    a CIDR written with a host-header port suffix used to fail ip_network() and
    silently deny ALL clients). Bracketless IPv6 literals (≥2 colons, no '/')
    are left untouched.
    """
    e = entry.strip()
    if "/" in e:
        # CIDR — strip anything after the network prefix (a stray :port/:*)
        return e.partition(":")[0]
    if e.count(":") == 1:  # host:port or host:* (IPv4 / hostname)
        return e.split(":", 1)[0]
    return e


def _host_matches(entry: str, client_host: str, client_ip) -> bool:
    host = _strip_port(entry).strip()
    if not host:
        return False
    if host == "*":
        return True
    if "/" in host and client_ip is not None:
        try:
            return client_ip in ipaddress.ip_network(host, strict=False)
        except ValueError:
            return False
    if host.endswith("*"):
        # STRING prefix match (not CIDR). client_host is always the numeric
        # socket peer IP (never a resolvable hostname), and a dotted prefix like
        # "192.168." rejects "192.1680.x". Prefer CIDR (192.168.0.0/16) for
        # unambiguous network matching; wildcards are a convenience.
        return client_host.startswith(host[:-1])
    return host == client_host


def is_lan_client_allowed(client_host: str) -> bool:
    """True iff remote mode is ON and ``client_host`` is in the trusted allowlist.

    Loopback is handled separately by callers — this governs *non*-loopback LAN
    clients only. Returns False whenever remote mode is off or the allowlist is
    empty, so the default posture stays loopback-only.
    """
    if not is_remote_mode() or not client_host:
        return False
    entries = _allowlist_entries()
    if not entries:
        return False
    try:
        client_ip = ipaddress.ip_address(client_host)
    except ValueError:
        client_ip = None
    return any(_host_matches(e, client_host, client_ip) for e in entries)


def is_remote_origin_allowed(origin: str) -> bool:
    """True iff remote mode is ON and ``origin``'s host is in the allowlist.

    ``origin`` is a full URL (``http://192.168.50.144:8765``). Empty origin is
    not this function's concern (loopback callers handle that). Used to relax
    the dashboard CSRF origin guard for trusted LAN dashboards.
    """
    if not is_remote_mode() or not origin:
        return False
    # Extract host from scheme://host[:port]
    rest = origin.split("://", 1)[-1]
    host = rest.split("/", 1)[0]
    # Strip a trailing :port (IPv4/hostname); leave bracketed IPv6 alone.
    if host.startswith("["):
        host = host.split("]", 1)[0].lstrip("[")
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return is_lan_client_allowed(host)


def _env_int(name: str, default: int) -> int:
    """Read a positive int from env, falling back to ``default`` on any error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def rate_limit_config() -> tuple[int, int, int]:
    """(write_max, read_max, window_seconds) for the dashboard rate limiter.

    Issue #40 Issue 3: the limiter was hardcoded (30 writes / 120 reads per 60s)
    with no way to raise it for distributed/LAN debugging, so a remote browser
    that retried a failing Brain load hit ``429 Too Many Requests``. These are
    now tunable via ``SLM_RATE_LIMIT_WRITE`` / ``SLM_RATE_LIMIT_READ`` /
    ``SLM_RATE_LIMIT_WINDOW`` (defaults unchanged for the local case).
    """
    write_max = _env_int("SLM_RATE_LIMIT_WRITE", 30)
    read_max = _env_int("SLM_RATE_LIMIT_READ", 120)
    window = _env_int("SLM_RATE_LIMIT_WINDOW", 60)
    return write_max, read_max, window


def is_rate_limit_exempt(client_host: str) -> bool:
    """True iff ``client_host`` should bypass the dashboard rate limiter.

    Loopback is always exempt (the dashboard polls itself rapidly). In remote
    mode, an allowlisted LAN client is the user's own remote browser doing the
    same rapid reads, so it is exempt too — otherwise normal dashboard polling
    trips the limiter (issue #40 Issue 3).
    """
    if client_host in ("127.0.0.1", "::1", "localhost"):
        return True
    return is_lan_client_allowed(client_host)


__all__ = (
    "is_remote_mode",
    "mcp_stateless",
    "is_lan_client_allowed",
    "is_remote_origin_allowed",
    "rate_limit_config",
    "is_rate_limit_exempt",
)
