"""Exact browser-origin validation for the local dashboard."""

from __future__ import annotations

from urllib.parse import urlsplit


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def origin_is_loopback(origin: str) -> bool:
    """Return whether an Origin is absent or an exact HTTP(S) loopback URL."""
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        # Accessing port validates malformed/non-numeric port values.
        _ = parsed.port
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.hostname.lower() in _LOOPBACK_HOSTS
        and parsed.username is None
        and parsed.password is None
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def origin_is_daemon(origin: str, *, port: int) -> bool:
    """Return whether ``origin`` is one of this daemon's loopback aliases.

    A loopback host alone is not a browser trust boundary: another local web
    server can run on a different port.  Credentialless dashboard writes are
    therefore limited to the port owned by this daemon.  Authenticated local
    integrations are handled separately by the write-identity boundary.
    """
    if not origin_is_loopback(origin):
        return False
    try:
        parsed = urlsplit(origin)
        return parsed.port == port
    except (TypeError, ValueError):
        return False


__all__ = ("origin_is_daemon", "origin_is_loopback")
