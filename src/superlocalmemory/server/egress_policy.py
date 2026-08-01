"""SSRF-safe egress validation for outbound provider requests.

Pre-admission, stateless validation for outbound HTTP fetches (embedding test,
provider test, local model probe). It enforces three protections:

* Cloud-metadata endpoints are blocked unconditionally and after hostname
  normalization (trailing dot / case / IDNA), so ``metadata.google.internal.``
  cannot slip past — including on the loopback path.
* The full DNS answer set is validated (all A/AAAA records), not just the first
  resolved address, which defeats mixed public/private answers (DNS rebinding).
* DNS resolution failure fails closed (deny) for untrusted callers rather than
  deferring the decision to the HTTP client.

Trusted callers (the local dashboard on loopback, or an allowlisted LAN
dashboard) keep the latitude to probe local/LAN model endpoints; metadata
endpoints are denied for everyone.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class EgressVerdict(str, Enum):
    ALLOW = "allow"
    DENY_SCHEME = "deny_scheme"
    DENY_CREDENTIALS = "deny_credentials"
    DENY_FRAGMENT = "deny_fragment"
    DENY_HOST = "deny_host"
    DENY_METADATA = "deny_metadata"
    DENY_PRIVATE = "deny_private"
    DENY_DNS_FAILURE = "deny_dns_failure"
    DENY_MIXED_DNS = "deny_mixed_dns"


# Cloud metadata endpoints — always blocked regardless of caller trust.
METADATA_HOSTS: frozenset[str] = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata",
        "metadata.azure.internal",
        "169.254.170.2",  # AWS ECS task metadata
        "fd00:ec2::254",  # AWS IMDS over IPv6
    }
)

_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


@dataclass(frozen=True, slots=True)
class EgressPolicy:
    """Configurable egress constraints (safe defaults)."""

    allowed_schemes: frozenset[str] = _ALLOWED_SCHEMES
    allow_private_for_local_actor: bool = True
    allow_private_for_lan_actor: bool = True
    reject_credentials: bool = True
    reject_fragment: bool = True


@dataclass(frozen=True, slots=True)
class EgressActor:
    """Minimal trust descriptor for the caller.

    ``is_local``  — request originated from the loopback dashboard.
    ``is_lan``    — request originated from an allowlisted LAN dashboard
                    (remote mode ON *and* client IP in the allowlist).
    A caller with neither flag is an untrusted/remote caller.
    """

    is_local: bool = False
    is_lan: bool = False

    @property
    def is_trusted(self) -> bool:
        return self.is_local or self.is_lan


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    addresses: tuple[str, ...] = ()
    has_private: bool = False
    has_public: bool = False
    error: str = ""

    @property
    def is_mixed(self) -> bool:
        return self.has_private and self.has_public


@dataclass(frozen=True, slots=True)
class EgressResult:
    verdict: EgressVerdict
    resolved_ip: str = ""
    error: str = ""
    hostname: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is EgressVerdict.ALLOW


def _normalize_host(hostname: str) -> str:
    """Lowercase, strip a trailing dot, IDNA-encode non-ASCII hosts.

    IDNA encoding is only attempted for hosts containing non-ASCII characters,
    so IP literals and ordinary ASCII hostnames are left untouched (the ``idna``
    codec rejects some all-ASCII inputs).
    """
    host = hostname.strip().lower().rstrip(".")
    if host and any(ord(ch) > 127 for ch in host):
        try:
            host = host.encode("idna").decode("ascii")
        except (UnicodeError, UnicodeDecodeError):
            pass  # keep the raw host — classification below still applies
    return host


def _is_dangerous_ip(
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _coerce_ip(
    ip_str: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return None
    # Unwrap IPv4-mapped IPv6 (``::ffff:127.0.0.1``) before classification.
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        return addr.ipv4_mapped
    return addr


def resolve_and_validate_dns(
    hostname: str, policy: EgressPolicy = EgressPolicy()
) -> ResolvedTarget:
    """Resolve every A/AAAA record and classify each address.

    Fails closed: any resolution error, or an empty answer set, returns a
    ``ResolvedTarget`` carrying a non-empty ``error``.
    """
    try:
        infos = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except (socket.gaierror, OSError, UnicodeError) as exc:
        return ResolvedTarget(error=f"dns resolution failed: {exc}")

    addresses: list[str] = []
    has_private = False
    has_public = False
    for info in infos:
        sockaddr = info[4]
        addr = _coerce_ip(sockaddr[0])
        if addr is None:
            continue
        addresses.append(str(addr))
        if _is_dangerous_ip(addr):
            has_private = True
        else:
            has_public = True

    if not addresses:
        return ResolvedTarget(error="no addresses resolved")
    return ResolvedTarget(
        addresses=tuple(addresses),
        has_private=has_private,
        has_public=has_public,
    )


def validate_egress_url(
    url: str,
    actor: EgressActor = EgressActor(),
    policy: EgressPolicy = EgressPolicy(),
) -> EgressResult:
    """Full SSRF-safe validation of an outbound URL for a given caller."""
    parsed = urlparse(url)

    if parsed.scheme not in policy.allowed_schemes:
        return EgressResult(
            verdict=EgressVerdict.DENY_SCHEME, error=f"scheme {parsed.scheme!r}"
        )
    if policy.reject_credentials and (parsed.username or parsed.password):
        return EgressResult(verdict=EgressVerdict.DENY_CREDENTIALS)
    if policy.reject_fragment and parsed.fragment:
        return EgressResult(verdict=EgressVerdict.DENY_FRAGMENT)

    hostname = _normalize_host(parsed.hostname or "")
    if not hostname:
        return EgressResult(verdict=EgressVerdict.DENY_HOST, error="empty host")

    # Metadata block — unconditional, before any trust short-circuit or DNS
    # (defeats the trailing-dot / loopback-path bypass and DNS rebinding).
    if hostname in METADATA_HOSTS:
        return EgressResult(verdict=EgressVerdict.DENY_METADATA, hostname=hostname)
    literal = _coerce_ip(hostname)
    if literal is not None and str(literal) in METADATA_HOSTS:
        return EgressResult(verdict=EgressVerdict.DENY_METADATA, hostname=hostname)

    trusted = (
        actor.is_local and policy.allow_private_for_local_actor
    ) or (actor.is_lan and policy.allow_private_for_lan_actor)

    # Literal IP target: classify directly, no DNS needed.
    if literal is not None:
        if _is_dangerous_ip(literal) and not trusted:
            return EgressResult(
                verdict=EgressVerdict.DENY_PRIVATE,
                resolved_ip=str(literal),
                hostname=hostname,
            )
        return EgressResult(
            verdict=EgressVerdict.ALLOW, resolved_ip=str(literal), hostname=hostname
        )

    # Trusted callers may probe local/LAN endpoints by name without a forced
    # DNS lookup — preserving pre-V4 loopback/LAN dashboard behaviour. Metadata
    # was already denied above for everyone.
    if trusted:
        return EgressResult(verdict=EgressVerdict.ALLOW, hostname=hostname)

    # Untrusted caller: resolve the full answer set and fail closed on error.
    resolved = resolve_and_validate_dns(hostname, policy)
    if resolved.error:
        return EgressResult(
            verdict=EgressVerdict.DENY_DNS_FAILURE,
            error=resolved.error,
            hostname=hostname,
        )
    if resolved.has_private:
        verdict = (
            EgressVerdict.DENY_MIXED_DNS
            if resolved.is_mixed
            else EgressVerdict.DENY_PRIVATE
        )
        return EgressResult(verdict=verdict, hostname=hostname)

    return EgressResult(
        verdict=EgressVerdict.ALLOW,
        resolved_ip=resolved.addresses[0],
        hostname=hostname,
    )
