# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Issue #112 — private-LAN endpoint trust: boundary and security cases.

These tests prove the EXACT BOUNDARIES of the trust mechanism and document the
security properties of each decision:

  - IPv6 ULA (fc00::/7) accepted
  - IPv6 link-local (fe80::) accepted
  - IPv4 link-local (169.254.x.x) accepted
  - IPv4-mapped IPv6 of private addresses accepted
  - 172.15.x.x NOT private (boundary below 172.16.0.0/12)
  - 172.32.x.x NOT private (boundary above 172.31.255.255)
  - Bare hostnames REFUSED (DNS not a trust boundary)
  - CGNAT 100.64.x.x REFUSED (ISP-shared space)
  - trust_plain_http_lan=False refuses all private-LAN addresses
  - Public addresses always refused for plain HTTP
"""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.remote_reranker import (
    _is_private_lan_host,
    validate_remote_reranker_config,
)

# ---------------------------------------------------------------------------
# Unit tests for _is_private_lan_host
# ---------------------------------------------------------------------------

class TestIsPrivateLanHost:
    """Unit-test the host classifier in isolation (no URL parsing)."""

    # --- accepted ranges ---

    @pytest.mark.parametrize("host", [
        "10.0.0.1", "10.255.255.255", "10.128.0.1",
    ])
    def test_rfc1918_10_slash_8(self, host: str) -> None:
        assert _is_private_lan_host(host), f"10.0.0.0/8 should be private: {host}"

    @pytest.mark.parametrize("host", [
        "172.16.0.1", "172.16.4.2", "172.31.0.1", "172.31.255.255",
    ])
    def test_rfc1918_172_16_slash_12(self, host: str) -> None:
        assert _is_private_lan_host(host), f"172.16.0.0/12 should be private: {host}"

    @pytest.mark.parametrize("host", [
        "192.168.0.1", "192.168.1.50", "192.168.255.255",
    ])
    def test_rfc1918_192_168_slash_16(self, host: str) -> None:
        assert _is_private_lan_host(host), f"192.168.0.0/16 should be private: {host}"

    @pytest.mark.parametrize("host", [
        "169.254.0.1", "169.254.255.255", "169.254.1.1",
    ])
    def test_ipv4_link_local(self, host: str) -> None:
        assert _is_private_lan_host(host), f"169.254.0.0/16 link-local should be private: {host}"

    @pytest.mark.parametrize("host", [
        "fc00::1", "fc00::ffff", "fd00::1", "fd12:3456:789a::1",
    ])
    def test_ipv6_ula(self, host: str) -> None:
        """fc00::/7 (ULA) — includes the commonly used fd00::/8 sub-range."""
        assert _is_private_lan_host(host), f"ULA IPv6 should be private: {host}"

    @pytest.mark.parametrize("host", [
        "fe80::1", "fe80::1%eth0", "fe80::a1b2:c3d4",
    ])
    def test_ipv6_link_local(self, host: str) -> None:
        """fe80::/10 link-local — common on Docker bridge and LXC networks."""
        assert _is_private_lan_host(host), f"IPv6 link-local should be private: {host}"

    @pytest.mark.parametrize("host", [
        "::ffff:192.168.1.1",   # RFC1918 in IPv4-mapped form
        "::ffff:10.0.0.1",
        "::ffff:172.16.0.1",
    ])
    def test_ipv4_mapped_ipv6_private(self, host: str) -> None:
        """IPv4-mapped IPv6 addresses wrapping RFC1918 must be accepted."""
        assert _is_private_lan_host(host), f"IPv4-mapped private should be private: {host}"

    # --- boundary: 172.16/12 edge ---

    def test_172_15_is_NOT_private(self) -> None:
        """172.15.255.255 is just below 172.16.0.0/12 and is public."""
        assert not _is_private_lan_host("172.15.255.255"), (
            "172.15.255.255 is OUTSIDE 172.16.0.0/12 and must not be trusted"
        )

    def test_172_32_is_NOT_private(self) -> None:
        """172.32.0.0 is just above 172.31.255.255 (top of 172.16.0.0/12)."""
        assert not _is_private_lan_host("172.32.0.0"), (
            "172.32.0.0 is OUTSIDE 172.16.0.0/12 and must not be trusted"
        )

    def test_172_16_0_0_boundary_is_private(self) -> None:
        assert _is_private_lan_host("172.16.0.0")

    def test_172_31_255_255_boundary_is_private(self) -> None:
        assert _is_private_lan_host("172.31.255.255")

    # --- rejected: not private ---

    @pytest.mark.parametrize("host", [
        "8.8.8.8", "1.1.1.1", "9.9.9.9", "5.5.5.5",
    ])
    def test_public_ipv4_rejected(self, host: str) -> None:
        # NOTE: RFC 5737 documentation addresses (203.0.113.x, 198.51.100.x)
        # are classified is_private=True in Python 3.11+ — use only addresses
        # that are globally routable across all Python versions.
        assert not _is_private_lan_host(host), f"Public IP must not be private: {host}"

    def test_cgnat_rejected(self) -> None:
        """100.64.0.0/10 is CGNAT — shared with ISPs, not operator-controlled."""
        assert not _is_private_lan_host("100.64.0.1"), (
            "CGNAT (100.64.0.0/10) is ISP-shared space and must not be trusted"
        )

    @pytest.mark.parametrize("host", [
        "my-reranker.lan",
        "reranker.local",
        "llm.internal",
        "host.example.com",
    ])
    def test_bare_hostnames_rejected(self, host: str) -> None:
        """Bare hostnames are never trusted even if they resolve to private IPs.

        DNS can be poisoned, updated, or redirected. A hostname that currently
        resolves to 192.168.x.x can be changed to point at a public IP without
        updating the SLM configuration. Only numeric addresses are provably
        bound to a private range at configuration time.
        """
        assert not _is_private_lan_host(host), (
            f"Bare hostname {host!r} must not be trusted — DNS is not a "
            f"trust boundary"
        )

    @pytest.mark.parametrize("host", [
        "::ffff:8.8.8.8",      # IPv4-mapped Google DNS (globally routable)
        "::ffff:1.1.1.1",      # IPv4-mapped Cloudflare DNS (globally routable)
    ])
    def test_ipv4_mapped_ipv6_public_rejected(self, host: str) -> None:
        """IPv4-mapped IPv6 wrapping globally-routable addresses must not pass.

        NOTE: RFC 5737 documentation addresses (203.0.113.x, 198.51.100.x) are
        classified is_private=True in Python 3.11+ so we use confirmed
        globally-routable DNS resolver IPs here.
        """
        assert not _is_private_lan_host(host), (
            f"IPv4-mapped public address {host!r} must not be treated as private"
        )


# ---------------------------------------------------------------------------
# Integration: validate_remote_reranker_config with the full URL
# ---------------------------------------------------------------------------

class TestValidateLanTrustIntegration:
    """End-to-end: the function the reporter called."""

    # --- default (trust_plain_http_lan=True) ---

    @pytest.mark.parametrize("url", [
        "http://10.0.0.7:8041/v1/rerank",
        "http://172.16.4.2:8041/v1/rerank",
        "http://192.168.1.50:8041/v1/rerank",
        "http://169.254.1.1:8041/v1/rerank",
        "http://[fc00::1]:8041/v1/rerank",
        "http://[fe80::1]:8041/v1/rerank",
        "http://[::ffff:192.168.1.1]:8041/v1/rerank",
    ])
    def test_private_addresses_accepted_by_default(self, url: str) -> None:
        assert validate_remote_reranker_config("openai", url) is None, (
            f"Private-LAN URL {url} should be accepted with default trust"
        )

    @pytest.mark.parametrize("url", [
        "http://172.15.255.255:8041/v1/rerank",
        "http://172.32.0.0:8041/v1/rerank",
    ])
    def test_non_private_boundary_addresses_refused(self, url: str) -> None:
        """172.15.x and 172.32.x fall outside RFC1918 and must be refused."""
        err = validate_remote_reranker_config("openai", url)
        assert err, f"{url} is outside RFC1918 and must require HTTPS"

    @pytest.mark.parametrize("url", [
        "http://8.8.8.8:8041/v1/rerank",
        "http://1.1.1.1:8041/v1/rerank",
        "http://9.9.9.9:8041/v1/rerank",
    ])
    def test_public_addresses_always_refused(self, url: str) -> None:
        err = validate_remote_reranker_config("openai", url)
        assert err, f"Public URL {url} must always require HTTPS"

    @pytest.mark.parametrize("url", [
        "http://my-reranker.lan:8041/v1/rerank",
        "http://llm.internal:8041/v1/rerank",
    ])
    def test_bare_hostnames_refused_even_with_trust_enabled(self, url: str) -> None:
        """Bare hostnames are refused for plain HTTP even when trust is on.

        DNS is not a trust boundary — a hostname that resolves to a private IP
        today can be changed to resolve to a public IP tomorrow without any
        change to the SLM configuration.
        """
        err = validate_remote_reranker_config("openai", url)
        assert err, (
            f"Bare hostname {url} must be refused for plain HTTP — "
            "use a numeric IP address instead"
        )

    # --- explicit opt-out (trust_plain_http_lan=False) ---

    @pytest.mark.parametrize("url", [
        "http://192.168.1.50:8041/v1/rerank",
        "http://10.0.0.7:8041/v1/rerank",
        "http://172.16.4.2:8041/v1/rerank",
    ])
    def test_private_addresses_refused_when_trust_disabled(self, url: str) -> None:
        """Hardened-mode opt-out: private LAN also requires HTTPS."""
        err = validate_remote_reranker_config(
            "openai", url, trust_plain_http_lan=False,
        )
        assert err, (
            f"{url} should be refused when trust_plain_http_lan=False"
        )
        assert "trust_plain_http_lan" in err, (
            "Error message should name the controlling config key"
        )

    def test_loopback_allowed_even_when_trust_disabled(self) -> None:
        """Loopback is always allowed — the opt-out does not affect it."""
        assert validate_remote_reranker_config(
            "openai",
            "http://127.0.0.1:8041/v1/rerank",
            trust_plain_http_lan=False,
        ) is None

    def test_https_always_allowed_regardless_of_trust_flag(self) -> None:
        """HTTPS is always accepted — trust_plain_http_lan only gates plain HTTP."""
        for flag in (True, False):
            assert validate_remote_reranker_config(
                "openai",
                "https://192.168.1.50:8041/v1/rerank",
                trust_plain_http_lan=flag,
            ) is None
