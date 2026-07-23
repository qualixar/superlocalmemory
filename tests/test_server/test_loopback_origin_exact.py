"""Loopback origin checks must resist suffix-domain and userinfo tricks."""

from __future__ import annotations

import pytest

from superlocalmemory.server.origin import origin_is_daemon, origin_is_loopback


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "http://localhost",
        "http://localhost:8765",
        "https://127.0.0.1:8765",
        "http://[::1]:8765",
    ],
)
def test_exact_loopback_origins_are_allowed(origin: str) -> None:
    assert origin_is_loopback(origin) is True


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost.evil.example",
        "http://127.0.0.1.attacker.example",
        "http://localhost@evil.example",
        "ftp://localhost",
        "http://localhost:bad",
        "http://localhost/path",
        "http://localhost?redirect=evil",
    ],
)
def test_origin_parser_rejects_prefix_and_parser_confusion(origin: str) -> None:
    assert origin_is_loopback(origin) is False


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8765",
        "http://127.0.0.1:8765",
        "https://[::1]:8765",
    ],
)
def test_only_the_daemon_port_is_a_credentialless_browser_origin(origin: str) -> None:
    assert origin_is_daemon(origin, port=8765) is True


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8417",
        "http://127.0.0.1:8767",
        "http://localhost",
    ],
)
def test_other_loopback_ports_are_not_the_daemon_origin(origin: str) -> None:
    assert origin_is_daemon(origin, port=8765) is False
