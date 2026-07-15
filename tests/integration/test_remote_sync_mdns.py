"""mDNS discovery must accept Zeroconf's packed IP address contract."""

from __future__ import annotations

import ipaddress
from types import SimpleNamespace
from unittest.mock import MagicMock


class _Zeroconf:
    def __init__(self, info):
        self._info = info

    def get_service_info(self, service_type, name):
        return self._info


def _client(monkeypatch):
    from superlocalmemory.mesh import remote_sync

    monkeypatch.setattr(remote_sync, "ZEROCONF_AVAILABLE", True)
    return remote_sync.RemoteSyncClient(MagicMock())


def test_mdns_accepts_packed_ipv4_address(monkeypatch) -> None:
    client = _client(monkeypatch)
    info = SimpleNamespace(
        addresses=[ipaddress.ip_address("192.0.2.8").packed],
        port=8877,
    )

    client.add_service(_Zeroconf(info), "_slm-mesh._tcp.local.", "peer")

    assert client._peer_url == "http://192.0.2.8:8877"


def test_mdns_formats_packed_ipv6_address_for_url(monkeypatch) -> None:
    client = _client(monkeypatch)
    info = SimpleNamespace(
        addresses=[ipaddress.ip_address("2001:db8::8").packed],
        port=8878,
    )

    client.add_service(_Zeroconf(info), "_slm-mesh._tcp.local.", "peer")

    assert client._peer_url == "http://[2001:db8::8]:8878"


def test_mdns_ignores_malformed_addresses(monkeypatch) -> None:
    client = _client(monkeypatch)
    info = SimpleNamespace(addresses=[b"not-an-ip"], port=8879)

    client.add_service(_Zeroconf(info), "_slm-mesh._tcp.local.", "peer")

    assert client._peer_url is None
