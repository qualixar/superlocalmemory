"""Phase 6b-2 — MeshAdvertiser: opt-in mDNS advertising tests.

All tests MOCK zeroconf.Zeroconf entirely — no real multicast sockets
are created.  Serial-safe (no pytest-xdist; mDNS is xdist-unsafe).

Environment variable controlling advertising: SLM_MESH_ADVERTISE
Truthy values: 1, on, true, yes  (case-insensitive)
Default (unset / anything else): advertising DISABLED — backward-compat.
"""

from __future__ import annotations

import threading
from socket import inet_ntoa
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_advertiser(
    port: int = 8765,
    node_id: str = "testhost",
    properties: dict | None = None,
    injected_ip: str | None = None,
):
    """Return a fresh MeshAdvertiser with zeroconf mocked out."""
    from superlocalmemory.mesh.discovery import MeshAdvertiser

    return MeshAdvertiser(
        service_port=port,
        node_id=node_id,
        properties=properties or {},
        _injected_ip=injected_ip,
    )


# ---------------------------------------------------------------------------
# Test: backward compatibility — advertising DEFAULT OFF
# ---------------------------------------------------------------------------

class TestDefaultOff:
    """SLM_MESH_ADVERTISE unset → start() must be a no-op (BC proof)."""

    def test_start_noop_when_env_unset(self, monkeypatch):
        """register_service MUST NOT be called when env var is unset."""
        monkeypatch.delenv("SLM_MESH_ADVERTISE", raising=False)

        mock_zc_cls = MagicMock()
        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser()
            adv.start()

        mock_zc_cls.assert_not_called()
        assert adv.is_advertising is False

    def test_start_noop_when_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "")

        mock_zc_cls = MagicMock()
        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser()
            adv.start()

        mock_zc_cls.assert_not_called()
        assert adv.is_advertising is False

    def test_start_noop_when_env_falsy(self, monkeypatch):
        """Values not in the truthy set must keep advertising disabled."""
        for val in ("0", "false", "off", "no", "disabled"):
            monkeypatch.setenv("SLM_MESH_ADVERTISE", val)

            mock_zc_cls = MagicMock()
            with patch(
                "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
            ), patch(
                "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
            ):
                adv = _make_advertiser()
                adv.start()

            assert not mock_zc_cls.called, f"unexpected call for value={val!r}"


# ---------------------------------------------------------------------------
# Test: opt-in advertising — env var set
# ---------------------------------------------------------------------------

class TestOptInAdvertising:
    """SLM_MESH_ADVERTISE=on → register_service called with correct params."""

    def _start_with_env(self, monkeypatch, env_value: str, port: int = 8765):
        """Enable advertising, return (advertiser, mock_zc_instance)."""
        monkeypatch.setenv("SLM_MESH_ADVERTISE", env_value)

        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(port=port, injected_ip="192.168.1.10")
            adv.start()

        return adv, mock_zc_instance

    def test_register_called_once(self, monkeypatch):
        """register_service must be called exactly once on start()."""
        adv, mock_zc = self._start_with_env(monkeypatch, "on")

        mock_zc.register_service.assert_called_once()
        assert adv.is_advertising is True

    def test_service_type_matches_browse_type(self, monkeypatch):
        """ServiceInfo type_ must match the service type RemoteSyncClient browses."""
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        captured_info = {}

        def fake_register(info):
            captured_info["info"] = info

        mock_zc_instance = MagicMock()
        mock_zc_instance.register_service.side_effect = fake_register
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        from superlocalmemory.mesh.discovery import _SERVICE_TYPE
        from zeroconf import ServiceInfo  # real class for construction

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(port=8765, injected_ip="192.168.1.10")
            adv.start()

        info = captured_info["info"]
        assert info.type == _SERVICE_TYPE, (
            f"ServiceInfo.type {info.type!r} != expected {_SERVICE_TYPE!r}"
        )
        assert info.port == 8765

    def test_correct_port_in_service_info(self, monkeypatch):
        """Port stored in ServiceInfo must match the configured daemon port."""
        custom_port = 9999
        _, mock_zc = self._start_with_env(monkeypatch, "true", port=custom_port)

        _call = mock_zc.register_service.call_args
        info = _call[0][0]  # first positional arg to register_service
        assert info.port == custom_port

    def test_instance_name_includes_port(self, monkeypatch):
        """Instance name must include port to avoid collision on shared host."""
        _, mock_zc = self._start_with_env(monkeypatch, "yes", port=8765)

        _call = mock_zc.register_service.call_args
        info = _call[0][0]
        assert "8765" in info.name, (
            f"Port not found in instance name {info.name!r} — "
            "two daemons on the same host will collide"
        )

    def test_truthy_values_case_insensitive(self, monkeypatch):
        """Truthy env values must be accepted case-insensitively."""
        for val in ("1", "on", "ON", "On", "true", "TRUE", "yes", "YES"):
            monkeypatch.setenv("SLM_MESH_ADVERTISE", val)

            mock_zc_instance = MagicMock()
            mock_zc_cls = MagicMock(return_value=mock_zc_instance)

            with patch(
                "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
            ), patch(
                "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
            ):
                adv = _make_advertiser(injected_ip="10.0.0.1")
                adv.start()

            assert adv.is_advertising is True, f"Expected advertising for value={val!r}"

    def test_address_in_service_info(self, monkeypatch):
        """Injected IP must appear in ServiceInfo.addresses as packed bytes."""
        from socket import inet_aton

        _, mock_zc = self._start_with_env(monkeypatch, "1")

        _call = mock_zc.register_service.call_args
        info = _call[0][0]
        packed = inet_aton("192.168.1.10")
        assert packed in info.addresses, (
            f"Expected {packed!r} in addresses {info.addresses!r}"
        )


# ---------------------------------------------------------------------------
# Test: stop() behavior
# ---------------------------------------------------------------------------

class TestStop:
    """stop() must unregister + close; idempotent on double-call."""

    def _started_advertiser(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(injected_ip="10.0.0.1")
            adv.start()

        return adv, mock_zc_instance

    def test_stop_calls_unregister_then_close(self, monkeypatch):
        adv, mock_zc = self._started_advertiser(monkeypatch)

        adv.stop()

        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()
        assert adv.is_advertising is False

    def test_stop_is_idempotent(self, monkeypatch):
        """Double stop must not raise and must not double-close."""
        adv, mock_zc = self._started_advertiser(monkeypatch)

        adv.stop()
        adv.stop()  # second call must be a no-op

        mock_zc.unregister_service.assert_called_once()
        mock_zc.close.assert_called_once()

    def test_stop_before_start_is_safe(self, monkeypatch):
        """stop() on a never-started advertiser must not raise."""
        monkeypatch.delenv("SLM_MESH_ADVERTISE", raising=False)

        with patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser()
            adv.stop()  # must not raise

        assert adv.is_advertising is False


# ---------------------------------------------------------------------------
# Test: ZEROCONF_AVAILABLE = False (zeroconf not installed)
# ---------------------------------------------------------------------------

class TestZeroconfUnavailable:
    """All methods are no-ops when zeroconf is not installed."""

    def test_start_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        with patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", False
        ), patch(
            "superlocalmemory.mesh.discovery.Zeroconf", None
        ):
            adv = _make_advertiser()
            adv.start()  # must not raise

        assert adv.is_advertising is False

    def test_stop_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        with patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", False
        ), patch(
            "superlocalmemory.mesh.discovery.Zeroconf", None
        ):
            adv = _make_advertiser()
            adv.stop()  # must not raise

        assert adv.is_advertising is False


# ---------------------------------------------------------------------------
# Test: fail-soft — register_service raises
# ---------------------------------------------------------------------------

class TestFailSoft:
    """register_service raising must not propagate — daemon startup must not break."""

    def test_register_raises_does_not_propagate(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "on")

        mock_zc_instance = MagicMock()
        mock_zc_instance.register_service.side_effect = OSError("mDNS unavailable")
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(injected_ip="10.0.0.1")
            adv.start()  # must NOT raise

        assert adv.is_advertising is False, "Failed registration must leave advertiser not-advertising"
        # Audit P1: the Zeroconf instance opened before register_service failed
        # MUST be closed so its multicast sockets + threads don't leak.
        mock_zc_instance.close.assert_called_once()

    def test_unregister_raises_on_stop_does_not_propagate(self, monkeypatch):
        """stop() must not propagate even if unregister_service raises."""
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(injected_ip="10.0.0.1")
            adv.start()

        # Now make unregister raise
        mock_zc_instance.unregister_service.side_effect = RuntimeError("network gone")

        adv.stop()  # must NOT raise — daemon graceful shutdown must continue
        # Audit P1: close() MUST still run even though unregister_service raised,
        # otherwise the Zeroconf sockets/threads leak.
        mock_zc_instance.close.assert_called_once()


# ---------------------------------------------------------------------------
# Test: IP resolution
# ---------------------------------------------------------------------------

class TestIpResolution:
    """IP fallback to 127.0.0.1 when no non-loopback address is found."""

    def test_loopback_fallback_when_injected(self, monkeypatch):
        """Inject 127.0.0.1 directly and verify it appears in ServiceInfo."""
        from socket import inet_aton

        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(injected_ip="127.0.0.1")
            adv.start()

        _call = mock_zc_instance.register_service.call_args
        info = _call[0][0]
        assert inet_aton("127.0.0.1") in info.addresses

    def test_resolve_fallback_when_socket_fails(self, monkeypatch):
        """_resolve_advertise_ip falls back to 127.0.0.1 when socket connect fails."""
        import socket

        from superlocalmemory.mesh.discovery import _resolve_advertise_ip

        original_socket = socket.socket

        class _BadSocket:
            def __init__(self, *a, **kw):
                pass

            def connect(self, addr):
                raise OSError("unreachable")

            def getsockname(self):
                return ("127.0.0.1", 0)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        with patch("superlocalmemory.mesh.discovery.socket.socket", _BadSocket):
            ip = _resolve_advertise_ip(injected_ip=None)

        assert ip == "127.0.0.1"


# ---------------------------------------------------------------------------
# Test: node_id appears in properties
# ---------------------------------------------------------------------------

class TestProperties:
    """node_id must appear in the broadcasted TXT properties."""

    def test_node_id_in_txt_properties(self, monkeypatch):
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")

        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)

        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(node_id="myhost", injected_ip="10.0.0.1")
            adv.start()

        _call = mock_zc_instance.register_service.call_args
        info = _call[0][0]
        # Properties are bytes-keyed in ServiceInfo
        props = info.properties
        assert b"node_id" in props, f"node_id missing from TXT record: {props}"
        assert props[b"node_id"] == b"myhost"

    def test_txt_properties_carry_no_secret(self, monkeypatch):
        """TXT record must never leak a secret (broadcast in plaintext on LAN)."""
        monkeypatch.setenv("SLM_MESH_ADVERTISE", "1")
        mock_zc_instance = MagicMock()
        mock_zc_cls = MagicMock(return_value=mock_zc_instance)
        with patch(
            "superlocalmemory.mesh.discovery.Zeroconf", mock_zc_cls
        ), patch(
            "superlocalmemory.mesh.discovery.ZEROCONF_AVAILABLE", True
        ):
            adv = _make_advertiser(node_id="myhost", injected_ip="10.0.0.1")
            adv.start()

        info = mock_zc_instance.register_service.call_args[0][0]
        blob = b"|".join(list(info.properties.keys()) + list(info.properties.values()))
        for needle in (b"secret", b"token", b"authorization", b"password"):
            assert needle not in blob.lower(), f"possible secret in TXT: {needle!r}"


# ---------------------------------------------------------------------------
# Test: cross-file service-type invariant (advertise type == browse type)
# ---------------------------------------------------------------------------

def test_service_type_matches_remote_sync_browse_literal():
    """Audit P2: the type discovery ADVERTISES must equal the type
    remote_sync BROWSES, or peers silently never discover each other.
    remote_sync uses a string literal (not a shared constant), so assert the
    exact literal appears in its source.
    """
    import inspect
    from superlocalmemory.mesh import remote_sync
    from superlocalmemory.mesh.discovery import _SERVICE_TYPE

    assert _SERVICE_TYPE == "_slm-mesh._tcp.local."
    src = inspect.getsource(remote_sync)
    assert _SERVICE_TYPE in src, (
        f"discovery advertises {_SERVICE_TYPE!r} but remote_sync does not browse "
        "that exact type — advertise/browse have diverged"
    )
