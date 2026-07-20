# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — Remote Sync Client.

HTTP-based synchronization with a remote SLM instance.
Populates broker._remote_peers from the remote /mesh/peers endpoint.
Proxies mesh_send to remote when the target peer lives on the remote machine.
Optional mDNS discovery via zeroconf.

Environment variables:
  SLM_MESH_PEER_URL: Full URL of remote SLM (e.g. http://192.168.1.100:8765)
  SLM_MESH_SHARED_SECRET: Shared auth secret for remote SLM
  SLM_MESH_DISCOVERY: 'on'|'off' (default 'on') — enable mDNS discovery
"""

from __future__ import annotations

import logging
import os
import threading
import time
import ipaddress
from typing import Any

import httpx

logger = logging.getLogger("superlocalmemory.mesh.remote_sync")


def _service_ip_addresses(info: Any) -> list[str]:
    """Return validated textual IPs from current and older Zeroconf APIs."""
    candidates: list[Any] = []
    parsed = getattr(info, "parsed_addresses", None)
    if callable(parsed):
        try:
            candidates.extend(parsed())
        except (OSError, TypeError, ValueError):
            pass
    candidates.extend(getattr(info, "addresses", None) or [])

    addresses: list[str] = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except (TypeError, ValueError):
            continue
        # An mDNS peer must name a routable endpoint, never a wildcard or
        # multicast destination.  Authentication is still enforced by mesh.
        if address.is_unspecified or address.is_multicast:
            continue
        rendered = str(address)
        if rendered not in addresses:
            addresses.append(rendered)
    return addresses


def _peer_url(host: str, port: int) -> str:
    """Format an IP literal safely for an HTTP authority."""
    address = ipaddress.ip_address(host)
    rendered = str(address)
    if address.version == 6:
        # RFC 6874 requires a percent sign in an IPv6 zone identifier to be
        # escaped when the literal appears inside a URI authority.
        rendered = rendered.replace("%", "%25")
        rendered = f"[{rendered}]"
    return f"http://{rendered}:{int(port)}"

# Optional zeroconf for mDNS discovery
try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None


class RemoteSyncClient:
    """HTTP-based sync client for multi-machine mesh coordination.

    Syncs remote peers from a peer SLM instance periodically.
    Proxies mesh_send to remote when target peer lives on remote machine.
    Optionally discovers remote SLM via mDNS.
    """

    def __init__(self, broker: Any) -> None:
        """Initialize sync client.

        Args:
            broker: Reference to MeshBroker instance
        """
        self._broker = broker
        self._peer_url: str | None = os.environ.get("SLM_MESH_PEER_URL") or None
        self._shared_secret: str | None = os.environ.get("SLM_MESH_SHARED_SECRET") or None
        self._discovery_enabled: bool = (
            os.environ.get("SLM_MESH_DISCOVERY", "on") != "off"
        )
        self._sync_thread: threading.Thread | None = None
        self._discovery_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._zeroconf: Zeroconf | None = None
        self._last_peers: dict[str, dict] = {}

    def start(self) -> None:
        """Start background sync and discovery threads."""
        if not self._peer_url and not self._discovery_enabled:
            logger.debug(
                "RemoteSyncClient: no peer URL and discovery disabled, skipping"
            )
            return

        # Start sync thread
        self._sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True, name="mesh-remote-sync"
        )
        self._sync_thread.start()

        # Start discovery thread if enabled and zeroconf available
        if self._discovery_enabled and ZEROCONF_AVAILABLE:
            self._discovery_thread = threading.Thread(
                target=self._discovery_loop, daemon=True, name="mesh-mdns-discovery"
            )
            self._discovery_thread.start()
            logger.info("RemoteSyncClient: mDNS discovery enabled")
        elif self._discovery_enabled and not ZEROCONF_AVAILABLE:
            logger.warning(
                "RemoteSyncClient: mDNS discovery requested but zeroconf not available"
            )

    def stop(self) -> None:
        """Stop background threads cleanly."""
        self._stop_event.set()
        if self._zeroconf:
            try:
                self._zeroconf.close()
            except Exception as e:
                logger.debug("RemoteSyncClient: error closing zeroconf: %s", e)
        # Wait for threads to finish (up to 2s)
        if self._sync_thread:
            self._sync_thread.join(timeout=2)
        if self._discovery_thread:
            self._discovery_thread.join(timeout=2)

    def _sync_loop(self) -> None:
        """Background thread: sync remote peers every 30s."""
        while not self._stop_event.is_set():
            try:
                if self._peer_url:
                    self._sync_peers_from_remote()
            except Exception as exc:
                logger.debug("RemoteSyncClient: sync error: %s", exc)

            # Wait 30s before next sync
            if self._stop_event.wait(30):
                break

    def _sync_peers_from_remote(self) -> None:
        """Fetch peers from remote /mesh/peers and update broker."""
        if not self._peer_url:
            return

        try:
            with httpx.Client(timeout=5) as client:
                headers = {}
                if self._shared_secret:
                    headers["Authorization"] = f"Bearer {self._shared_secret}"

                resp = client.get(
                    f"{self._peer_url}/mesh/peers", headers=headers, timeout=5
                )
                resp.raise_for_status()

                data = resp.json()
                remote_peers = data.get("peers", [])

                # Convert list to dict by peer_id
                current = {p.get("peer_id"): p for p in remote_peers}

                # Add/update peers
                for peer_id, peer_info in current.items():
                    self._broker.add_remote_peer(peer_id, peer_info)

                # Remove stale peers (ones that disappeared from remote)
                for peer_id in list(self._last_peers.keys()):
                    if peer_id not in current:
                        self._broker.remove_remote_peer(peer_id)

                self._last_peers = current
                logger.debug(
                    "RemoteSyncClient: synced %d remote peers from %s",
                    len(current),
                    self._peer_url,
                )
        except httpx.RequestError as e:
            logger.debug("RemoteSyncClient: HTTP error during sync: %s", e)
        except Exception as e:
            logger.debug("RemoteSyncClient: unexpected error during sync: %s", e)

    def send_to_remote(self, to_peer: str, message_data: dict) -> dict:
        """Proxy mesh_send to remote /mesh/send endpoint.

        Args:
            to_peer: Target peer ID on remote machine
            message_data: Dict with from_peer, content, type, etc.

        Returns:
            Dict with {"ok": True, ...} or {"ok": False, "error": "..."}
        """
        if not self._peer_url:
            return {"ok": False, "error": "no remote peer URL configured"}

        try:
            with httpx.Client(timeout=10) as client:
                headers = {}
                if self._shared_secret:
                    headers["Authorization"] = f"Bearer {self._shared_secret}"

                payload = {
                    "from_peer": message_data.get("from_peer", ""),
                    "to_peer": to_peer,
                    "content": message_data.get("content", ""),
                    "type": message_data.get("type", "text"),
                }

                resp = client.post(
                    f"{self._peer_url}/mesh/send",
                    json=payload,
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.RequestError as e:
            logger.debug(
                "RemoteSyncClient: HTTP error sending to remote peer %s: %s",
                to_peer,
                e,
            )
            return {"ok": False, "error": f"remote send failed: {e}"}
        except Exception as e:
            logger.debug(
                "RemoteSyncClient: unexpected error sending to remote peer %s: %s",
                to_peer,
                e,
            )
            return {"ok": False, "error": f"remote send error: {e}"}

    def _discovery_loop(self) -> None:
        """Background thread: discover remote SLM via mDNS."""
        if not ZEROCONF_AVAILABLE:
            return

        try:
            self._zeroconf = Zeroconf()
            ServiceBrowser(self._zeroconf, "_slm-mesh._tcp.local.", self)
            logger.info("RemoteSyncClient: mDNS browser started")

            # Keep thread alive
            while not self._stop_event.is_set():
                time.sleep(1)
        except Exception as e:
            logger.debug("RemoteSyncClient: mDNS discovery error: %s", e)
        finally:
            if self._zeroconf:
                try:
                    self._zeroconf.close()
                except Exception:
                    pass

    def add_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        """Zeroconf callback: service discovered."""
        try:
            if not ZEROCONF_AVAILABLE:
                return
            info = zeroconf.get_service_info(service_type, name)
            if info:
                for addr in _service_ip_addresses(info):
                    port = info.port or 8765
                    peer_url = _peer_url(addr, port)
                    self._update_peer_url(addr, port)
                    logger.info(
                        "RemoteSyncClient: discovered SLM at %s", peer_url
                    )
                    return
        except Exception as e:
            logger.debug("RemoteSyncClient: mDNS add_service error: %s", e)

    def remove_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        """Zeroconf callback: service disappeared."""
        logger.debug("RemoteSyncClient: service removed: %s", name)

    def update_service(self, zeroconf: Any, service_type: str, name: str) -> None:
        """Zeroconf callback: service updated."""
        self.add_service(zeroconf, service_type, name)

    def _update_peer_url(self, host: str, port: int) -> None:
        """Update peer URL from discovery."""
        new_url = _peer_url(host, port)
        if self._peer_url != new_url:
            self._peer_url = new_url
            logger.info("RemoteSyncClient: updated peer URL to %s", new_url)
