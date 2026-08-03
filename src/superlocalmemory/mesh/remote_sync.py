# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLM Mesh — Remote Sync Client.

HTTP-based synchronization with a remote SLM instance.
Populates broker._remote_peers from the remote /mesh/peers endpoint.
Proxies mesh_send to remote when the target peer lives on the remote machine.
Optional mDNS discovery via zeroconf.

New in 3b-1: durable remote outbox — failed sends are enqueued and retried
  with exponential back-off until delivered or TTL-expired.
New in 3b-3: TLS-pinned transport — opt-in SHA-256 cert pinning + custom CA.

Environment variables:
  SLM_MESH_PEER_URL:        Full URL of remote SLM (e.g. http://192.168.1.100:8765)
  SLM_MESH_SHARED_SECRET:   Shared auth secret for remote SLM
  SLM_MESH_DISCOVERY:       'on'|'off' (default 'on') — enable mDNS discovery
  SLM_MESH_TLS:             'on'|'off' (default 'off') — use https:// for discovered peers
  SLM_MESH_TLS_CA:          Path to custom CA bundle (PEM) for TLS verification
  SLM_MESH_TLS_PIN:         Hex SHA-256 of peer leaf cert DER for cert pinning
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import ssl
import threading
import time
import ipaddress
from typing import Any

import httpx

logger = logging.getLogger("superlocalmemory.mesh.remote_sync")

_PRODUCTION_TRUTHY = frozenset({"1", "true", "yes", "on", "production", "prod"})


def _is_production_mode() -> bool:
    """True when SLM_MESH_PRODUCTION signals a hardened deployment."""
    return os.environ.get("SLM_MESH_PRODUCTION", "").strip().lower() in _PRODUCTION_TRUTHY


def _is_plaintext_url(url: str) -> bool:
    """True for http:// and ws:// URLs (not TLS-protected)."""
    lower = url.strip().lower()
    return lower.startswith("http://") or lower.startswith("ws://")


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
    """Format an IP literal safely for an HTTP authority.

    Always produces http:// — preserved exactly for backward compatibility.
    Use _peer_url_with_scheme() when an https:// override is needed.
    """
    address = ipaddress.ip_address(host)
    rendered = str(address)
    if address.version == 6:
        # RFC 6874 requires a percent sign in an IPv6 zone identifier to be
        # escaped when the literal appears inside a URI authority.
        rendered = rendered.replace("%", "%25")
        rendered = f"[{rendered}]"
    return f"http://{rendered}:{int(port)}"


def _peer_url_with_scheme(host: str, port: int, tls: bool = False) -> str:
    """Format an IP literal safely for an HTTP or HTTPS authority.

    Args:
        host: IP address string (v4 or v6).
        port: Port number.
        tls: When True, emit https:// instead of http://.

    Returns:
        Scheme-correct URL string.
    """
    address = ipaddress.ip_address(host)
    rendered = str(address)
    if address.version == 6:
        rendered = rendered.replace("%", "%25")
        rendered = f"[{rendered}]"
    scheme = "https" if tls else "http"
    return f"{scheme}://{rendered}:{int(port)}"


def _get_cert_sha256(
    host: str,
    port: int,
    ca_file: str | None = None,
    timeout: int = 5,
) -> str:
    """Return hex SHA-256 of the server's leaf certificate DER bytes.

    Opens a raw TLS connection, retrieves the peer cert in DER form, and
    returns its SHA-256 digest as a lowercase hex string (no colons).

    Raises:
        ssl.SSLError: TLS negotiation failed or cert not trusted.
        socket.error: Connection refused or timed out.
        OSError: Any underlying network error.
    """
    ctx = ssl.create_default_context(cafile=ca_file)
    with socket.create_connection((host, port), timeout=timeout) as raw_sock:
        with ctx.wrap_socket(raw_sock, server_hostname=host) as tls_sock:
            cert_der = tls_sock.getpeercert(binary_form=True)
            if not cert_der:
                raise ssl.SSLError("peer returned no certificate")
            return hashlib.sha256(cert_der).hexdigest().lower()


# Optional zeroconf for mDNS discovery
try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    Zeroconf = None
    ServiceBrowser = None
    ServiceInfo = None

# Optional durable outbox (3b-1) — fail-open so missing module never breaks start()
try:
    from .outbox_remote import RemoteOutbox
    _OUTBOX_AVAILABLE = True
except ImportError:
    RemoteOutbox = None  # type: ignore[assignment,misc]
    _OUTBOX_AVAILABLE = False


class RemoteSyncClient:
    """HTTP-based sync client for multi-machine mesh coordination.

    Syncs remote peers from a peer SLM instance periodically.
    Proxies mesh_send to remote when target peer lives on remote machine.
    Optionally discovers remote SLM via mDNS.

    3b-1: Failed sends are stored in a durable SQLite outbox and retried
    with exponential back-off so messages survive peer downtime.

    3b-3: Optional TLS cert pinning (SLM_MESH_TLS_PIN) and custom CA
    (SLM_MESH_TLS_CA). Default behavior (no env) is byte-for-byte identical
    to previous releases.
    """

    def __init__(self, broker: Any) -> None:
        """Initialize sync client.

        Args:
            broker: Reference to MeshBroker instance.
        """
        self._broker = broker
        self._peer_url: str | None = os.environ.get("SLM_MESH_PEER_URL") or None
        self._shared_secret: str | None = os.environ.get("SLM_MESH_SHARED_SECRET") or None
        self._discovery_enabled: bool = (
            os.environ.get("SLM_MESH_DISCOVERY", "on") != "off"
        )
        # M05: the shared secret is a bearer token — whoever receives it can
        # replay it. mDNS discovery is unauthenticated (anyone on the LAN can
        # advertise _slm-mesh._tcp.local.), so we must NOT push the secret to a
        # discovered peer unless the operator explicitly trusts LAN discovery.
        # An explicitly-configured peer (SLM_MESH_PEER_URL) is trusted; a peer
        # set programmatically stays trusted; only the discovery path can
        # downgrade trust.
        self._peer_url_from_config: bool = self._peer_url is not None
        self._trust_discovered: bool = (
            os.environ.get("SLM_MESH_TRUST_DISCOVERED", "off").strip().lower()
            in ("1", "on", "true", "yes")
        )
        self._peer_url_trusted: bool = True
        self._sync_thread: threading.Thread | None = None
        self._discovery_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._zeroconf: Zeroconf | None = None
        self._last_peers: dict[str, dict] = {}
        # 3b-1: lazily initialised durable outbox — None until first access
        self._outbox: RemoteOutbox | None = None  # type: ignore[type-arg]

    def start(self) -> None:
        """Start background sync and discovery threads.

        In production mode (SLM_MESH_PRODUCTION=1), plaintext peer URLs
        (http:// or ws://) are rejected so credentials cannot be sent in
        the clear and traffic cannot be intercepted.
        """
        if self._peer_url and _is_production_mode() and _is_plaintext_url(self._peer_url):
            raise ValueError(
                f"Production mesh requires TLS transport (https:// or wss://); "
                f"got plaintext peer URL: {self._peer_url!r}. "
                "Set SLM_MESH_PEER_URL to an https:// endpoint or "
                "unset SLM_MESH_PRODUCTION to allow plaintext in dev/local mode."
            )

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

    # ------------------------------------------------------------------
    # TLS helpers (3b-3)
    # ------------------------------------------------------------------

    def _http_client(self, timeout: int) -> httpx.Client:
        """Build an httpx.Client with appropriate TLS configuration.

        Honors SLM_MESH_TLS_CA (custom CA bundle path). When unset, uses
        system CAs (verify=True). The cert-pin check is done separately in
        _check_cert_pin() before the first byte is sent.

        Default (no env) produces httpx.Client(timeout=timeout) — identical
        to the previous hard-coded behavior.
        """
        ca_path = os.environ.get("SLM_MESH_TLS_CA") or None
        if ca_path:
            return httpx.Client(verify=ca_path, timeout=timeout)
        return httpx.Client(timeout=timeout)

    def _check_cert_pin(self, peer_url: str) -> tuple[bool, str]:
        """Pre-flight SHA-256 certificate pin check for an https:// URL.

        Opens a brief raw TLS connection to retrieve the leaf certificate,
        computes its SHA-256 hash, and compares it to the configured pin.
        This happens BEFORE the actual HTTP request so no payload is sent
        to a peer with a mismatched certificate.

        Normalization: both actual and expected hex strings are lowercased
        and colon-stripped before comparison to avoid case-sensitivity bugs.

        Args:
            peer_url: URL being targeted.

        Returns:
            (True, "") if pin matches or no pin is configured.
            (False, reason) if pin is configured and does not match.
        """
        pin_env = os.environ.get("SLM_MESH_TLS_PIN") or None
        if not pin_env:
            return True, ""

        parsed = httpx.URL(peer_url)
        if parsed.scheme != "https":
            # Pinning is only meaningful over TLS
            return True, ""

        host = parsed.host
        port = parsed.port or 443
        # Normalise: lowercase and strip colons (handles both plain hex and
        # colon-separated fingerprint formats e.g. "AB:CD:...").
        expected_pin = pin_env.strip().lower().replace(":", "")
        ca_path = os.environ.get("SLM_MESH_TLS_CA") or None

        try:
            actual_pin = _get_cert_sha256(host, port, ca_file=ca_path)
        except (ssl.SSLError, socket.error, OSError) as exc:
            return False, f"cert pin check connection failed: {exc}"

        if actual_pin != expected_pin:
            return False, (
                f"certificate pin mismatch for {host}:{port} "
                f"(expected {expected_pin!r}, got {actual_pin!r})"
            )
        return True, ""

    # ------------------------------------------------------------------
    # Outbox helpers (3b-1)
    # ------------------------------------------------------------------

    def _get_outbox(self) -> RemoteOutbox | None:  # type: ignore[return]
        """Lazily initialise the RemoteOutbox using the broker's DB path.

        Returns None (and logs once) if the outbox cannot be initialised
        so the online send path is always unaffected.
        """
        if self._outbox is not None:
            return self._outbox
        if not _OUTBOX_AVAILABLE:
            return None
        db_path = getattr(self._broker, "_db_path", None)
        if not db_path:
            logger.debug("RemoteSyncClient: no db_path on broker — outbox disabled")
            return None
        try:
            self._outbox = RemoteOutbox(db_path)  # type: ignore[call-arg]
        except Exception as exc:
            logger.error("RemoteSyncClient: failed to init RemoteOutbox: %s", exc)
        return self._outbox

    def _enqueue_on_failure(
        self,
        peer_url: str,
        to_peer: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        now: float,
    ) -> None:
        """Enqueue a failed send to the durable outbox.

        Headers are stored for audit purposes only. The drain loop re-signs
        fresh headers (new nonce + timestamp) before each retry attempt to
        avoid stale-timestamp rejections at the receiving peer.
        """
        outbox = self._get_outbox()
        if outbox is not None:
            outbox.enqueue(peer_url, to_peer, payload, headers, now=now)

    def _build_signed_headers(
        self,
        payload: dict[str, Any],
        to_peer: str,
        base_headers: dict[str, str],
    ) -> dict[str, str]:
        """Add fresh HMAC signing headers to base_headers if a secret is configured.

        Generates a new nonce + current timestamp so each signing is unique
        and replay-safe. Returns base_headers unchanged when signing is
        not applicable.
        """
        if not (self._shared_secret and self._peer_url_trusted):
            return dict(base_headers)

        import secrets as _sec
        from .broker_security import sign_mesh_message

        from_peer = payload.get("from_peer", "")
        content = payload.get("content", "")
        nonce = _sec.token_hex(16)
        ts = str(int(time.time()))
        sig = sign_mesh_message(
            self._shared_secret, from_peer, to_peer, content, nonce, ts,
        )
        return {
            **base_headers,
            "X-Mesh-Sig": sig,
            "X-Mesh-Nonce": nonce,
            "X-Mesh-Ts": ts,
        }

    def _drain_outbox(self) -> None:
        """Re-attempt delivery for due outbox items (called from _sync_loop).

        Processes up to _BATCH_LIMIT rows per call (bounded by outbox.due()).
        Each row gets fresh signing headers. On success: deleted. On any
        failure: mark_retry() with exponential back-off. Always prunes
        expired rows at the end of the drain pass.

        This method only runs when self._peer_url is set (guarded in
        _sync_loop) — when no peer is configured the outbox is inert.
        """
        outbox = self._get_outbox()
        if outbox is None:
            return

        now = time.time()
        due_rows = outbox.due(now)
        if not due_rows:
            outbox.prune_expired(now)
            return

        for row in due_rows:
            row_id: int = row["id"]
            peer_url: str = row["peer_url"]
            to_peer: str = row["to_peer"]

            try:
                payload: dict[str, Any] = json.loads(row["payload"])
            except (json.JSONDecodeError, ValueError) as exc:
                logger.debug(
                    "RemoteOutbox: corrupt payload in row %d — deleting: %s",
                    row_id, exc,
                )
                outbox.delete(row_id)
                continue

            # Pre-flight pin check (3b-3)
            pin_ok, pin_err = self._check_cert_pin(peer_url)
            if not pin_ok:
                logger.debug(
                    "RemoteOutbox: pin check failed for row %d (%s): %s",
                    row_id, peer_url, pin_err,
                )
                outbox.mark_retry(row_id, now)
                continue

            # Rebuild fresh headers: auth bearer + fresh HMAC signature
            base_headers = self._auth_headers()
            full_headers = self._build_signed_headers(payload, to_peer, base_headers)

            try:
                with self._http_client(timeout=10) as client:
                    resp = client.post(
                        f"{peer_url}/mesh/send",
                        json=payload,
                        headers=full_headers,
                        timeout=10,
                    )
                    resp.raise_for_status()

                outbox.delete(row_id)
                logger.debug(
                    "RemoteOutbox: delivered row %d → %s (to_peer=%s)",
                    row_id, peer_url, to_peer,
                )
            except httpx.RequestError as exc:
                logger.debug(
                    "RemoteOutbox: HTTP error for row %d: %s", row_id, exc
                )
                outbox.mark_retry(row_id, now)
            except httpx.HTTPStatusError as exc:
                logger.debug(
                    "RemoteOutbox: non-2xx for row %d: %s", row_id, exc
                )
                outbox.mark_retry(row_id, now)
            except Exception as exc:
                logger.debug(
                    "RemoteOutbox: unexpected error for row %d: %s", row_id, exc
                )
                outbox.mark_retry(row_id, now)

        outbox.prune_expired(now)

    # ------------------------------------------------------------------
    # Core sync loop
    # ------------------------------------------------------------------

    def _sync_loop(self) -> None:
        """Background thread: sync remote peers every 30s, then drain outbox."""
        while not self._stop_event.is_set():
            if self._peer_url:
                try:
                    self._sync_peers_from_remote()
                except Exception as exc:
                    logger.debug("RemoteSyncClient: sync error: %s", exc)

                # 3b-1: drain outbox after every peer sync
                try:
                    self._drain_outbox()
                except Exception as exc:
                    logger.debug("RemoteSyncClient: outbox drain error: %s", exc)

            # Wait 30s before next sync
            if self._stop_event.wait(30):
                break

    def _auth_headers(self) -> dict[str, str]:
        """Bearer header for the current peer — but ONLY if that peer is
        trusted. Prevents leaking the shared secret to a spoofed mDNS peer
        (M05)."""
        if self._shared_secret and self._peer_url_trusted:
            return {"Authorization": f"Bearer {self._shared_secret}"}
        return {}

    def _sync_peers_from_remote(self) -> None:
        """Fetch peers from remote /mesh/peers and update broker."""
        if not self._peer_url:
            return

        try:
            with self._http_client(timeout=5) as client:
                headers = self._auth_headers()

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

        On success: returns the remote response dict (unchanged from pre-3b).
        On any failure (RequestError, non-2xx, unexpected): enqueues the
        message to the durable outbox for retry, then returns the same
        {"ok": False, ...} dict as before. The return contract is unchanged.

        Args:
            to_peer: Target peer ID on remote machine.
            message_data: Dict with from_peer, content, type, etc.

        Returns:
            Dict with {"ok": True, ...} or {"ok": False, "error": "..."}.
        """
        if not self._peer_url:
            return {"ok": False, "error": "no remote peer URL configured"}

        from_peer = message_data.get("from_peer", "")
        content = message_data.get("content", "")
        payload = {
            "from_peer": from_peer,
            "to_peer": to_peer,
            "content": content,
            "type": message_data.get("type", "text"),
        }
        base_headers = self._auth_headers()
        signed_headers = self._build_signed_headers(payload, to_peer, base_headers)

        # 3b-3: pre-flight cert pin check before sending any data
        pin_ok, pin_err = self._check_cert_pin(self._peer_url)
        if not pin_ok:
            logger.debug(
                "RemoteSyncClient: cert pin check failed for %s: %s",
                self._peer_url, pin_err,
            )
            err = f"certificate pin mismatch: {pin_err}"
            self._enqueue_on_failure(
                self._peer_url, to_peer, payload, base_headers, now=time.time()
            )
            return {"ok": False, "error": err}

        try:
            with self._http_client(timeout=10) as client:
                resp = client.post(
                    f"{self._peer_url}/mesh/send",
                    json=payload,
                    headers=signed_headers,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()

        except httpx.RequestError as e:
            logger.debug(
                "RemoteSyncClient: HTTP error sending to remote peer %s: %s",
                to_peer, e,
            )
            self._enqueue_on_failure(
                self._peer_url, to_peer, payload, base_headers, now=time.time()
            )
            return {"ok": False, "error": f"remote send failed: {e}"}

        except httpx.HTTPStatusError as e:
            logger.debug(
                "RemoteSyncClient: non-2xx sending to remote peer %s: %s",
                to_peer, e,
            )
            self._enqueue_on_failure(
                self._peer_url, to_peer, payload, base_headers, now=time.time()
            )
            return {"ok": False, "error": f"remote send non-2xx: {e}"}

        except Exception as e:
            logger.debug(
                "RemoteSyncClient: unexpected error sending to remote peer %s: %s",
                to_peer, e,
            )
            self._enqueue_on_failure(
                self._peer_url, to_peer, payload, base_headers, now=time.time()
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
        """Update peer URL from mDNS discovery.

        Never overrides an explicitly-configured SLM_MESH_PEER_URL — explicit
        config is the source of truth and must not be hijacked by a spoofed
        mDNS announcement. A discovered peer is marked UNTRUSTED (the shared
        secret is withheld) unless SLM_MESH_TRUST_DISCOVERED is enabled (M05).

        3b-3: When SLM_MESH_TLS=on, discovered peers use https:// instead
        of the default http://. Default OFF preserves today's behavior.
        """
        if self._peer_url_from_config:
            logger.debug(
                "RemoteSyncClient: ignoring mDNS-discovered peer %s:%s — "
                "SLM_MESH_PEER_URL is explicitly configured",
                host, port,
            )
            return
        tls_enabled = (
            os.environ.get("SLM_MESH_TLS", "off").strip().lower()
            in ("1", "on", "true", "yes")
        )
        new_url = _peer_url_with_scheme(host, port, tls=tls_enabled)
        if self._peer_url != new_url:
            self._peer_url = new_url
            self._peer_url_trusted = self._trust_discovered
            logger.info(
                "RemoteSyncClient: updated peer URL to %s (mDNS-discovered, "
                "trusted=%s)", new_url, self._peer_url_trusted,
            )
            if self._shared_secret and not self._trust_discovered:
                logger.warning(
                    "RemoteSyncClient: a shared secret is set but "
                    "SLM_MESH_TRUST_DISCOVERED is off — the secret will NOT be "
                    "sent to mDNS-discovered peer %s. Set "
                    "SLM_MESH_TRUST_DISCOVERED=on to trust LAN-discovered peers.",
                    new_url,
                )
