"""mDNS/DNS-SD advertising for the SLM daemon mesh substrate.

This module provides :class:`MeshAdvertiser`, which registers the running SLM
daemon as a ``_slm-mesh._tcp.local.`` service so that other SLM instances on
the same LAN can discover it via :class:`~superlocalmemory.mesh.remote_sync.RemoteSyncClient`.

Advertising is **opt-in and disabled by default** (backward-compatible).
Existing installations see zero behavior change.  Enable via::

    SLM_MESH_ADVERTISE=1    # also accepts: on, true, yes (case-insensitive)

WHY OPT-IN:  SLM runs on corporate-managed endpoints where unsolicited mDNS
multicast traffic may violate network policy or trigger security tooling.
Advertising fires only when an operator explicitly sets ``SLM_MESH_ADVERTISE``.

Service type: ``_slm-mesh._tcp.local.``
This MUST match the type browsed by :class:`RemoteSyncClient` in
``superlocalmemory.mesh.remote_sync`` (grep ``_slm-mesh._tcp.local.``).

Instance name: ``slm-{node_id}-{port}._slm-mesh._tcp.local.``
The port suffix prevents :exc:`~zeroconf.NonUniqueNameException` when two
SLM daemon instances run on the same host (e.g. primary on 8765, test on 8766).

Properties TXT record: contains ``node_id`` and any caller-supplied metadata.
**NEVER include secrets, tokens, or passwords** — TXT records are broadcast in
plaintext across the LAN and are visible to every host on the subnet.

All operations are fail-soft: any error is logged as WARNING and the daemon
continues.  ``stop()`` is idempotent.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from socket import inet_aton
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVICE_TYPE: str = "_slm-mesh._tcp.local."
"""DNS-SD service type — must match what RemoteSyncClient browses."""

_ADVERTISE_TRUTHY: frozenset[str] = frozenset({"1", "on", "true", "yes"})
"""Accepted values for SLM_MESH_ADVERTISE (matched case-insensitively)."""

# ---------------------------------------------------------------------------
# Optional zeroconf dependency guard
# Mirrors the exact pattern used in remote_sync.py so both modules behave
# identically when zeroconf is not installed.
# ---------------------------------------------------------------------------
try:
    from zeroconf import ServiceInfo, Zeroconf

    ZEROCONF_AVAILABLE: bool = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    Zeroconf = None  # type: ignore[assignment,misc]
    ServiceInfo = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_advertising_enabled() -> bool:
    """Return True only when ``SLM_MESH_ADVERTISE`` is set to a truthy value.

    Case-insensitive: ``'1'``, ``'on'``, ``'true'``, ``'yes'`` are accepted.
    Everything else (including unset) returns False (default-OFF, BC-safe).
    """
    return (
        os.environ.get("SLM_MESH_ADVERTISE", "").strip().lower()
        in _ADVERTISE_TRUTHY
    )


def _resolve_advertise_ip(injected_ip: Optional[str] = None) -> str:
    """Return the best local non-loopback IPv4 address for mDNS advertising.

    Uses a UDP connect trick (no packet is sent — the kernel consults the
    routing table to decide which source address would be used to reach a
    well-known public address).  Falls back to ``127.0.0.1`` when:

    * No routable interface is found.
    * The resolved address is itself a loopback address.
    * Any :exc:`OSError` is raised (e.g. no network at all).

    The ``127.0.0.1`` fallback allows two-daemon integration tests on the
    loopback interface to work without a real LAN.

    Args:
        injected_ip: When provided, skip resolution entirely and return this
                     value.  Used by tests to avoid real socket calls.

    Returns:
        A dotted-decimal IPv4 address string.
    """
    if injected_ip is not None:
        return injected_ip
    # Operator override for multi-homed / VPN hosts where the auto-detected
    # route would advertise the wrong NIC (audit P2). Explicit wins.
    env_ip = os.environ.get("SLM_MESH_ADVERTISE_IP", "").strip()
    if env_ip:
        return env_ip
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip: str = sock.getsockname()[0]
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    # No routable non-loopback interface — peers off this host cannot reach a
    # 127.0.0.1 advertisement. Warn so the operator can set SLM_MESH_ADVERTISE_IP.
    logger.warning(
        "MeshAdvertiser: no routable non-loopback IPv4 found; advertising "
        "127.0.0.1 (off-host peers cannot reach this node). Set "
        "SLM_MESH_ADVERTISE_IP to advertise a reachable address."
    )
    return "127.0.0.1"


# ---------------------------------------------------------------------------
# MeshAdvertiser
# ---------------------------------------------------------------------------


class MeshAdvertiser:
    """Register this SLM daemon as a discoverable mDNS service on the LAN.

    Advertising is **disabled by default** (opt-in via ``SLM_MESH_ADVERTISE``).

    Lifecycle::

        advertiser = MeshAdvertiser(service_port=8765, node_id="hostname")
        advertiser.start()   # no-op unless SLM_MESH_ADVERTISE is truthy
        ...
        advertiser.stop()    # idempotent; safe to call even if never started

    Thread safety:
        ``start()`` and ``stop()`` are protected by a :class:`threading.Lock`.
        The lock prevents double-registration and eliminates the race between
        a slow ``register_service`` call and a concurrent ``stop()`` call:
        ``start()`` holds the lock for the entire registration so ``stop()``
        sees a consistent, fully-initialised state.

    CRIT fixes applied:
        1. **Instance-name collision** — port is embedded in the instance label
           (``slm-{node_id}-{port}``), so two daemons on the same host register
           distinct names and avoid :exc:`~zeroconf.NonUniqueNameException`.
        2. **Event-loop blocking** — ``register_service`` probes for ~750 ms.
           Callers on an async startup path MUST wrap ``start()`` in
           ``asyncio.to_thread(advertiser.start)`` (see lifespan wiring in
           ``unified_daemon.py``).  ``MeshAdvertiser`` itself remains sync so
           it is trivially testable without an event loop.
        3. **stop() / register race** — ``_lock`` is held for the entire
           duration of ``_start_locked()``, so a concurrent ``stop()`` blocks
           until registration completes.  ``stop()`` clears state under the
           lock before releasing it, preventing double-close.
    """

    def __init__(
        self,
        service_port: int,
        node_id: str,
        properties: Optional[dict[str, str]] = None,
        *,
        _injected_ip: Optional[str] = None,
    ) -> None:
        """Create a MeshAdvertiser.

        Args:
            service_port: TCP port the SLM HTTP daemon is bound to.
            node_id: Stable unique identifier for this node (hostname is a
                     good choice).  Must be safe for DNS labels.
            properties: Optional extra metadata broadcast in the mDNS TXT
                        record.  Keep tiny.  **NEVER include secrets** — TXT
                        records are visible in plaintext to every host on the
                        LAN.
            _injected_ip: Test hook — override IP resolution.  Not part of the
                          public API.
        """
        self._service_port: int = service_port
        self._node_id: str = node_id
        # Defensive copy — immutable style; caller can't mutate our dict later.
        self._properties: dict[str, str] = dict(properties) if properties else {}
        self._injected_ip: Optional[str] = _injected_ip

        # Mutable state — all mutations happen under _lock.
        self._zeroconf: Any = None
        self._info: Any = None
        self._advertising: bool = False
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register the SLM service in mDNS.

        No-op when:
        * zeroconf is not installed (``ZEROCONF_AVAILABLE is False``).
        * ``SLM_MESH_ADVERTISE`` is unset or not in the truthy set.

        Any exception from the zeroconf stack is caught, logged as WARNING,
        and swallowed — the daemon must NEVER fail to start because of
        optional mDNS advertising.

        Note:
            ``register_service`` blocks for ~750 ms (mDNS probe phase).
            Call this from an async context via ``asyncio.to_thread(advertiser.start)``.
        """
        if not ZEROCONF_AVAILABLE:
            logger.debug(
                "MeshAdvertiser.start: zeroconf not installed; skipping mDNS advertising"
            )
            return
        if not _is_advertising_enabled():
            logger.debug(
                "MeshAdvertiser.start: SLM_MESH_ADVERTISE not set (or not truthy); "
                "mDNS advertising remains disabled (default-OFF, BC-safe)"
            )
            return

        with self._lock:
            if self._advertising:
                # Idempotent — already registered; nothing to do.
                return
            try:
                self._start_locked()
            except Exception as exc:
                logger.warning(
                    "MeshAdvertiser.start failed (non-fatal, daemon continues): %s", exc
                )
                # Clean up any partial state so stop() is still safe.
                self._zeroconf = None
                self._info = None
                self._advertising = False

    def stop(self) -> None:
        """Unregister the mDNS service and close Zeroconf.

        Idempotent — safe to call multiple times or when ``start()`` was never
        called.  Any exception is caught and logged as WARNING.

        CRIT note: state is cleared under ``_lock`` *before* calling
        ``unregister_service`` / ``close``, so a second concurrent ``stop()``
        sees ``_advertising=False`` and exits immediately without attempting
        to close an already-closed Zeroconf instance.
        """
        with self._lock:
            if not self._advertising:
                return
            # Capture and clear state under the lock atomically.
            zc: Any = self._zeroconf
            info: Any = self._info
            self._advertising = False
            self._zeroconf = None
            self._info = None

        # Perform I/O outside the lock so a concurrent stop() call (highly
        # unlikely given daemon lifecycle, but possible in tests) can't
        # deadlock waiting for us.
        # Audit P1: close() MUST run even when unregister_service raises,
        # otherwise the Zeroconf multicast sockets + threads leak. Best-effort
        # unregister, then close in a finally.
        try:
            if zc is not None and info is not None:
                try:
                    zc.unregister_service(info)
                except Exception as exc:
                    logger.warning(
                        "MeshAdvertiser.stop: unregister failed (closing anyway): %s",
                        exc,
                    )
        finally:
            if zc is not None:
                try:
                    zc.close()
                except Exception as exc:
                    logger.warning(
                        "MeshAdvertiser.stop: close failed (non-fatal): %s", exc
                    )

    @property
    def is_advertising(self) -> bool:
        """True when the service is currently registered in mDNS."""
        return self._advertising

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_locked(self) -> None:
        """Build :class:`~zeroconf.ServiceInfo` and register it.

        Called under ``self._lock``.  Raises on any error; the caller
        (``start()``) catches and logs.
        """
        ip: str = _resolve_advertise_ip(self._injected_ip)

        # Instance label carries port AND pid so the mDNS name is unique:
        # - port disambiguates two daemons on the SAME host (CRIT fix #1);
        # - pid disambiguates two machines that share a short hostname on the
        #   segment (audit P2 — cross-machine NonUniqueNameException).
        # The TXT node_id stays hostname-only (see _properties below).
        instance_label: str = f"slm-{self._node_id}-{self._service_port}-{os.getpid()}"
        service_name: str = f"{instance_label}.{_SERVICE_TYPE}"

        # Build bytes-keyed properties dict (zeroconf API requirement).
        # SECURITY: only whitelisted string content — never mirror env vars.
        # TXT records are broadcast in plaintext to the entire LAN subnet.
        safe_props: dict[str, str] = {
            "node_id": self._node_id,
            **self._properties,
        }
        byte_props: dict[bytes, bytes] = {
            k.encode("utf-8"): v.encode("utf-8") for k, v in safe_props.items()
        }

        info = ServiceInfo(
            type_=_SERVICE_TYPE,
            name=service_name,
            addresses=[inet_aton(ip)],
            port=self._service_port,
            properties=byte_props,
        )

        zc = Zeroconf()
        # register_service sends mDNS probe + announce packets.
        # This call blocks for ~750 ms (3 × 250 ms probe interval).
        # Callers on the async event loop MUST use asyncio.to_thread().
        try:
            zc.register_service(info)
        except Exception:
            # Audit P1: register_service failed AFTER Zeroconf() opened its
            # multicast sockets + background threads. Close it before
            # propagating so those resources are released instead of leaking
            # for the daemon's lifetime (start() catches and logs).
            try:
                zc.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            raise

        # Only update state after successful registration.
        self._zeroconf = zc
        self._info = info
        self._advertising = True

        logger.info(
            "MeshAdvertiser: registered '%s' on %s:%d",
            service_name,
            ip,
            self._service_port,
        )
