# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The remote listener: a second, TLS-only door for AI tools on other computers.

Off by default. ``slm remote enable --listen HOST:PORT`` writes
``<data root>/remote/remote.json``; ``SLM_REMOTE_LISTEN``, ``SLM_REMOTE_TLS_CERT``
and ``SLM_REMOTE_TLS_KEY`` override it. The main listener (127.0.0.1, plain
HTTP, used by the CLI, hooks and the dashboard) is unchanged.

Through this listener:

* only ``/mcp``, ``/mcp/...`` and ``GET /health`` exist - everything else is 404
  before any route runs (dashboard, ``/api``, ``/internal``, hooks);
* every caller is remote, even one on 127.0.0.1: the peer address is replaced
  by ``remote-listener-peer``, so no loopback trust anywhere in SLM applies;
* nothing answers with a redirect (``/mcp`` is served as ``/mcp/``).

A broken remote configuration never stops the main listener: the daemon logs why
remote access did not start and keeps serving this computer.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import os
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("superlocalmemory.remote")

LISTEN_ENV = "SLM_REMOTE_LISTEN"
CERT_ENV = "SLM_REMOTE_TLS_CERT"
KEY_ENV = "SLM_REMOTE_TLS_KEY"
CONFIG_FILE = ("remote", "remote.json")
TLS_DIR = ("remote", "tls")
REMOTE_PEER = "remote-listener-peer"

REFUSAL_TEMPLATE = (
    "Remote access is configured but not started: {reason} Run 'slm remote check' for "
    "details. The local daemon is running normally."
)


class RemoteListenerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RemoteListenerConfig:
    host: str
    port: int
    cert: Path
    key: Path
    server_names: tuple[str, ...]
    not_after: datetime
    #: The TLS settings built from ``cert`` and ``key`` while checking them.
    tls: Any = field(default=None, compare=False, repr=False)


def data_path(*parts: str) -> Path:
    from superlocalmemory.infra.data_root import state_path

    return state_path(*parts)


def read_settings() -> dict[str, str]:
    """The saved ``remote.json`` settings, ``{}`` when absent or unreadable."""
    path = data_path(*CONFIG_FILE)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("remote.json is unreadable (%s); remote access stays off", exc)
        return {"listen": "", "_error": "remote.json is unreadable"}
    return {k: str(v) for k, v in raw.items() if isinstance(v, (str, int))} \
        if isinstance(raw, dict) else {}


def parse_listen(value: str) -> tuple[str, int]:
    text = value.strip()
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port_text = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port_text = text.rpartition(":")
    try:
        ipaddress.ip_address(host)
        port = int(port_text)
    except ValueError as exc:
        raise RemoteListenerError(
            "invalid_listen",
            f"'{value}' is not HOST:PORT with an IP address (for example "
            "192.168.1.10:8443 or [::1]:8443).") from exc
    if not 1 <= port <= 65535:
        raise RemoteListenerError("invalid_listen", f"Port {port} is out of range.")
    return host, port


def _key_problem(path: Path) -> str | None:
    if os.name != "posix":
        return None
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return (f"the TLS private key {path} can be read by other users "
                f"(run: chmod 600 {path})")
    return None


def _inspect_cert(path: Path) -> tuple[tuple[str, ...], datetime]:
    from cryptography import x509
    from cryptography.x509.oid import ExtendedKeyUsageOID

    try:
        cert = x509.load_pem_x509_certificate(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RemoteListenerError("cert_unreadable",
                                  f"the TLS certificate {path} cannot be read.") from exc
    now = datetime.now(timezone.utc)
    if cert.not_valid_before_utc > now:
        raise RemoteListenerError("cert_not_yet_valid", "the TLS certificate is not valid yet.")
    if cert.not_valid_after_utc <= now:
        raise RemoteListenerError(
            "cert_expired",
            "the TLS certificate has expired (renew it: slm remote tls init --force ...).")
    try:
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound as exc:
        raise RemoteListenerError("cert_no_server_auth",
                                  "the TLS certificate is not a server certificate.") from exc
    if ExtendedKeyUsageOID.SERVER_AUTH not in eku:
        raise RemoteListenerError("cert_no_server_auth",
                                  "the TLS certificate is not a server certificate.")
    try:
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound as exc:
        raise RemoteListenerError("cert_no_san",
                                  "the TLS certificate names no host (no SAN).") from exc
    names = [n.lower().rstrip(".") for n in san.get_values_for_type(x509.DNSName)]
    for ip in san.get_values_for_type(x509.IPAddress):
        names.append(f"[{ip}]" if ip.version == 6 else str(ip))
    if not names:
        raise RemoteListenerError("cert_no_san", "the TLS certificate names no host (no SAN).")
    return tuple(dict.fromkeys(names)), cert.not_valid_after_utc


def load_remote_listener_config(
    env: Mapping[str, str] | None = None,
    *,
    main_port: int | None = None,
) -> RemoteListenerConfig | None:
    """The validated configuration, ``None`` when remote access is off.

    Raises :class:`RemoteListenerError` when it is configured but unusable.
    """
    environ = env if env is not None else os.environ
    settings = read_settings()
    listen = (environ.get(LISTEN_ENV) or settings.get("listen") or "").strip()
    if settings.get("_error") and not environ.get(LISTEN_ENV):
        raise RemoteListenerError("config_unreadable", "remote/remote.json is unreadable.")
    if not listen:
        return None
    host, port = parse_listen(listen)
    problem = uvicorn_problem()
    if problem:
        raise RemoteListenerError("uvicorn_untested", problem + ".")
    if main_port is not None and port == int(main_port):
        raise RemoteListenerError("port_conflict",
                                  f"port {port} is the main daemon port; choose another.")
    from superlocalmemory.core.remote_mode import mcp_stateless

    if not mcp_stateless():
        raise RemoteListenerError(
            "stateful_mcp", "remote access needs the default stateless MCP transport "
            "(unset SLM_MCP_STATEFUL / SLM_MCP_STATELESS=0).")
    cert = Path(environ.get(CERT_ENV) or settings.get("tls_cert")
                or data_path(*TLS_DIR, "server.pem")).expanduser()
    key = Path(environ.get(KEY_ENV) or settings.get("tls_key")
               or data_path(*TLS_DIR, "server.key")).expanduser()
    for label, path in (("certificate", cert), ("private key", key)):
        if not path.is_file():
            raise RemoteListenerError(
                "tls_missing", f"the TLS {label} {path} does not exist "
                "(create one: slm remote tls init --name <host> --ip <address>).")
    problem = _key_problem(key)
    if problem:
        raise RemoteListenerError("tls_key_exposed", problem + ".")
    names, not_after = _inspect_cert(cert)
    return RemoteListenerConfig(host, port, cert, key, names, not_after,
                                tls=_tls_context(cert, key))


def _tls_context(cert: Path, key: Path) -> Any:
    """The listener's TLS settings, or the reason the pair cannot be used."""
    import ssl

    from superlocalmemory.server.remote_conn_guard import server_tls_context

    try:
        return server_tls_context(cert, key)
    except (OSError, ssl.SSLError, ValueError) as exc:
        raise RemoteListenerError(
            "tls_unusable", f"the TLS certificate {cert} and private key {key} cannot be "
            "used together (renew them: slm remote tls init --force ...).") from exc


def try_load_config(main_port: int | None = None) -> tuple[RemoteListenerConfig | None,
                                                            RemoteListenerError | None]:
    try:
        return load_remote_listener_config(main_port=main_port), None
    except RemoteListenerError as exc:
        return None, exc


def mcp_allowed_hosts(config: RemoteListenerConfig) -> list[str]:
    """``Host`` header values the MCP SDK must accept on the remote listener."""
    hosts = [f"{name}:{config.port}" for name in config.server_names]
    if config.port == 443:
        hosts += list(config.server_names)
    return hosts


# -- the ASGI wrapper -----------------------------------------------------------------

_NOT_FOUND = b'{"error":"not_found"}'


class RemoteListenerASGI:
    """Outermost wrapper for the remote listener only."""

    def __init__(self, app: Any, server_names: tuple[str, ...] = ()) -> None:
        self.app = app
        self._names = frozenset(n.strip("[]") for n in server_names)

    def _host_is_ours(self, scope: dict[str, Any]) -> bool:
        from superlocalmemory.server.host_guard import _host_name

        for name, value in scope.get("headers") or ():
            if name == b"host":
                return _host_name(value.decode("latin-1")) in self._names
        return False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind == "lifespan":
            return  # the main server owns startup and shutdown
        if kind != "http":
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            return
        path = scope.get("path", "")
        if path == "/health" and scope.get("method") in ("GET", "HEAD"):
            pass
        elif path == "/mcp" or path.startswith("/mcp/"):
            if path == "/mcp":
                path = "/mcp/"
        else:
            await send({"type": "http.response.start", "status": 404,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(_NOT_FOUND)).encode())]})
            await send({"type": "http.response.body", "body": _NOT_FOUND})
            return
        client = scope.get("client") or ("", 0)
        from superlocalmemory.server.remote_access import REMOTE_LISTENER_SCOPE_KEY

        scope = dict(scope, path=path, raw_path=path.encode("ascii", "ignore"),
                     client=(REMOTE_PEER, client[1]), slm_remote_peer=client[0])
        scope[REMOTE_LISTENER_SCOPE_KEY] = True
        # A name on this listener's own certificate is this computer (the
        # host guard would otherwise refuse names not in SLM_ALLOWED_HOSTS).
        scope["slm_remote_host_verified"] = self._host_is_ours(scope)
        await self.app(scope, receive, send)


# -- running the pair -----------------------------------------------------------------


def bind_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    try:
        sock.bind((host, port))
        sock.listen(socket.SOMAXCONN)
    except OSError:
        sock.close()
        raise
    return sock


#: uvicorn releases whose Server internals this pairing was tested against:
#: ``serve(sockets)``, ``shutdown(sockets)``, ``capture_signals()``,
#: ``should_exit`` and ``started``. ``tests/server/test_uvicorn_contract.py``
#: fails clearly if any of them changes. Outside this range remote access is
#: refused rather than run on untested internals.
UVICORN_TESTED = ((0, 46), (0, 47))


def uvicorn_problem() -> str | None:
    try:
        import uvicorn
    except ImportError:
        return "uvicorn is not installed"
    parts = tuple(int(p) for p in uvicorn.__version__.split(".")[:2] if p.isdigit())
    low, high = UVICORN_TESTED
    if not low <= parts < high:
        return (f"uvicorn {uvicorn.__version__} is not a tested version for remote access "
                f"(tested: {low[0]}.{low[1]}.x); install the version SLM pins")
    return None


def make_servers(app: Any, main_config: Any, remote: RemoteListenerConfig):
    """The two uvicorn servers. Imported lazily: uvicorn is a daemon dependency."""
    import uvicorn

    from superlocalmemory.server.forwarded_guard import uvicorn_proxy_options
    from superlocalmemory.server.remote_conn_guard import (
        server_tls_context,
        tls_gate_protocol_class,
    )

    # TLS is done by the connection guard, not uvicorn, so the handshake has a
    # deadline and a connection still in it counts toward the connection caps.
    gate_class = tls_gate_protocol_class(
        remote.tls if remote.tls is not None else server_tls_context(remote.cert, remote.key))

    class QuietServer(uvicorn.Server):
        """The remote server: no signal handlers (the main server owns them) and
        no lifespan (``lifespan="off"``; the app's lifespan yields no state)."""

        @contextlib.contextmanager
        def capture_signals(self):  # noqa: D401 — the main server owns signals
            yield

        async def shutdown(self, sockets=None):
            for gate in list(gate_class.slm_limits.handshaking):
                gate.shutdown()
            await super().shutdown(sockets=sockets)

    remote_cfg = uvicorn.Config(
        RemoteListenerASGI(app, remote.server_names), host=remote.host, port=remote.port,
        lifespan="off",
        # Callers must finish TLS and then send each request's headers within
        # deadlines, and the number of connections is capped (see
        # remote_conn_guard). Only /mcp (plain HTTP inside TLS) is served here,
        # so WebSocket upgrades are off.
        http=gate_class, ws="none",
        # Callers connect directly over TLS: forwarding headers are never read
        # here, whatever SLM_TRUSTED_PROXIES says for the main listener.
        **uvicorn_proxy_options({}),
        log_level="warning", timeout_graceful_shutdown=10,
    )
    remote_srv = QuietServer(remote_cfg)

    class PrimaryServer(uvicorn.Server):
        """Stops the remote server before the application shuts down."""

        async def shutdown(self, sockets=None):
            remote_srv.should_exit = True
            task = getattr(self, "_slm_remote_task", None)
            if task is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(task), timeout=10)
            await super().shutdown(sockets=sockets)

    main_srv = PrimaryServer(main_config)
    return main_srv, remote_srv


async def serve_pair(main_srv: Any, main_sockets: list, remote_srv: Any,
                     remote_sockets: list) -> None:
    """Run both. The remote server starts once the main server is up; if it dies
    the main server keeps running; when the main server exits, so does it."""
    main_task = asyncio.create_task(main_srv.serve(sockets=main_sockets))
    while not getattr(main_srv, "started", False) and not main_task.done():
        await asyncio.sleep(0.05)
    if main_task.done():
        await main_task
        return

    async def _remote() -> None:
        try:
            await remote_srv.serve(sockets=remote_sockets)
        except BaseException as exc:  # noqa: BLE001 — never take the main daemon down
            logger.critical("Remote listener stopped unexpectedly: %s. The local daemon "
                            "keeps running.", exc)

    remote_task = asyncio.create_task(_remote())
    main_srv._slm_remote_task = remote_task
    try:
        await main_task
    finally:
        remote_srv.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(remote_task, timeout=12)


__all__ = [
    "CERT_ENV",
    "KEY_ENV",
    "LISTEN_ENV",
    "REFUSAL_TEMPLATE",
    "REMOTE_PEER",
    "RemoteListenerASGI",
    "RemoteListenerConfig",
    "RemoteListenerError",
    "bind_socket",
    "load_remote_listener_config",
    "make_servers",
    "mcp_allowed_hosts",
    "parse_listen",
    "read_settings",
    "serve_pair",
    "try_load_config",
]
