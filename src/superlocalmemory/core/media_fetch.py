# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Download one picture from a web link without letting the link reach a private network.

A link is a way for a caller to make this machine open a connection, so every
step is checked:

* ``https`` only, no ``user:password@``, no proxy settings from the environment.
* The host name is resolved once. Every address in the answer must be a public
  one (after unwrapping IPv4-mapped, NAT64, 6to4 and Teredo forms); one private
  address refuses the whole answer.
* The connection goes to the checked address itself, with the original host in
  the ``Host`` header and in the TLS server name, so the name is never looked up
  a second time (no DNS rebinding window).
* Redirects are followed by hand, at most three, each one checked again.
* At most 25 MB is read (counted while streaming) and the whole fetch has 10 s.

The bytes are returned untouched; the caller sniffs them by their magic bytes.
The ``Content-Type`` header is passed on for logging only. Refusal reasons are
fixed sentences: they never repeat the link, its query string or its user info.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)

MAX_BYTES = 25 * 1024 * 1024
TIMEOUT_S = 10.0
MAX_REDIRECTS = 3
HOSTS_ENV = "SLM_MEDIA_URL_HOSTS"
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_IP = ipaddress.IPv4Address | ipaddress.IPv6Address
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_V4_COMPAT = ipaddress.ip_network("::/96")
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")
Resolver = Callable[[str, int], list[str]]


class MediaFetchRefused(Exception):
    """The link was not fetched; ``reason`` is safe to show (it never holds the link)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class FetchedMedia:
    data: bytes
    final_url: str
    content_type_header: str


@dataclass(frozen=True)
class _Plan:
    remote: bool
    hosts: tuple[str, ...]
    resolver: Resolver
    deadline: float
    clock: Callable[[], float]
    max_bytes: int


# -- address checks ------------------------------------------------------------

def unwrap(ip: _IP) -> tuple[_IP, ...]:
    """The IPv4 addresses hidden inside an IPv6 address (none for plain addresses)."""
    if ip.version != 6:
        return ()
    found: list[_IP] = []
    if ip.ipv4_mapped is not None:
        found.append(ip.ipv4_mapped)
    elif ip in _NAT64 or ip in _V4_COMPAT:
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    if ip.sixtofour is not None:
        found.append(ip.sixtofour)
    if ip.teredo is not None:
        found.extend(ip.teredo)
    return tuple(found)


def vet_address(address: str | _IP) -> bool:
    """True only for an address on the public internet, however it is written."""
    try:
        ip = ipaddress.ip_address(address) if isinstance(address, str) else address
    except ValueError:
        return False
    inner = unwrap(ip)
    if ip.version == 6 and (ip.ipv4_mapped is not None or ip in _V4_COMPAT or ip in _NAT64):
        return all(vet_address(i) for i in inner)
    if ip.version == 6 and ip in _NAT64_LOCAL:
        return False
    return ip.is_global and not ip.is_multicast and all(vet_address(i) for i in inner)


def _system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


# -- the link ------------------------------------------------------------------

def _refuse(reason: str) -> MediaFetchRefused:
    return MediaFetchRefused(reason)


def _parse(link: str) -> httpx.URL:
    try:
        url = httpx.URL(link.strip())
    except (httpx.InvalidURL, ValueError):
        raise _refuse("That link is not a valid web address.") from None
    if url.scheme != "https":
        raise _refuse("Only https links can be fetched.")
    if url.userinfo or not url.raw_host:
        raise _refuse("That link is not allowed (no user name or password in links).")
    return url


def _allow_list(configured: str | None) -> tuple[str, ...]:
    raw = os.environ.get(HOSTS_ENV, "") if configured is None else configured
    return tuple(h.strip().lower() for h in raw.split(",") if h.strip())


def _host_allowed(host: str, plan: _Plan) -> None:
    if not plan.hosts:
        if plan.remote:
            raise _refuse("Links are not accepted from remote callers until a host list is set.")
        return
    if host.lower() not in plan.hosts:
        raise _refuse("That host is not on the allowed list for image links.")


def _vetted_addresses(host: str, port: int, plan: _Plan) -> list[str]:
    try:
        literal = str(ipaddress.ip_address(host))
    except ValueError:
        literal = ""
    try:
        found = [literal] if literal else plan.resolver(host, port)
    except OSError:
        raise _refuse("That host name could not be looked up.") from None
    if not found:
        raise _refuse("That host name could not be looked up.")
    if not all(vet_address(a) for a in found):
        raise _refuse("That link points to a private or reserved network address.")
    return found


# -- one request ---------------------------------------------------------------

def _remaining(plan: _Plan) -> float:
    left = plan.deadline - plan.clock()
    if left <= 0:
        raise _refuse("Fetching that link took too long.")
    return left


def _send(client: httpx.Client, url: httpx.URL, ip: str, plan: _Plan) -> httpx.Response:
    host = url.raw_host.decode("ascii")
    shown = host if ":" not in host else f"[{host}]"
    port = f":{url.port}" if url.port else ""
    request = client.build_request(
        "GET", url.copy_with(host=ip),
        headers={"Host": shown + port, "Accept": "image/*", "Accept-Encoding": "identity"},
        extensions={"sni_hostname": host}, timeout=_remaining(plan),
    )
    return client.send(request, stream=True)


def _open(client: httpx.Client, url: httpx.URL, plan: _Plan) -> httpx.Response:
    host = url.raw_host.decode("ascii")
    _host_allowed(host, plan)
    addresses = _vetted_addresses(host, url.port or 443, plan)
    last: Exception | None = None
    for ip in addresses:
        try:
            return _send(client, url, ip, plan)
        except httpx.ConnectError as exc:  # try the next vetted address
            last = exc
        except httpx.TimeoutException:
            raise _refuse("Fetching that link took too long.") from None
        except (httpx.HTTPError, OSError):
            raise _refuse("That link could not be fetched.") from None
    raise _refuse("That link could not be reached.") from last


def _read_body(resp: httpx.Response, plan: _Plan) -> bytes:
    limit = plan.max_bytes
    too_big = _refuse(f"That image is too large ({limit // (1024 * 1024)} MB limit).")
    try:
        declared = int(resp.headers.get("content-length", "0") or 0)
    except ValueError:
        declared = 0
    if declared > limit:
        raise too_big
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > limit:
                raise too_big
            chunks.append(chunk)
            _remaining(plan)
    except httpx.TimeoutException:
        raise _refuse("Fetching that link took too long.") from None
    except httpx.HTTPError:
        raise _refuse("That link could not be fetched.") from None
    return b"".join(chunks)


def _next_url(resp: httpx.Response, url: httpx.URL) -> httpx.URL:
    target = resp.headers.get("location", "")
    if not target:
        raise _refuse("That link redirected without saying where.")
    return _parse(urljoin(str(url), target))


def _clean(url: httpx.URL) -> str:
    return str(url.copy_with(query=None, fragment=None, userinfo=b""))


def fetch_media(
    link: str, *, remote: bool, max_bytes: int = MAX_BYTES, timeout_s: float = TIMEOUT_S,
    resolver: Resolver | None = None, transport: Any = None, allowed_hosts: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FetchedMedia:
    """Fetch ``link`` (GET, https only) or raise ``MediaFetchRefused``."""
    plan = _Plan(remote, _allow_list(allowed_hosts), resolver or _system_resolver,
                 clock() + timeout_s, clock, max_bytes)
    url = _parse(link)
    try:
        with httpx.Client(transport=transport, trust_env=False, follow_redirects=False) as client:
            for _ in range(MAX_REDIRECTS + 1):
                resp = _open(client, url, plan)
                try:
                    if resp.status_code in _REDIRECTS:
                        url = _next_url(resp, url)
                        continue
                    if resp.status_code != 200:
                        raise _refuse("That link did not return an image.")
                    return FetchedMedia(_read_body(resp, plan), _clean(url),
                                        resp.headers.get("content-type", ""))
                finally:
                    resp.close()
    except MediaFetchRefused as refused:
        logger.info("image link refused for host %s", url.raw_host.decode("ascii", "replace"))
        raise refused
    raise _refuse("That link redirected too many times.")
