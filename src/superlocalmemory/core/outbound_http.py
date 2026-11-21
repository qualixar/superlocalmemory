# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one door memory text leaves this machine through.

SLM keeps credentials exactly as written; that is part of what it is for. The
boundary is the network. Every HTTP request whose body can carry memory text
(a fact, a query, a summary, entity text, chat context) is built here, and two
rules are applied on the URL the request is actually sent to:

* **Another machine** (a cloud provider, a LAN Ollama, anything that is not
  loopback) never receives a credential. Every string in the body passes
  ``outbound_redaction.for_endpoint``, which replaces each recognised
  credential with ``[redacted]``. A body that cannot be read as JSON is refused
  rather than sent unscreened.
* **This machine** sees the text as written, and is reached directly: an
  environment proxy (``HTTP_PROXY`` and friends) is never used, because a
  proxy would carry the unscreened text to whatever host it runs on.

Redirects are never followed, so the URL screened is the URL that receives
the body. ``tests/test_security/test_outbound_gate_guard.py`` fails the build
when a module outside this one opens an outbound connection that is not on its
reviewed list, so a new exit cannot quietly skip the screen.

``httpx`` is imported lazily: the stdlib-only hooks import this module too.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import urllib.request
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from superlocalmemory.core.outbound_redaction import for_endpoint, is_local_endpoint


class OutboundBlocked(ValueError):
    """A body that could not be screened was not sent to another machine."""


# -- the decision --------------------------------------------------------------


def outbound_json(payload: Any, url: str) -> Any:
    """A copy of a JSON-shaped ``payload`` as it may be sent to ``url``.

    Every string value is screened for another machine and left as written
    for this one. Keys, numbers, booleans and ``None`` are kept. The caller's
    object is never modified.
    """
    if is_local_endpoint(url):
        return payload
    return _screen(payload, url)


def _screen(value: Any, url: str) -> Any:
    if isinstance(value, str):
        return for_endpoint(value, url)
    if isinstance(value, Mapping):
        return {key: _screen(item, url) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_screen(item, url) for item in value]
    return value


def outbound_content(content: bytes | str | None, url: str) -> bytes | str | None:
    """Raw request bytes as they may be sent to ``url``.

    JSON is screened like :func:`outbound_json`; when nothing in it changes
    the original bytes are returned untouched, so a caller that measured or
    signed them still sends exactly what it measured. Anything that is not
    JSON is refused for another machine.
    """
    if not content or is_local_endpoint(url):
        return content
    try:
        # A body that is not UTF-8 cannot be JSON: refused like any other
        # unscreenable body, never a bare decode error from the caller.
        text = content.decode("utf-8") if isinstance(content, bytes) else content
        parsed = json.loads(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise OutboundBlocked(
            "refusing to send a body that is not JSON to another machine; "
            "it cannot be screened for credentials"
        ) from exc
    screened = _screen(parsed, url)
    if screened == parsed:
        return content
    encoded = json.dumps(screened, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False)
    return encoded.encode("utf-8") if isinstance(content, bytes) else encoded


def transport_options(url: str) -> dict[str, bool]:
    """httpx client options for ``url``: never a proxy for this machine,
    never a redirect for any machine."""
    return {"trust_env": not is_local_endpoint(url), "follow_redirects": False}


def _gated_body(url: str, json_body: Any, content: Any) -> dict[str, Any]:
    if json_body is not None and content is not None:
        raise ValueError("pass json or content, not both")
    if json_body is not None:
        return {"json": outbound_json(json_body, url)}
    if content is not None:
        return {"content": outbound_content(content, url)}
    return {}


# -- httpx: one-shot ----------------------------------------------------------


def post_json(url: str, payload: Any, *, headers: Mapping[str, str] | None = None,
              timeout: Any = 30.0) -> Any:
    """POST ``payload`` as JSON to ``url`` through the gate. Returns the
    ``httpx.Response``; raises what httpx raises."""
    import httpx

    return httpx.post(url, json=outbound_json(payload, url), headers=headers,
                      timeout=timeout, **transport_options(url))


# -- httpx: a reusable client ---------------------------------------------------


class GatedClient:
    """A reusable connection pool whose every body passes the gate.

    It holds at most two ``httpx.Client`` objects — one that may use the
    user's proxy for other machines, one that never does for this machine —
    built lazily under a lock. It exposes no way to send an unscreened body.
    """

    def __init__(self, *, timeout: Any = None, transport: Any = None,
                 verify: Any = True) -> None:
        self._timeout = timeout
        self._transport = transport
        self._verify = verify
        self._lock = threading.Lock()
        self._clients: dict[bool, Any] = {}
        self._closed = False

    def _client_for(self, url: str) -> Any:
        import httpx

        options = transport_options(url)
        with self._lock:
            if self._closed:
                raise RuntimeError("the outbound client has been closed")
            client = self._clients.get(options["trust_env"])
            if client is None:
                kwargs: dict[str, Any] = dict(options, verify=self._verify)
                if self._timeout is not None:
                    kwargs["timeout"] = self._timeout
                if self._transport is not None:
                    kwargs["transport"] = self._transport
                client = httpx.Client(**kwargs)
                self._clients[options["trust_env"]] = client
            return client

    def post(self, url: str, *, json: Any = None, content: Any = None,  # noqa: A002
             headers: Mapping[str, str] | None = None, **kwargs: Any) -> Any:
        body = _gated_body(url, json, content)
        return self._client_for(url).post(url, headers=headers, **body, **kwargs)

    @contextlib.contextmanager
    def stream(self, method: str, url: str, *, json: Any = None,  # noqa: A002
               content: Any = None, headers: Mapping[str, str] | None = None,
               **kwargs: Any) -> Iterator[Any]:
        body = _gated_body(url, json, content)
        with self._client_for(url).stream(method, url, headers=headers,
                                          **body, **kwargs) as response:
            yield response

    def close(self) -> None:
        with self._lock:
            self._closed = True
            clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            client.close()


# -- httpx: async streaming -------------------------------------------------------


class GatedAsyncClient:
    """Reusable async pools whose requests and streams cannot skip screening."""

    def __init__(self, *, timeout: Any = None, limits: Any = None,
                 transport: Any = None, verify: Any = True) -> None:
        self._timeout = timeout
        self._limits = limits
        self._transport = transport
        self._verify = verify
        self._clients: dict[bool, Any] = {}
        self._closed = False

    def _client_for(self, url: str) -> Any:
        import httpx

        if self._closed:
            raise RuntimeError("the outbound client has been closed")
        options = transport_options(url)
        client = self._clients.get(options["trust_env"])
        if client is None:
            kwargs: dict[str, Any] = dict(options, verify=self._verify)
            for name, value in (("timeout", self._timeout), ("limits", self._limits),
                                ("transport", self._transport)):
                if value is not None:
                    kwargs[name] = value
            client = httpx.AsyncClient(**kwargs)
            self._clients[options["trust_env"]] = client
        return client

    @staticmethod
    def _arguments(url: str, json_body: Any, content: Any,
                   headers: Mapping[str, str] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "data" in kwargs or "files" in kwargs:
            raise OutboundBlocked("outbound bodies must use screened JSON or content")
        body = _gated_body(url, json_body, content)
        # Redaction can change the byte count. Let httpx compute the length of
        # the screened body; never reuse the caller's original measurement.
        filtered = {k: v for k, v in (headers or {}).items() if k.lower() != "content-length"}
        return dict(kwargs, **body, headers=filtered, follow_redirects=False)

    async def request(self, method: str, url: str, *, json: Any = None,  # noqa: A002
                      content: Any = None, headers: Mapping[str, str] | None = None,
                      **kwargs: Any) -> Any:
        arguments = self._arguments(url, json, content, headers, kwargs)
        return await self._client_for(url).request(method, url, **arguments)

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await self.request("POST", url, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await self.request("GET", url, **kwargs)

    @contextlib.asynccontextmanager
    async def stream(self, method: str, url: str, *, json: Any = None,  # noqa: A002
                     content: Any = None, headers: Mapping[str, str] | None = None,
                     **kwargs: Any) -> AsyncIterator[Any]:
        arguments = self._arguments(url, json, content, headers, kwargs)
        async with self._client_for(url).stream(method, url, **arguments) as response:
            yield response

    async def aclose(self) -> None:
        self._closed = True
        clients, self._clients = list(self._clients.values()), {}
        results = await asyncio.gather(*(client.aclose() for client in clients),
                                       return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result


@contextlib.asynccontextmanager
async def astream_json(method: str, url: str, payload: Any, *,
                       headers: Mapping[str, str] | None = None,
                       timeout: Any = 120.0, transport: Any = None) -> AsyncIterator[Any]:
    """Stream a JSON request through the gate; yields the ``httpx.Response``."""
    import httpx

    kwargs: dict[str, Any] = dict(transport_options(url), timeout=timeout)
    if transport is not None:
        kwargs["transport"] = transport
    async with httpx.AsyncClient(**kwargs) as client:
        async with client.stream(method, url, json=outbound_json(payload, url),
                                 headers=headers) as response:
            yield response


# -- urllib (the stdlib-only callers) -----------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Turns every 3xx into an ``HTTPError`` carrying that status.

    urllib's default handler re-sends a request, headers included, to
    wherever the ``Location`` points. Returning ``None`` makes it stop.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _opener(*, direct: bool) -> urllib.request.OpenerDirector:
    # Built per call: ProxyHandler() reads the environment when constructed.
    proxies = urllib.request.ProxyHandler({}) if direct else urllib.request.ProxyHandler()
    return urllib.request.build_opener(proxies, _NoRedirect())


def urlopen(request: urllib.request.Request | str, *, timeout: float) -> Any:
    """``urllib.request.urlopen`` through the gate.

    The body is screened for another machine. For this machine an
    environment or system proxy is never used. A redirect is never followed
    for any machine: it is raised as ``urllib.error.HTTPError`` with the 3xx
    status, so headers such as the hook token never reach a second origin.
    """
    req = request if isinstance(request, urllib.request.Request) else (
        urllib.request.Request(request))
    url = req.full_url
    data = req.data
    if data is not None and not is_local_endpoint(url):
        screened = outbound_content(data, url)
        if screened is not data:
            req = urllib.request.Request(
                url, data=screened, headers=dict(req.header_items()),
                method=req.get_method(),
            )
    return _send(req, timeout=timeout)


def _send(req: urllib.request.Request, timeout: float) -> Any:
    """Open ``req`` with no redirects; direct (no proxy) for this machine."""
    return _opener(direct=is_local_endpoint(req.full_url)).open(req, timeout=timeout)


__all__ = [
    "GatedAsyncClient",
    "GatedClient",
    "OutboundBlocked",
    "astream_json",
    "outbound_content",
    "outbound_json",
    "post_json",
    "transport_options",
    "urlopen",
]


def local_ollama_connection(timeout: float = 30):
    """Cancellable stdlib connection to the fixed local generator.

    This transport never uses environment proxies or follows redirects. Its
    destination cannot be supplied by a caller; memory remains on loopback.
    The socket stays available to the calibration cancellation owner.
    """
    import http.client
    return http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout)
