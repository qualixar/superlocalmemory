# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which remote key, if any, the current MCP request came in on.

The agent segment of ``/mcp/<agent>`` is chosen by the caller, so it is
attribution, never identity: a tool on another computer can send
``/mcp/claude`` as easily as the local Claude can. Anything that is private to
an agent on this computer (the ``slm_cache_*`` and ``slm_compress`` stores) must
therefore also be keyed by the remote key that authenticated the request.

:class:`server.remote_tool_policy.RemoteToolScopeASGI` sets the key id for
every request that carries a remote principal; local requests leave it unset.
Like the agent id (:mod:`mcp.agent_context`) it is a ContextVar, so concurrent
requests never see each other's value.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the grant type lives with the connection code; no runtime import
    from superlocalmemory.remote_connections.grant import RemoteGrant

_current_remote_key_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "slm_remote_key_id", default=None,
)


def current_remote_key_id() -> str | None:
    """The key id of the remote caller, or ``None`` for a caller on this computer."""
    return _current_remote_key_id.get()


@contextmanager
def remote_caller(key_id: str) -> Iterator[None]:
    """Mark everything run inside as coming from remote key ``key_id``."""
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("a remote caller needs a key id")
    token = _current_remote_key_id.set(key_id)
    try:
        yield
    finally:
        _current_remote_key_id.reset(token)


@dataclass(frozen=True)
class RemotePeer:
    """A web app calling through a remote connection, as the mesh sees it.

    Set around a mesh tool call that arrives with a verified grant. The mesh
    tools then serve it in process (see :class:`RemoteMeshTarget`) and never
    over the daemon's HTTP interface, where it could not be told from a session
    on this computer.

    ``peer_ref`` is a stable, opaque reference for the app (never a secret);
    ``app`` is its short name and ``display_name`` what the owner sees.
    """

    peer_ref: str
    app: str
    display_name: str


_current_remote_peer: contextvars.ContextVar[RemotePeer | None] = contextvars.ContextVar(
    "slm_remote_peer", default=None,
)


def current_remote_peer() -> RemotePeer | None:
    """The web app behind this request, or ``None`` for a caller on this computer."""
    return _current_remote_peer.get()


@contextmanager
def remote_peer(peer: RemotePeer | None) -> Iterator[None]:
    """Run everything inside as the given web app (``None`` clears it)."""
    token = _current_remote_peer.set(peer)
    try:
        yield
    finally:
        _current_remote_peer.reset(token)


@dataclass(frozen=True)
class RemoteMeshTarget:
    """Where a web app's mesh calls run: the daemon's own broker, in process.

    ``broker`` is the instance the daemon's mesh routes use (``None`` when the
    mesh is not running or is switched off), ``profile`` the profile of the
    remote key and ``connection_id`` the connection the app came in on.
    """

    broker: Any
    profile: str
    connection_id: str


_current_remote_mesh: contextvars.ContextVar[RemoteMeshTarget | None] = contextvars.ContextVar(
    "slm_remote_mesh", default=None,
)


def current_remote_mesh() -> RemoteMeshTarget | None:
    """The in-process mesh a web app's call runs in, or ``None``."""
    return _current_remote_mesh.get()


@contextmanager
def remote_mesh(target: RemoteMeshTarget | None) -> Iterator[None]:
    """Run everything inside against the given in-process mesh."""
    token = _current_remote_mesh.set(target)
    try:
        yield
    finally:
        _current_remote_mesh.reset(token)


_current_remote_grant: contextvars.ContextVar["RemoteGrant | None"] = contextvars.ContextVar(
    "slm_remote_grant", default=None,
)


def current_remote_grant() -> "RemoteGrant | None":
    """The verified grant for this request, or ``None``.

    Only :class:`remote_connections.origin.CanonicalMcpOrigin` sets it, after
    checking the gateway's signature; a local client cannot present one.
    """
    return _current_remote_grant.get()


@contextmanager
def remote_grant(grant: "RemoteGrant | None") -> Iterator[None]:
    """Run everything inside with the given verified grant."""
    token = _current_remote_grant.set(grant)
    try:
        yield
    finally:
        _current_remote_grant.reset(token)


__all__ = [
    "RemoteMeshTarget", "RemotePeer", "current_remote_grant", "current_remote_key_id",
    "current_remote_mesh", "current_remote_peer", "remote_caller", "remote_grant",
    "remote_mesh", "remote_peer",
]
