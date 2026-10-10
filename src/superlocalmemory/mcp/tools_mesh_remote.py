# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The mesh tools for a connected web app, run in process.

A web app's call carries a verified grant, which :mod:`server.remote_tool_policy`
turns into a :class:`~superlocalmemory.mcp.remote_caller.RemotePeer` and a
:class:`~superlocalmemory.mcp.remote_caller.RemoteMeshTarget`. These functions
use the daemon's own broker directly, so the app's identity is never sent over
the daemon's loopback HTTP interface, where it could be lost or forged. A web
app never falls back to the local session: with no broker it is refused.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from superlocalmemory.mcp.remote_caller import RemoteMeshTarget, RemotePeer, current_remote_mesh
from superlocalmemory.mesh.envelope import PREFACE

WAIT_MIN_S = 1
WAIT_MAX_S = 20
MAX_STATE_KEY = 256

_NOT_AVAILABLE = {"ok": False, "error": "mesh is not available"}
_TOO_MANY_WAITS = {"ok": False, "error": "too many waits, retry shortly"}
_STATE_READ_ONLY = {"ok": False, "error":
                    "remote access can only read mesh state, and needs a key"}


def _hide(message: dict) -> dict:
    """A message as a web app sees it: no project path, which is a path on this computer."""
    return {k: v for k, v in message.items() if k != "project_path"}


def _join(target: RemoteMeshTarget, peer: RemotePeer) -> dict:
    """Register the app (or refresh it); the answer is ``{"ok": True}`` or a refusal."""
    return target.broker.ensure_web_peer(
        peer.peer_ref, app=peer.app, display_name=peer.display_name,
        connection_id=target.connection_id, profile_id=target.profile)


def _clamp_wait(timeout_s: object) -> int:
    try:
        return max(WAIT_MIN_S, min(int(timeout_s), WAIT_MAX_S))  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return WAIT_MIN_S


def _peers(target: RemoteMeshTarget, peer: RemotePeer) -> dict:
    joined = _join(target, peer)
    if not joined.get("ok"):
        return joined
    listed = target.broker.list_peer_directory(target.profile)
    return {"peers": listed, "count": len(listed), "my_peer_id": peer.peer_ref}


def _send(target: RemoteMeshTarget, peer: RemotePeer, to: str, message: str,
          refs: list[str] | None, reply_to: int | None) -> dict:
    joined = _join(target, peer)
    if not joined.get("ok"):
        return joined
    return target.broker.web_send(
        peer.peer_ref, peer.app, to, message, refs=list(refs or ()),
        reply_to=reply_to, profile_id=target.profile)


def _inbox(target: RemoteMeshTarget, peer: RemotePeer) -> dict:
    joined = _join(target, peer)
    if not joined.get("ok"):
        return joined
    msgs = [_hide(m) for m in target.broker.claim_web_inbox(peer.peer_ref, target.profile)]
    return {"messages": msgs, "count": len(msgs), "unread": len(msgs), "preface": PREFACE}


def _wait(target: RemoteMeshTarget, peer: RemotePeer, timeout_s: int) -> dict:
    joined = _join(target, peer)
    if not joined.get("ok"):
        return joined
    try:
        found, timed_out = target.broker.wait_web_inbox(
            peer.peer_ref, timeout_s=timeout_s, profile_id=target.profile)
    except RuntimeError:
        return dict(_TOO_MANY_WAITS)
    msgs = [_hide(m) for m in found]
    return {"messages": msgs, "count": len(msgs), "timed_out": timed_out, "preface": PREFACE}


def _state(target: RemoteMeshTarget, peer: RemotePeer, key: str, action: str) -> dict:
    if action != "get" or not isinstance(key, str) or not key.strip() or len(key) > MAX_STATE_KEY:
        return dict(_STATE_READ_ONLY)
    joined = _join(target, peer)
    if not joined.get("ok"):
        return joined
    entry = target.broker.get_state_key(key, profile_id=target.profile)
    if entry is None:
        return {"key": key, "value": None}
    return {k: entry[k] for k in ("key", "value", "set_by", "updated_at")}


async def _run(work: Callable[[RemoteMeshTarget], Any]) -> Any:
    """Run ``work`` in a worker thread against the daemon's broker, or refuse."""
    target = current_remote_mesh()
    if target is None or target.broker is None:
        return dict(_NOT_AVAILABLE)
    return await asyncio.to_thread(work, target)


async def peers(peer: RemotePeer) -> dict:
    return await _run(lambda t: _peers(t, peer))


async def send(peer: RemotePeer, to: str, message: str, refs: list[str] | None,
               reply_to: int | None) -> dict:
    return await _run(lambda t: _send(t, peer, to, message, refs, reply_to))


async def inbox(peer: RemotePeer) -> dict:
    return await _run(lambda t: _inbox(t, peer))


async def wait(peer: RemotePeer, timeout_s: object) -> dict:
    """The broker wait holds a worker thread, bounded by the broker's own cap."""
    seconds = _clamp_wait(timeout_s)
    return await _run(lambda t: _wait(t, peer, seconds))


async def state(peer: RemotePeer, key: str, action: str) -> dict:
    return await _run(lambda t: _state(t, peer, key, action))
