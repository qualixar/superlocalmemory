# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The one decision door: who decided, through which channel (4.2.30, D-49, I-10).

Every route that records a decision (firewall promotions, proposal sets, dispute settlements,
audit labels, and their undo) asks this module who the caller is. Nothing here asks whether a
person is behind the caller: any local caller may decide, and a caller from another machine is
governed by its role through the ordinary permission check the route already made
(``rbac_enforce.require_permission``: a viewer cannot write, in any mode).

* **Locality comes from the connection**, the peer address of the request, never from a
  parameter the caller sends. A remote caller is recorded on the ``http`` channel whatever
  header it sets, and as ``api-key:<fingerprint>``.
* **The actor** is the connection's agent when a capability holder names it
  (``agent:<name>``), ``dashboard`` for the dashboard page's own door (the install token plus
  the page's channel header, local only), else a client name the caller gave
  (``client:<name>``) or ``client:local``. The name is a label that is recorded and shown, not a
  credential and not a claim about a person.
* **The channel** is ``dashboard`` only for the dashboard's own door; otherwise the channel the
  client states (``mcp`` or ``cli``, sent by SuperLocalMemory's own clients) or ``http``.

Ids and names only; no memory text.
"""

from __future__ import annotations

from fastapi import Request

from superlocalmemory.core.agent_attribution import CHANNEL_HEADER as _CHANNEL_HEADER
from superlocalmemory.core.decisions import Decider, clean_actor, clean_channel
from superlocalmemory.server.loopback import is_loopback
from superlocalmemory.server.write_identity import _TEST_ISOLATION_ALLOWED

CHANNEL_HEADER = _CHANNEL_HEADER  # which of our own doors made the call; grants nothing
_LOCAL_ACTOR_PREFIX = "local-capability:"
_CAPABILITY_PREFIX = "daemon-capability:"


def peer_is_local(request: Request) -> bool:
    """True when the connection's peer address is on this computer (never a parameter)."""
    host = request.client.host if request.client else ""
    if host == "testclient" and _TEST_ISOLATION_ALLOWED:
        return True
    return is_loopback(host)


def _connection_agent(request: Request, trusted_actor_id: str) -> str:
    """The agent a capability holder vouches for (``X-SLM-Connection-Agent``), else ``""``."""
    from superlocalmemory.core.agent_attribution import CONNECTION_HEADER
    from superlocalmemory.mcp.agent_context import sanitize_agent_id

    if not str(trusted_actor_id or "").startswith(_CAPABILITY_PREFIX):
        return ""
    name = (request.headers.get(CONNECTION_HEADER, "") or "").strip()
    return sanitize_agent_id(name) if name else ""


def decider_for(request: Request, trusted_actor_id: str, claimed: str = "") -> Decider:
    """Who is deciding, from the connection and the credential, never from a locality parameter."""
    actor_id = str(trusted_actor_id or "")
    if not peer_is_local(request):
        fingerprint = actor_id.rsplit(":", 1)[-1][:12]
        return Decider(clean_actor(f"api-key:{fingerprint}" if fingerprint else "", "api-key"),
                       "http")
    stated = clean_channel(request.headers.get(CHANNEL_HEADER, ""), "")
    if stated == "dashboard" and actor_id.startswith(_LOCAL_ACTOR_PREFIX):
        return Decider("dashboard", "dashboard")
    agent = _connection_agent(request, actor_id)
    if agent:
        return Decider(clean_actor(f"agent:{agent}"), stated if stated in ("mcp", "cli") else "http")
    name = clean_actor(claimed, "")
    return Decider(f"client:{name}" if name else "client:local",
                   stated if stated in ("mcp", "cli") else "http")


__all__ = ["CHANNEL_HEADER", "decider_for", "peer_is_local"]
