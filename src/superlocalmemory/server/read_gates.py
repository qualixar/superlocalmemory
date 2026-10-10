# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Read permission for dashboard and mesh reads when team accounts are on.

Moved out of ``unified_daemon.py`` (which re-exports the old private names) in
4.1.20, when two gaps were closed:

* **Reads sent as POST.** The READ check ran for GET only, so the dashboard
  search (``POST /api/search``) and the memory chat
  (``POST /api/v3/chat/stream``) returned memory to a caller without READ on
  the workspace. :data:`READ_ONLY_POST_PATHS` puts them under the same check.
* **Mesh reads.** ``GET /mesh/*`` proved machine access only; a signed-in user
  without READ, or anyone in company mode without a session, could read peers,
  inboxes, shared state and locks, and ``?profile=`` reached any workspace.
  :func:`mesh_read_gate` applies READ on the workspace the request names.

Every gate is a no-op until team accounts exist (single-operator installs are
unchanged) and fails closed (503) when the account store cannot be read.
"""

from __future__ import annotations

import hmac

_SENSITIVE_READ_PREFIXES = (
    "/api/memories", "/api/facts", "/api/clusters", "/api/graph",
    "/api/v3/associations", "/api/v3/core-memory",
    "/api/v3/soft-prompts", "/api/v3/dashboard", "/api/v3/mode",
    "/api/v3/embedding/config", "/api/v3/scope/config",
    "/api/v3/storage/config", "/api/v3/daemon/config",
    "/api/v3/mesh/config", "/api/v3/trust/config",
    "/api/v3/forgetting/config", "/api/v3/mcp/profiles",
    "/api/learning", "/api/behavioral",
    # Event stream, agent activity, trust signals, and v3 profiling data
    # expose cross-agent coordination signals and behavioral profiles.
    "/events", "/api/events", "/api/agents", "/api/trust/",
    "/api/v3/abstraction", "/api/v3/insights",
    "/api/v3/answer-check/history",
    # The timeline lists memory text by date.
    "/api/v3/timeline",
    # 4.1.21 (#113): summaries quote memories and their pickers name projects
    # and sessions; saved views hold a person's queries and run recall. Prefix,
    # so /api/summary/projects and /api/summary/sessions are covered too —
    # the project list was an exact-path miss before.
    "/api/summary", "/api/v3/views",
    # 4.1.25 audit: these answered a session-less caller in company mode.
    # Learning and trust counters, agent trust scores, the feature switches,
    # compliance and audit records, workspace names, lifecycle and tier
    # counts, adapter state, provider and model settings, and setup status.
    "/api/v3/learning", "/api/v3/trust", "/api/v3/features",
    "/api/compliance", "/api/lifecycle", "/api/tiers", "/api/profiles",
    "/api/adapters", "/api/v3/answer-check", "/api/v3/auto",
    "/api/v3/graph", "/api/v3/runtime", "/api/v3/provider",
    "/api/v3/hooks", "/api/v3/ide", "/api/v3/math", "/api/v3/ollama",
    "/api/v3/components",
)
_SENSITIVE_READ_EXACT_PATHS = (
    "/api/search", "/api/v3/recall/trace", "/api/patterns",
    "/api/feedback/stats", "/api/stats", "/api/timeline",
    # L3-01: project/agent names and per-bucket memory counts — the same
    # cross-tenant metadata the prefixes above already gate.
    "/api/v3/facets",
)
#: POST routes that only read and return memory content (4.1.20).
READ_ONLY_POST_PATHS = frozenset({"/api/search", "/api/v3/chat/stream"})


def is_sensitive_dashboard_read(method: str, path: str) -> bool:
    if method == "POST":
        return path in READ_ONLY_POST_PATHS
    return (
        method == "GET"
        and (
            path.startswith(_SENSITIVE_READ_PREFIXES)
            or path in _SENSITIVE_READ_EXACT_PATHS
            or path.startswith("/api/v3/recall")
        )
    )


#: Status reads the command line makes (``slm features``). They hold no memory content,
#: so the daemon capability, which only this computer's own user can read, is enough
#: for them even where every person must sign in. Anyone else needs a session.
CLI_STATUS_READ_PREFIXES = ("/api/v3/features",)


def is_cli_status_read(method: str, path: str) -> bool:
    return method == "GET" and path.startswith(CLI_STATUS_READ_PREFIXES)


def _json(status: int, message: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"error": message})


def _session_token(request) -> str:
    return (request.headers.get("x-slm-user-session", "")
            or (request.cookies.get("slm_session", "") if request.cookies else ""))


def _rbac_active(rbac) -> bool | None:
    """True/False, or None when the account store could not be read."""
    try:
        return rbac.user_count() > 0
    except Exception:  # noqa: BLE001
        return None


def rbac_read_gate(request, app_state, *, profile: str | None = None,
                   machine_principal: bool = False):
    """READ on ``profile`` (default: the active one). ``None`` allows; else the
    refusal response.

    ``machine_principal`` is a caller that proved it is one of this computer's
    own agents (the daemon capability) or a mesh node (the shared secret):
    a program, not a person, so company mode's "every person signs in" does
    not apply to it. A session it presents is still checked.
    """
    rbac = getattr(app_state, "rbac", None)
    if rbac is None:
        return None
    active = _rbac_active(rbac)
    if active is None:
        # Fail CLOSED: if we cannot determine RBAC state we must not silently
        # allow reads (a DB error would otherwise open the whole read surface).
        return _json(503, "authorization temporarily unavailable")
    if not active:
        return None  # single-operator install — reads are open
    token = _session_token(request)
    user = rbac.resolve_session(token) if token else None
    if user is None:
        if rbac.require_login() and not machine_principal:
            return _json(401, "Login required to read memory.")
        return None  # owner/operator, personal mode
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.server.routes.helpers import get_active_profile

    if rbac.has_permission(user["user_id"], profile or get_active_profile(),
                           Permission.READ):
        return None
    return _json(403, "Your role cannot read this workspace.")


# -- mesh ---------------------------------------------------------------------------


def _capability_ok(request, app_state) -> bool:
    descriptor = getattr(app_state, "daemon_descriptor", None)
    presented = request.headers.get("x-slm-daemon-capability", "")
    if descriptor is None or not presented:
        return False
    return (hmac.compare_digest(presented, str(getattr(descriptor, "capability", "")))
            and hmac.compare_digest(request.headers.get("x-slm-target-instance", ""),
                                    str(getattr(descriptor, "instance_id", ""))))


def status_details_allowed(request, app_state) -> bool:
    """Whether ``GET /status`` may carry paths and counts for this caller.

    Where every person must sign in, a caller with no session (and without the
    daemon's private capability, which the command line and MCP send) gets the
    discovery fields only. Everywhere else the answer is unchanged.
    """
    return rbac_read_gate(
        request, app_state, machine_principal=_capability_ok(request, app_state),
    ) is None


def mesh_read_gate(request, app_state):
    """READ for ``GET /mesh/*``. ``None`` allows; else the refusal response.

    The workspace is the one the request names (``?profile=``) or the active
    one. A mesh node authenticated by the shared secret reads only the
    workspace this node serves: the fleet secret is shared by every node, so
    it must not open every workspace on the machine.
    """
    if request.method != "GET" or not request.url.path.startswith("/mesh/"):
        return None
    from superlocalmemory.server.access_gate import mesh_secret_ok
    from superlocalmemory.server.routes.helpers import get_active_profile

    active_profile = get_active_profile()
    target = (request.query_params.get("profile", "") or "").strip() or active_profile
    headers = {k.lower(): v for k, v in request.headers.items()}
    fleet_node = mesh_secret_ok(headers, app_state)
    if fleet_node and not _session_token(request) and target != active_profile:
        return _json(403, "A mesh node may read only the workspace this node serves.")
    return rbac_read_gate(
        request, app_state, profile=target,
        machine_principal=fleet_node or _capability_ok(request, app_state),
    )


__all__ = [
    "CLI_STATUS_READ_PREFIXES",
    "READ_ONLY_POST_PATHS",
    "is_cli_status_read",
    "is_sensitive_dashboard_read",
    "mesh_read_gate",
    "rbac_read_gate",
    "status_details_allowed",
]
