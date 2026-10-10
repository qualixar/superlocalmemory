# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Keep every remote tool call inside the one profile its key is bound to.

A remote key is bound to one profile (:mod:`server.remote_keys`). Every
remote-callable tool is in exactly one of three groups:

* :data:`ROUTED_TOOLS` are served for the key's profile through the
  per-request profile path, whatever profile this computer is using. The
  wrapper sets ``profile_id`` to the key's profile; the host's active profile
  is neither read for the work nor moved. Since 4.1.21 this is every
  remote-callable tool that reads or writes a profile's memory. Each takes
  ``profile_id`` and serves it either directly (``mcp/request_profile``) or
  through a daemon route that accepts it, authorized on THAT profile:
  ``/recall``, ``/remember``, ``DELETE``/``PATCH /api/memories/{id}``,
  ``/api/corrections``, ``/api/memory-kinds`` and ``/api/v3/views/run``. A
  write through a daemon route reaches the canonical writer only for the
  mutation kinds ``core/mutation_routing`` lets go to a non-active profile.
* :data:`PROFILE_FREE_TOOLS` touch no profile's memory (the key's own cache and
  compression store, the version, the machine-wide mode, the product
  attribution). They always run.
* :data:`ACTIVE_ONLY_TOOLS` would work on the active profile only. None is left
  in 4.1.21; the group stays because it is also where a new remote tool lands
  until it is routed (default deny for profile choice, see
  ``tests/test_security/test_remote_profile_binding.py``). Such a tool runs
  only while the key's profile is the active one, under a profile lease held
  for the whole call (a switch waits for it). Otherwise it is refused with
  ``remote_profile_not_active``: the host is using another workspace right
  now; try again later or ask the host owner. The refusal does not name that
  workspace.

Writes keep every check a local write for that profile has: only a write key
reaches them (``server/remote_tool_policy``); ``@admits`` and the daemon routes
check the caller's role on the routed profile, not the active one; the
canonical writer rechecks the profile in its own transaction; and each call is
logged with the remote key and its profile (``remote_tool_policy._audit``).

Three rules hold for every call, read key or write key alike:

1. **The work happens in the bound profile** (routed, or active-only as above).
2. **A profile named in the arguments must be the bound one.** ``profile_id``
   anywhere in the arguments, including inside structured values such as
   ``payload``, is compared with the key's profile. A different profile is
   refused, never rewritten.
3. **A remote save stays in the bound profile.** ``scope`` other than
   ``personal`` and a non-empty ``shared_with`` are refused, and a save that
   names no scope is made ``personal`` explicitly, so a host whose default
   scope is ``shared`` or ``global`` cannot turn a remote save into a memory
   other profiles recall. ``include_shared`` / ``include_global`` on a recall
   are allowed: the recall runs as the bound profile, so they reach only what
   other profiles chose to share with it or with every profile.

The mesh tools a connected web app may call (``remote_tool_policy.MESH_TOOLS``)
take no ``profile_id``: the mesh is per profile, and they run in process against
the key's profile (``mcp/remote_caller.RemoteMeshTarget``), with no lease and no
active-profile check, so a key bound to one profile reaches only that profile's
peers, messages and shared state.

Every argument a remote-callable tool takes is classified below.
``tests/test_security/test_remote_profile_binding.py`` reads the live tool
registry and fails when a tool gains an argument that is not classified here,
and an unclassified argument is refused at run time (default deny).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

#: Arguments that name a profile.
PROFILE_ARGUMENTS: frozenset[str] = frozenset({"profile_id"})

#: Arguments that widen a recall to memories other profiles shared with the
#: bound profile or made global. Bounded by the profile the call runs as.
READ_SCOPE_ARGUMENTS: frozenset[str] = frozenset({"include_shared", "include_global"})

#: Arguments that would make a saved memory visible to other profiles.
WRITE_SCOPE_ARGUMENTS: frozenset[str] = frozenset({"scope", "shared_with"})

#: The remote-callable tools that take ``scope``; each remote save through them
#: is pinned to ``personal``. The registry test keeps this list complete.
SCOPED_WRITE_TOOLS: frozenset[str] = frozenset({"remember", "remember_media", "remember_document"})

#: Served for the key's profile by the per-request profile path (4.1.19), with
#: the host's active profile untouched.
ROUTED_TOOLS: frozenset[str] = frozenset({
    "recall", "remember", "list_corrections", "review_correction",
    # 4.1.21: everything else a remote agent uses on its own profile.
    "search", "fetch", "list_recent", "prestage_context", "recall_trace",
    "session_init", "close_session", "observe", "update_memory", "delete_memory",
    "core_memory", "get_status", "health", "memory_used", "get_memory_summary",
    "get_lifecycle_status", "get_retention_stats", "get_soft_prompts",
    "get_learned_patterns", "correct_pattern", "get_behavioral_patterns",
    "report_feedback", "report_outcome", "settle_session_outcomes", "log_tool_event",
    "get_assertions", "reinforce_assertion", "contradict_assertion",
    "set_memory_kind", "memory_kinds_status", "review_memory_kinds",
    "confirm_memory_kinds", "run_view", "manage_view", "skill_health", "skill_lineage",
    "slm_loop_history", "slm_loop_show", "get_brain_evidence_status",
    # 4.1.25: images and documents, for a key that opted in (remote_tool_policy.MEDIA_TOOLS).
    "remember_media", "get_media", "remember_document", "media_status",
    "record_agent_experience", "record_cognitive_turn", "finalize_cognitive_turn",
})

#: Remote-callable tools that take ``profile_id``. Each gets the key's profile
#: written in explicitly.
PROFILE_ARGUMENT_TOOLS: frozenset[str] = ROUTED_TOOLS

#: Read or write no profile's memory: a key's own cache and compression store
#: (keyed by the key, see mcp/remote_caller), aggregate counters, the version,
#: the machine-wide mode (one config for every profile) and the product's
#: fixed attribution.
PROFILE_FREE_TOOLS: frozenset[str] = frozenset({
    "get_version", "slm_cache_get", "slm_cache_set", "slm_compress", "slm_retrieve",
    "slm_optimize_stats", "get_mode", "get_attribution",
})

#: Remote-callable tools that can only serve the active profile, each with the
#: reason it cannot be routed. Empty in 4.1.21.
ACTIVE_ONLY_TOOLS: dict[str, str] = {}

#: Every other argument of a remote-callable tool. None selects a profile.
NEUTRAL_ARGUMENTS: frozenset[str] = frozenset({
    "about", "action", "agent_id", "as_of", "assertion_id", "case_id", "category",
    "ccr_id", "content", "context", "correction", "duration_ms", "event_type",
    "event_valid_until", "expected_version", "fact_id", "fact_ids", "fast", "feedback",
    "filters",
    "finalize", "idempotency_key", "importance", "include_history", "include_unknown",
    "input_summary", "items", "key", "kind", "known_as_of", "limit", "max_age_days",
    "max_results", "memory_ids", "message", "metadata", "min_confidence", "mode", "name",
    "new_name", "offset", "outcome",
    "output_summary", "pattern_id", "pattern_type", "payload", "prefer_project", "project",
    "project_strict",
    "project_path", "query", "recall_query_id", "receipt_id", "refs", "replaces",
    "reply_to", "reversible", "run_id",
    "saved_by", "session_date", "session_id", "skill_name", "tags", "tags_match", "target",
    "timeout_s", "to", "tool_name", "ttl_seconds", "valid_at", "value", "window",
})

#: Arguments only the image and document tools take. Accepted for those tools alone, so a
#: later tool with a ``path`` argument is never let through by accident.
MEDIA_ARGUMENTS: frozenset[str] = frozenset({
    "base64", "download_url", "file_name", "job_id", "media_id", "path", "variant",
})
MEDIA_ARGUMENT_TOOLS: frozenset[str] = frozenset({
    "remember_media", "get_media", "remember_document", "media_status",
})

CLASSIFIED_ARGUMENTS: frozenset[str] = (
    PROFILE_ARGUMENTS | READ_SCOPE_ARGUMENTS | WRITE_SCOPE_ARGUMENTS | NEUTRAL_ARGUMENTS
    | MEDIA_ARGUMENTS
)

PROFILE_DENIAL = "remote_profile_not_allowed"
INACTIVE_DENIAL = "remote_profile_not_active"
SCOPE_DENIAL = "remote_scope_not_allowed"
ARGUMENT_DENIAL = "remote_argument_not_allowed"

#: Nested values are searched this deep for a ``profile_id``; deeper is refused.
_MAX_DEPTH = 16


class BindingRefusal(ValueError):
    """The call would leave the key's profile. ``code`` is stable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{message} [{code}]")
        self.code = code


def _names_bound_profile(value: object, bound: str) -> bool:
    """Absent, empty and the bound profile are fine; anything else is not."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    named = value.strip()
    return not named or named == bound


def _nested_profiles(value: object, depth: int = 0) -> Iterator[object]:
    """Every ``profile_id`` value inside a structured argument."""
    if depth > _MAX_DEPTH:
        raise BindingRefusal(ARGUMENT_DENIAL, "An argument is nested too deeply.")
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name in PROFILE_ARGUMENTS:
                yield item
            yield from _nested_profiles(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_profiles(item, depth + 1)


def _profile_refusal(key_name: str, bound: str) -> BindingRefusal:
    return BindingRefusal(
        PROFILE_DENIAL,
        f"Remote key '{key_name}' is bound to profile '{bound}' and cannot reach "
        "another profile.")


def _check_media_argument(tool: str, name: str, value: Any) -> None:
    """Image and document arguments belong to those tools only, and a remote app can
    never name a file on this computer."""
    if name not in MEDIA_ARGUMENTS:
        return
    if tool not in MEDIA_ARGUMENT_TOOLS:
        raise BindingRefusal(
            ARGUMENT_DENIAL, f"Argument '{name}' is not accepted over remote access.")
    if name == "path" and value not in (None, ""):
        raise BindingRefusal(
            ARGUMENT_DENIAL, "Remote apps cannot name a file on this computer.")


def _check_mesh_state(arguments: Mapping[str, Any]) -> None:
    """A remote caller may only read a shared key, never write or delete one."""
    key = arguments.get("key")
    if arguments.get("action", "get") != "get":
        raise BindingRefusal(ARGUMENT_DENIAL, "Remote access can only read mesh state.")
    if not isinstance(key, str) or not key.strip() or len(key) > 256:
        raise BindingRefusal(ARGUMENT_DENIAL, "mesh_state needs a key of 1 to 256 characters.")


def bind_arguments(tool: str, arguments: object, *, key_name: str,
                   bound: str) -> dict[str, Any]:
    """The arguments to run ``tool`` with as profile ``bound``.

    Raises :class:`BindingRefusal` when the call would leave that profile.
    """
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise BindingRefusal(ARGUMENT_DENIAL, "Tool arguments must be a JSON object.")
    if tool == "mesh_state":
        _check_mesh_state(arguments)
    for name, value in arguments.items():
        if name not in CLASSIFIED_ARGUMENTS:
            raise BindingRefusal(
                ARGUMENT_DENIAL, f"Argument '{name}' is not accepted over remote access.")
        _check_media_argument(tool, name, value)
        if name in PROFILE_ARGUMENTS and not _names_bound_profile(value, bound):
            raise _profile_refusal(key_name, bound)
        if name == "scope" and value not in (None, "", "personal"):
            raise BindingRefusal(
                SCOPE_DENIAL,
                f"Remote key '{key_name}' saves only to its own profile '{bound}'; "
                "scope must be 'personal'. Share a memory from the SLM computer.")
        if name == "shared_with" and value not in (None, "", [], ()):
            raise BindingRefusal(
                SCOPE_DENIAL,
                f"Remote key '{key_name}' saves only to its own profile '{bound}'; "
                "it cannot share a memory with other profiles.")
        if any(not _names_bound_profile(v, bound) for v in _nested_profiles(value)):
            raise _profile_refusal(key_name, bound)
    bound_arguments = dict(arguments)
    if tool in PROFILE_ARGUMENT_TOOLS:
        bound_arguments["profile_id"] = bound
    if tool in SCOPED_WRITE_TOOLS and not bound_arguments.get("scope"):
        bound_arguments["scope"] = "personal"
    return bound_arguments


def inactive_refusal(tool: str, key_name: str, bound: str) -> BindingRefusal:
    return BindingRefusal(
        INACTIVE_DENIAL,
        f"The SLM computer is using another workspace right now, so '{tool}' cannot "
        f"run for remote key '{key_name}' (profile '{bound}'). Try again later or ask "
        f"the host owner. Meanwhile recall, remember and the other memory tools "
        f"keep working for profile '{bound}'.")


def runtime_from_scope(scope: Mapping[str, Any]) -> Any | None:
    """The daemon's profile runtime, or ``None`` when this is not the daemon."""
    state = getattr(scope.get("app"), "state", None)
    if state is None:
        return None
    from superlocalmemory.server.profile_runtime import get_profile_runtime

    return get_profile_runtime(state)


@asynccontextmanager
async def profile_lease(runtime: Any):
    """Hold a profile operation lease; yields the active profile id.

    The blocking acquire runs in a worker thread (it waits only while a switch
    is in progress), shielded so a cancelled request cannot strand a lease.
    """
    acquire = asyncio.ensure_future(asyncio.to_thread(runtime.acquire_operation))
    try:
        snapshot = await asyncio.shield(acquire)
    except asyncio.CancelledError:
        await acquire
        runtime.release_operation()
        raise
    try:
        yield snapshot.profile_id
    finally:
        runtime.release_operation()


RuntimeLookup = Callable[[Mapping[str, Any]], Any]

__all__ = [
    "ACTIVE_ONLY_TOOLS",
    "ARGUMENT_DENIAL",
    "BindingRefusal",
    "CLASSIFIED_ARGUMENTS",
    "INACTIVE_DENIAL",
    "NEUTRAL_ARGUMENTS",
    "PROFILE_ARGUMENTS",
    "PROFILE_ARGUMENT_TOOLS",
    "PROFILE_DENIAL",
    "PROFILE_FREE_TOOLS",
    "ROUTED_TOOLS",
    "READ_SCOPE_ARGUMENTS",
    "RuntimeLookup",
    "SCOPED_WRITE_TOOLS",
    "SCOPE_DENIAL",
    "WRITE_SCOPE_ARGUMENTS",
    "bind_arguments",
    "inactive_refusal",
    "profile_lease",
    "runtime_from_scope",
]
