# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Core MCP Tools (13 tools).

remember, recall, search, fetch, list_recent, get_status, build_graph,
switch_profile, backup_status, memory_used, get_learned_patterns,
correct_pattern, get_attribution.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable

from mcp.types import ToolAnnotations

from superlocalmemory.core.admission import admits
from superlocalmemory.core.config import CANONICAL_LIST_LIMIT, CANONICAL_RECALL_LIMIT
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.mcp._daemon_proxy import daemon_unavailable_error
from superlocalmemory.mcp.remote_visibility import visible_facts
from superlocalmemory.mcp.shared import authorize_mcp_mutation, parse_id_list

logger = logging.getLogger(__name__)


def _projection_queue_depth(db: object) -> int:
    """Facts queued for the graph and vector projections, or 0 when there are none.

    Imported inside the function: this module is loaded on every MCP start over
    stdio, where import cost is startup latency a user feels.
    """
    try:
        from superlocalmemory.storage import projection_outbox
        return projection_outbox.depth(db)
    except Exception:
        return 0


async def _runtime_profile(get_engine: Callable, explicit: str = "") -> str:
    """Resolve an MCP default profile from daemon runtime truth."""
    if explicit:
        return explicit
    import asyncio

    try:
        from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

        if await asyncio.to_thread(is_daemon_running):
            status = await asyncio.to_thread(daemon_request, "GET", "/status")
            if isinstance(status, dict) and status.get("profile"):
                return str(status["profile"])
            raise RuntimeError("resident daemon did not report its active profile")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.debug("daemon profile resolution failed: %s", exc)
    return str(get_engine().profile_id)


async def _call_profile(get_engine: Callable, profile_id: object) -> tuple[str, dict | None]:
    """``(profile, refusal)``: the named profile (it must exist), else the active one.

    The active profile is resolved exactly as before ``profile_id`` existed
    (:func:`_runtime_profile`), so an unnamed call is unchanged.
    """
    from superlocalmemory.mcp.request_profile import tool_profile

    if profile_id is None or (isinstance(profile_id, str) and not profile_id.strip()):
        return await _runtime_profile(get_engine), None
    return tool_profile(get_engine(), profile_id)


def _routed_daemon_call(method: str, path: str, body: dict | None = None) -> dict | None:
    """A daemon mutation for a named profile.

    The daemon's 404 for a profile that does not exist (or a memory that is
    not in it) is an answer, returned with its code so the caller does not
    retry it; ``None`` still means the daemon did not answer.
    """
    from superlocalmemory.cli.daemon import DaemonConflict, DaemonNotFound, daemon_request

    try:
        return daemon_request(method, path, body, preserve_not_found=True,
                              preserve_conflict=True)
    except DaemonNotFound as exc:
        # The route's own code (unknown_profile) when it gave one.
        return {"success": False, "code": exc.error_code or exc.code, "retryable": False,
                "error": exc.error_message or exc.message}
    except DaemonConflict as exc:  # e.g. a correction already open: retrying cannot help
        return {"success": False, "code": "CONFLICT", "retryable": False, "error": exc.detail}

def _emit_event(event_type: str, payload: dict | None = None,
                source_agent: str = "mcp_client") -> None:
    """Emit an event to the EventBus (best-effort, never raises)."""
    try:
        from superlocalmemory.infra.event_bus import EventBus
        bus = EventBus.get_instance(str(state_path("memory.db")))
        bus.emit(event_type, payload=payload, source_agent=source_agent,
                 source_protocol="mcp")
    except Exception:
        pass


# `remember` only adds a memory. The hints are declared explicitly because the
# MCP default for an unannotated tool is destructiveHint=true. They live in a
# constant so the decorator stays on one line, which is the form the packaging
# test's source-level tool-name discovery reads.
_ADDS_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)


def register_core_tools(server, get_engine: Callable) -> None:
    """Register the 13 core MCP tools on *server*."""

    # Adds a memory; never deletes or overwrites one (see _ADDS_ONLY_ANNOTATIONS).
    @server.tool(annotations=_ADDS_ONLY_ANNOTATIONS)
    @admits(OperationKind.REMEMBER)
    async def remember(
        content: str, tags: str = "", project: str = "",
        importance: int = 5, session_id: str = "",
        agent_id: str = "mcp_client",
        scope: str | None = None,
        shared_with: str = "",
        idempotency_key: str = "",
        session_date: str = "",
        profile_id: str = "",
        kind: str = "",
        replaces: str | None = None,
    ) -> dict:
        """Store content to memory with intelligent indexing.

        Extracts atomic facts, resolves entities, builds graph edges,
        and indexes for hybrid retrieval with graph-aware enhancement.

        Multi-scope: ``scope`` sets visibility (personal/shared/global).
        ``shared_with`` is a comma-separated list of profile_ids for
        shared scope.

        ``session_date`` says WHEN the memory is about, as opposed to when it
        was written. Omit it and the memory is dated today, which is what every
        memory got before 4.0.10 because there was no way to say otherwise.
        Accepts YYYY-MM-DD or a full ISO 8601 timestamp.

        ``profile_id`` is an explicit namespace anchor: a non-empty value
        routes this one write to that profile (which must already exist —
        an unknown id is rejected, never created); empty = the active
        profile, byte-identical to the legacy call. Routing never moves
        the active-profile pointer.

        ``kind`` says what sort of memory this is. Set it whenever you know:
        ``rule`` (a standing instruction: "always/never ..."), ``decision`` (a
        choice that settles one question), ``status`` (the current state of
        something, which a later update will replace), ``procedure`` (steps or
        commands), ``prospective`` (a plan or to-do), ``opinion`` (a
        preference or view), ``correction`` (says an earlier memory was wrong),
        ``episodic`` (something that happened) or ``semantic`` (a lasting
        fact). A declared kind is confirmed: rules and decisions you save this
        way are loaded at the start of later sessions. Leave it empty when
        unsure; SLM may suggest one later, and a suggestion changes nothing.

        ``replaces`` is the id of an earlier memory this one replaces - set it
        when you are updating something you saved before (a status, a
        decision, a rule). Use the ``fact_id`` remember returned for it, or a
        ``fact_id`` / ``memory_id`` from recall. The old memory is then no
        longer returned by recall or loaded at session start; nothing is
        deleted. The response's ``replaced`` says what was retired, or why
        nothing was; the memory itself is saved either way. An unknown id, or
        another profile's memory, is refused before anything is saved. Undo
        with ``review_correction(case_id, "rollback", version)``.
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        # Bind the write to a session the same way the read path does.
        #
        # recall has resolved this through a four-step ladder since S9-DASH-10;
        # remember stored whatever it was handed, which for a caller that does
        # not pass one is nothing. Result on the author's store: 192 of 3,894
        # facts carry a session_id (4.9%). The engine's session-diversity
        # promotion cannot promote a fact with no session, so it was running
        # against a corpus where 95% of rows looked like the same session.
        #
        # allow_agent_fallback is OFF here, unlike recall. `mcp:<agent_id>` is a
        # useful key for settling one outcome; as a stored session_id it would
        # file every memory an agent ever wrote under one session, and diversity
        # promotion would then treat a whole history as a single conversation —
        # worse than the empty string it replaces.
        from superlocalmemory.mcp.session_binding import resolve_session_id

        session_id = resolve_session_id(
            session_id, agent_id=agent_id, allow_agent_fallback=False,
        )
        from superlocalmemory.core.project_identity import storable_project

        meta = {
            "project": storable_project(project),
            "importance": importance,
            "agent_id": agent_id,
            "session_id": session_id,
        }
        # Sent as the request's own ``kind`` field, never inside metadata (the
        # daemon strips reserved keys there): see mcp/_remember_kind.py.
        from superlocalmemory.mcp import _remember_kind as rk

        declared_kind, kind_error = rk.parse_declared_kind(kind)
        if kind_error is not None:
            return kind_error
        replaces_id = None
        if replaces is not None:
            from superlocalmemory.core.replaces_input import (
                ReplacesRejected,
                normalize_replaces,
            )

            try:
                replaces_id = normalize_replaces(replaces)
            except ReplacesRejected as exc:
                return {"success": False, "code": exc.code, "retryable": False,
                        "error": exc.message}
        effective_idempotency_key = idempotency_key
        if not effective_idempotency_key:
            # Derive a stable key before the first attempt so every retry of
            # the same logical call carries the same key.  When a session token
            # is present it is included in the material to keep per-session
            # stores separate.  Without a session token the key is derived from
            # the remaining call parameters so repeated observations with the
            # same content, agent, and scope are deduplicated across retries.
            # ``replaces`` is part of what was asked: the same words replacing a
            # different memory are a different request. Added only when set,
            # so every key derived for a plain call is unchanged.
            replaces_part = (f"\0replaces={replaces_id}" if replaces_id is not None else ""
                             ) + rk.kind_key_part(declared_kind)  # so is the kind
            if session_id:
                material = (
                    f"{agent_id}\0{session_id}\0{scope or ''}\0{shared_with}\0{content}"
                    + replaces_part
                )
                effective_idempotency_key = "mcp:" + hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest()
            else:
                material = (
                    f"{agent_id}\0{scope or ''}\0{shared_with}\0{content}" + replaces_part
                )
                effective_idempotency_key = "mcp:req:" + hashlib.sha256(
                    material.encode("utf-8")
                ).hexdigest()
        # Parse shared_with from comma-separated string
        _shared_list = [s.strip() for s in shared_with.split(",") if s.strip()] if shared_with else None
        # v3.5.5 WRITE-THROUGH: route through the daemon's /remember, which does
        # a synchronous verbatim insert (memory is keyword/BM25-recallable the
        # instant this returns) and enqueues async enrichment. This closes the
        # recall window so a parallel/next agent finds memories saved seconds ago.
        # Falls back to the capability-owned worker only if the daemon is
        # unreachable. Raw pending.db writes are legacy replay input only.
        daemon_owned = False
        try:
            import asyncio as _asyncio

            from superlocalmemory.cli.daemon import daemon_request, is_daemon_running
            # is_daemon_running() and daemon_request() both use blocking urllib
            # against the same uvicorn server — run in threads so the MCP
            # event loop stays unblocked (#34 class bug).
            daemon_owned = await _asyncio.to_thread(is_daemon_running)
            if daemon_owned:
                # A positively identified daemon owns this database. Never
                # spawn a WorkerPool writer after a transient daemon failure:
                # that creates the competing writers which SQLite WAL cannot
                # support. Retry the canonical path, then return an explicit
                # retryable result to the MCP client.
                for attempt in range(3):
                    body = {
                        "content": content, "tags": tags, "metadata": meta,
                        "scope": scope, "shared_with": _shared_list,
                        "session_id": session_id,
                        "session_date": session_date,
                        "idempotency_key": effective_idempotency_key or None,
                    }
                    if (profile_id or "").strip():
                        # Per-request profile routing (spec section 3/5): the
                        # anchor is only put on the wire when the caller set
                        # it, so an unset profile_id keeps the legacy request
                        # byte-identical. The daemon routes THIS one write to
                        # that profile without moving the active pointer.
                        # 4.1.14 audit: stripped — whitespace-only is legacy,
                        # padded ids travel canonical.
                        body["profile_id"] = profile_id.strip()
                    # A 422 refuses THIS request (a reused key, a bad
                    # ``replaces``): an answer, never an outage to retry.
                    request_flags = {"preserve_not_found": True, "preserve_unprocessable": True}
                    if replaces_id is not None:
                        # Sent only when set, so a plain call is unchanged.
                        body["replaces"] = replaces_id
                    body, request_flags = rk.with_declared_kind(body, request_flags, declared_kind)
                    resp = None
                    try:
                        resp = await _asyncio.to_thread(
                            daemon_request, "POST", "/remember", body,
                            **request_flags,
                        )
                    except Exception as exc:
                        # 4.1.14 audit: a live daemon's unknown-profile 404
                        # surfaces immediately — neither the 3x retry below
                        # nor the pool fallback can heal a 404.
                        if type(exc).__name__ == "DaemonNotFound" and hasattr(exc, "code"):
                            return {
                                "success": False,
                                "code": getattr(exc, "code"),
                                "retryable": False,
                                "error": getattr(exc, "message", "daemon returned 404"),
                            }
                        if type(exc).__name__ == "DaemonUnprocessable" and hasattr(exc, "code"):
                            return {
                                "success": False,
                                "code": getattr(exc, "code"),
                                "retryable": False,
                                "error": getattr(exc, "message", str(exc)),
                            }
                        if type(exc).__name__ == "DaemonRefused":
                            # A refusal (401/403) is an answer, not an outage:
                            # reporting it as retryable invited endless retries.
                            return {
                                "success": False,
                                "code": "NOT_AUTHORIZED",
                                "retryable": False,
                                "error": str(exc),
                            }
                        raise
                    if resp and (resp.get("fact_ids") is not None or resp.get("ok")):
                        fids = resp.get("fact_ids") or []
                        materialization_state = resp.get("materialization_state")
                        if materialization_state is None:
                            materialization_state = (
                                "complete" if resp.get("status") == "stored" else "queryable"
                            )
                        pending = materialization_state != "complete"
                        if resp.get("status") == "accepted":
                            # Durable but not yet searchable: the writer was
                            # busy. Never reported as "queryable now".
                            accepted_reply = {
                                "success": True,
                                "fact_ids": [],
                                "count": 0,
                                "pending": True,
                                "pending_id": None,
                                "operation_id": None,
                                "materialization_state": "accepted",
                                "admission_id": resp.get("admission_id"),
                                "idempotency_key": resp.get("idempotency_key"),
                                "message": (
                                    "Saved durably; being indexed now and searchable "
                                    "within seconds. Resend with the same "
                                    "idempotency_key for the final receipt."
                                ),
                            }
                            if resp.get("replaced") is not None:
                                accepted_reply["replaced"] = resp["replaced"]
                            return accepted_reply
                        stored_reply = {
                            "success": True,
                            "fact_ids": fids,
                            "count": int(resp.get("count", len(fids))),
                            "pending": pending,
                            "pending_id": resp.get("pending_id") if pending else None,
                            "operation_id": resp.get("operation_id"),
                            "materialization_state": materialization_state,
                            "message": (
                                "Stored through canonical daemon ingestion."
                                if not pending
                                else "Queryable now; canonical enrichment is still running."
                            ),
                        }
                        return rk.with_receipt_notes(stored_reply, resp)
                    if attempt < 2:
                        await _asyncio.sleep(0.05 * (attempt + 1))
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": (
                        daemon_unavailable_error()
                    ),
                }
        except Exception as dexc:
            logger.debug("MCP remember via daemon failed, pending fallback: %s", dexc)
            if daemon_owned:
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": (
                        daemon_unavailable_error()
                    ),
                }

        try:
            import asyncio as _asyncio

            from superlocalmemory.mcp._daemon_proxy import choose_pool

            worker_meta = {
                **meta,
                "tags": tags,
                "scope": scope or "personal",
                "shared_with": _shared_list or [],
                # L3-24: carried through the same way the daemon-owned branch
                # above always has (its "session_date": session_date body
                # field) -- DaemonPoolProxy.store lifts this back out of
                # metadata into the request's own field.
                "session_date": session_date,
                "idempotency_key": (
                    effective_idempotency_key
                    or "mcp:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
                ),
            }
            if (profile_id or "").strip():
                # DaemonPoolProxy.store forwards metadata["profile_id"] as
                # the per-request routing anchor on POST /remember. Kept out
                # of the metadata entirely when unset so the legacy fallback
                # call stays byte-identical. 4.1.14 audit: stripped.
                worker_meta["profile_id"] = profile_id.strip()
            if replaces_id is not None:
                # DaemonPoolProxy.store lifts this out of the metadata into
                # the request's own ``replaces`` field.
                worker_meta["replaces"] = replaces_id

            def _store_via_daemon_pool():
                pool = choose_pool()
                return pool.store(content, worker_meta, **rk.store_kwargs(declared_kind))

            stored = await _asyncio.to_thread(_store_via_daemon_pool)
            if not isinstance(stored, dict) or not stored.get("ok"):
                if isinstance(stored, dict) and stored.get("code") == "DAEMON_UNAVAILABLE":
                    return {
                        "success": False,
                        "code": "DAEMON_UNAVAILABLE",
                        "retryable": True,
                        "error": stored.get(
                            "error",
                            daemon_unavailable_error(),
                        ),
                    }
                # 4.1.14 audit: structured non-retryable answers from the
                # pool (unknown_profile, PROFILE_MISMATCH) pass through
                # verbatim — mislabeling them DAEMON_UNAVAILABLE retryable
                # would send clients into a hopeless retry loop.
                if isinstance(stored, dict) and stored.get("code"):
                    return {
                        "success": False,
                        "code": stored.get("code"),
                        "retryable": bool(stored.get("retryable", False)),
                        "error": stored.get("error", "daemon request failed"),
                    }
                return {
                    "success": False,
                    "code": "DAEMON_UNAVAILABLE",
                    "retryable": True,
                    "error": daemon_unavailable_error(),
                }
            fact_ids = list(stored.get("fact_ids") or [])
            materialization_state = str(
                stored.get("materialization_state") or "complete"
            )
            allowed_states = {"queryable", "enriching", "complete"}
            if materialization_state not in allowed_states:
                raise RuntimeError(
                    "canonical worker returned invalid materialization state: "
                    f"{materialization_state}"
                )
            pending = materialization_state != "complete"
            operation_id = stored.get("operation_id")
            pending_id = stored.get("pending_id")
            if pending and pending_id is None:
                pending_id = operation_id
            pool_reply = {
                "success": True,
                "fact_ids": fact_ids,
                "count": int(stored.get("count", len(fact_ids))),
                "pending": pending,
                "pending_id": pending_id if pending else None,
                "operation_id": operation_id,
                "materialization_state": materialization_state,
                "message": (
                    "Stored through canonical local ingestion."
                    if not pending
                    else "Queryable now; canonical enrichment is still running."
                ),
            }
            return rk.with_receipt_notes(pool_reply, stored)
        except Exception:
            logger.exception("remember failed")
            return {
                "success": False,
                "code": "DAEMON_UNAVAILABLE",
                "retryable": True,
                "error": daemon_unavailable_error(),
            }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def recall(
        query: str, limit: int = CANONICAL_RECALL_LIMIT, agent_id: str = "mcp_client",
        session_id: str = "", fast: bool | None = None,
        include_global: bool | None = None,
        include_shared: bool | None = None,
        window: str = "",
        as_of: str | None = None,
        known_as_of: str | None = None,
        valid_at: str | None = None,
        include_unknown: bool = False,
        profile_id: str = "",
        project: str = "",
        saved_by: str = "",
        about: str = "",
        kind: str = "",
        prefer_project: str = "",
        tags: "str | list[str]" = "",
        tags_match: str = "all",
        project_strict: bool = False,
    ) -> dict:
        """Search memories through hybrid retrieval, RRF fusion, and reranking.

        Fast local retrieval (six channels + reranker) returns in ~1-2s. This
        tool does NOT run an internal LLM reformulation round — YOU (the calling
        model) are the reasoner. Drive refinement using the confidence signals
        in the response:
          • ``no_confident_match: true`` → nothing cleared the evidence floor.
            Do NOT invent a memory. Rewrite the query into 1-3 more specific
            sub-queries (split multi-hop questions; try entity names, synonyms,
            or a broader phrasing) and call ``recall`` again before concluding
            the information is unknown.
          • ``answer_confidence`` low / ``abstained: true`` → the top hit is
            weak. Re-query with a sharper phrasing, or widen with
            ``include_shared=true`` / ``include_global=true`` if appropriate.
          • Confident match → use it directly; no second call needed.
        One extra targeted recall is cheap and beats a wrong "not found".

        Optional ``session_id`` threads through to the
        engine's outcome-queue so PostToolUse / Stop hooks can attach
        engagement signals to this recall. Claude Code should pass its
        ``CLAUDE_SESSION_ID``. Omitting it degrades to "no closed-loop
        learning for this recall" — the recall itself always works.

        Multi-scope: ``include_global`` / ``include_shared`` control which
        scopes participate in retrieval. Leave them unset (``None``) to use the
        configured default — shared memory is OPT-IN, so by default recall
        returns only this profile's own facts. Pass ``True`` to opt in per call.

        Time window: optional ``window`` restricts results to a event-time
        range. Accepts a relative span (``"24h"``, ``"7d"``, ``"30d"``,
        ``"1y"``) or an explicit range (``"2026-07-01..2026-07-31"``). Empty =
        no time filter.

        Point-in-time: optional ``as_of`` (ISO-8601 string, e.g.
        ``"2026-01-01T00:00:00+00:00"``) pins recall to a temporal snapshot;
        omit or pass ``None`` for current-state recall.

        ``profile_id`` is an explicit namespace anchor: a non-empty value
        serves this one recall against that profile (which must already
        exist); empty = the active profile, byte-identical to the legacy
        call. The active-profile pointer is never read or moved by it.
        
        Narrowing (4.1.19): ``project`` keeps only memories saved under that
        project, ``saved_by`` only those saved by that agent (e.g.
        ``claude-desktop``), ``about`` only those that mention that name (a
        person, project or tool). Each is a hard filter. Questions phrased as
        "what did we decide", "how do I", "what is the current status of" get
        decisions, how-tos and the newest current-state memory first.

        Projects (4.1.21): ``prefer_project`` (a name or a path) ranks memories
        saved under that project above others of similar relevance and removes
        nothing - pass your working directory on every recall in a project.
        ``project`` keeps only that project's memories; when none of the
        memories found were saved under it, the unfiltered results come back
        and ``project_scope.filter.applied`` is false with a ``note`` saying
        so - never a silent empty answer. A name matches the same project saved
        as a full path, ignoring case.
        ``project_strict=True`` (4.1.22) turns that fall-back off: only that
        project's memories, even if none - for automations that must never
        act on another project's memories. ``project_scope.filter.identity``
        says which name was matched and warns when one name stands for more
        than one saved project path.

        ``kind`` (4.1.19 WP8) keeps only results whose kind — the same nine
        values ``remember``'s ``kind`` parameter takes — equals this value,
        including memories SLM only mapped from their legacy type. Refused
        (``INVALID_KIND``) before anything is retrieved if it does not parse.
        Composes with ``project``/``saved_by``/``about`` as AND.

        ``tags`` (4.1.22): only memories saved with these exact labels (a
        comma-separated string, or a list — a label containing a comma needs
        the list form). Matched by canonical identity: case, surrounding
        whitespace and how the tag happened to be stored (a CSV string, a
        JSON-array string, or a real list) never matter. ``tags_match`` is
        ``"all"`` (every label must be present — the default) or ``"any"``.
        A hard filter that composes with every other facet as AND; unlike
        ``project`` it never falls back to unfiltered results — an empty
        answer says why in ``tag_scope`` (nobody ever saved that tag, versus
        something has it but not among this question's matches).
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        from superlocalmemory.core.kind_query import InvalidKind, resolve_kind
        try:
            _kind = resolve_kind(kind)
        except InvalidKind as exc:
            return {"success": False, "code": "INVALID_KIND", "retryable": False,
                    "error": str(exc)}
        # 4.1.20 (R5): an unreadable window is refused, not passed on to run
        # unfiltered — the same check and code as the CLI and HTTP.
        from superlocalmemory.retrieval.time_filter import InvalidTimeFilter, check_window
        try:
            window = check_window(window) or ""
        except InvalidTimeFilter as exc:
            return {"success": False, "code": exc.code, "retryable": False,
                    "field": exc.field, "error": str(exc)}
        import asyncio
        try:
            from superlocalmemory.mcp._daemon_proxy import choose_pool
            from superlocalmemory.mcp._recall_metadata import forward_recall_metadata
            from superlocalmemory.mcp.session_binding import resolve_session_id

            # S9-DASH-10's four-step ladder, now shared with remember() so the
            # read path and the write path cannot disagree about which session
            # they are in. remember() had no ladder at all, which is why 95% of
            # stored facts carry no session_id. See mcp/session_binding.py.
            #
            # The per-agent fallback stays ON here: this id settles a pending
            # outcome, and `mcp:<agent_id>` is deliberately not matched by the
            # Stop hook, so the reaper settles it at a neutral 0.5 rather than
            # attributing engagement to a session that never existed.
            effective_sid = resolve_session_id(
                session_id, agent_id=agent_id, allow_agent_fallback=True,
            )
            # Resolve the daemon proxy inside the worker too. ``choose_pool``
            # verifies daemon ownership through a synchronous /health request;
            # when this tool is served by the daemon's mounted HTTP MCP app,
            # resolving it on Uvicorn's event-loop thread makes that loop wait
            # on its own health response forever. Stdio did not exhibit this
            # because its MCP process is external to the daemon.
            #
            # V3.4.26: WorkerPool now concurrent — parallel calls no longer
            # block behind a single threading.Lock. See worker_pool.py.
            # Phase 4b: normalize as_of at MCP boundary. Invalid → reject.
            # Audit P2: treat empty/whitespace as_of as ABSENT (like HTTP does),
            # not as an invalid value — only a non-blank unparseable string is
            # rejected.
            def _normalize_temporal(value: str | None) -> str | None:
                if value is None or not str(value).strip():
                    return None
                from superlocalmemory.retrieval.temporal_utils import normalize_as_of
                return normalize_as_of(value)

            raw_as_of, raw_known_as_of, raw_valid_at = as_of, known_as_of, valid_at
            as_of = _normalize_temporal(raw_as_of)
            known_as_of = _normalize_temporal(raw_known_as_of)
            valid_at = _normalize_temporal(raw_valid_at)
            # Preserve backwards-compatible as_of validation while exposing
            # named two-clock boundaries. A supplied non-blank invalid value
            # is rejected rather than silently becoming current recall.
            for raw, normalized, code in (
                (raw_as_of, as_of, "invalid_as_of"),
                (raw_known_as_of, known_as_of, "invalid_known_as_of"),
                (raw_valid_at, valid_at, "invalid_valid_at"),
            ):
                if raw is not None and str(raw).strip() and normalized is None:
                    return {"success": False, "error": code}

            from superlocalmemory.core.admission import enforce_read_scope
            _incl_global, _incl_shared = enforce_read_scope(include_global, include_shared)

            # A relayed call must answer before its relay gives up: what is left
            # of the deadline the laptop stamped on this request becomes the
            # recall's budget. Read here, on the request's own context (the
            # worker thread below is not it). No deadline -> nothing is sent.
            from superlocalmemory.mcp.request_deadline import remaining_budget_s
            _budget_s = remaining_budget_s()

            def _recall_via_daemon_pool():
                pool = choose_pool()
                return pool.recall(
                    query, limit=limit, session_id=effective_sid,
                    fast=fast, include_global=_incl_global,
                    include_shared=_incl_shared, window=window or None,
                    as_of=as_of,
                    known_as_of=known_as_of,
                    valid_at=valid_at,
                    include_unknown=include_unknown,
                    # Per-request profile routing (spec section 3/5): threaded
                    # only when set, so an unset anchor keeps the legacy call
                    # byte-identical and pool shapes that predate the
                    # parameter are never asked for it. 4.1.14 audit:
                    # stripped (whitespace-only is legacy).
                    **({"profile_id": profile_id.strip()} if (profile_id or "").strip() else {}),
                    # 4.1.19 facets, only when set.
                    **{k: v.strip() for k, v in (("project", project), ("saved_by", saved_by),
                                                 ("about", about),
                                                 ("prefer_project", prefer_project))
                       if (v or "").strip()},
                    **({"project_strict": True}
                       if project_strict and (project or "").strip() else {}),
                    # 4.1.19 WP8: the already-validated, normalized kind.
                    **({"kind": _kind} if _kind else {}),
                    # 4.1.22: forwarded only when set, same as every
                    # other facet above.
                    **({"tags": tags if isinstance(tags, list) else tags.strip()}
                       if (tags if isinstance(tags, list) else (tags or "").strip())
                       else {}),
                    **({"tags_match": tags_match.strip()}
                       if (tags_match or "").strip().lower() == "any" else {}),
                    **({"budget_s": _budget_s} if _budget_s is not None else {}),
                )

            result = await asyncio.to_thread(
                _recall_via_daemon_pool,
            )
            if result.get("ok"):
                from superlocalmemory.mcp.tools_media import with_recall_images
                return await with_recall_images({
                    "success": True,
                    "results": result.get("results", []),
                    "count": result.get("result_count", 0),
                    "query_type": result.get("query_type", "unknown"),
                    "channel_weights": result.get("channel_weights", {}),
                    "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                    # 4.1.14 audit: the served namespace travels with the
                    # answer — a routed recall must not look active-profiled
                    # at the MCP boundary either.
                    "profile": result.get("profile", ""),
                    "profile_generation": result.get("profile_generation"),
                    "retrieval_mode": result.get("retrieval_mode", ""),
                    # v3.6.6: surface evidence-floor signal to MCP clients.
                    "no_confident_match": result.get("no_confident_match", False),
                    # M-10: every field of the HTTP envelope's metadata —
                    # score contract, verdict, query_id (quote it back to
                    # report_outcome), who chose the order, abandoned channels,
                    # temporal frame — and any field added there later.
                    **forward_recall_metadata(result),
                }, profile_id)
            # 4.1.14 audit: structured daemon answers (unknown_profile)
            # pass through with code and retryability intact — collapsing
            # them to a bare error string repeats the DAEMON_UNAVAILABLE
            # mislabel one layer down.
            if isinstance(result, dict) and result.get("code"):
                return {
                    "success": False,
                    "code": result.get("code"),
                    "retryable": bool(result.get("retryable", False)),
                    "error": result.get("error", "Recall failed"),
                }
            return {"success": False, "error": result.get("error", "Recall failed")}
        except Exception as exc:
            logger.exception("recall failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def search(query: str, limit: int = CANONICAL_RECALL_LIMIT, kind: str = "",
                     profile_id: str = "", tags: "str | list[str]" = "",
                     tags_match: str = "all") -> dict:
        """Full-text search across memories using FTS5 with BM25 ranking.

        ``kind`` (4.1.19 WP8) keeps only results whose kind — the same nine
        values ``remember``'s ``kind`` parameter takes — equals this value,
        including memories SLM only mapped from their legacy type. Refused
        (``INVALID_KIND``) before anything is retrieved if it does not parse.

        ``profile_id`` reads another profile (empty = the active one); remote
        access sets it to the key's profile, whatever this computer is using.
        A name that is not an existing profile is refused, never silently
        treated as empty (4.1.22).

        ``tags`` / ``tags_match`` (4.1.22): only memories carrying the tags,
        exactly as ``recall`` takes them; ``tag_scope`` says what was found.
        """
        from superlocalmemory.core import tag_query
        from superlocalmemory.core.kind_query import (
            InvalidKind,
            engine_display_min_confidence,
            resolve_kind,
            search_facts,
        )
        from superlocalmemory.storage.memory_kinds import kind_fields
        try:
            parsed_kind = resolve_kind(kind)
        except InvalidKind as exc:
            return {"success": False, "code": "INVALID_KIND", "retryable": False,
                    "error": str(exc)}
        try:
            engine = get_engine()
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            # Read once: used for BOTH the --kind filter above and labelling
            # each item below, so a fact cannot pass the filter at one
            # threshold and be labelled (confirmed/suggested/legacy) at another.
            _display_min_confidence = engine_display_min_confidence(engine)
            _truncated: list[bool] = []
            _tags = tag_query.TagFilter.of(tags, tags_match)
            facts = search_facts(
                engine._db, query, pid, limit, parsed_kind,
                display_min_confidence=_display_min_confidence,
                truncated=_truncated, tag_filter=_tags,
            )
            facts = visible_facts(engine._db, pid, facts)
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "fact_type": f.fact_type.value,
                    "confidence": round(f.confidence, 3),
                    "date": f.observation_date,
                    **kind_fields(f, display_min_confidence=_display_min_confidence),
                })
            result = {"success": True, "results": items, "count": len(items)}
            # 4.1.19 L2-13/M2: told, never a silent short answer, when the
            # kind filter's windowed fetch hit its hard cap before `limit`
            # was filled and the store was not exhausted.
            if _truncated and _truncated[0]:
                result["kind_filter_truncated"] = True
            return tag_query.with_report(result, engine._db, pid, _tags)
        except Exception as exc:
            logger.exception("search failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def fetch(fact_ids: "str | list[str]", profile_id: str = "") -> dict:
        """Fetch full details for specific fact IDs (comma-separated or a list).

        Reports every id it could not resolve. Before 4.1.15 an unmatched token
        returned ``success: true, count: 0`` -- indistinguishable from a
        correct answer for a fact that does not exist, on the one tool an agent
        uses to verify that a write landed. GitHub #135.

        ``profile_id`` reads another profile (empty = the active one); remote
        access sets it to the key's profile, whatever this computer is using.
        A name that is not an existing profile is refused, never silently
        treated as empty (4.1.22).
        """
        try:
            engine = get_engine()
            ids = parse_id_list(fact_ids)
            if not ids:
                return {
                    "success": False,
                    "error": (
                        f"fetch could not read any fact id from {fact_ids!r}. "
                        "Pass a comma-separated string or a list of ids."
                    ),
                    "results": [], "count": 0, "not_found": [],
                }
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            facts = visible_facts(engine._db, pid, engine._db.get_facts_by_ids(ids, pid))
            found = {f.fact_id for f in facts}
            # #150: the project each memory was saved under ("" when none).
            from superlocalmemory.retrieval.project_scope import stored_projects

            projects = stored_projects(engine._db, [f.fact_id for f in facts])
            not_found = [fid for fid in ids if fid not in found]
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content,
                    "fact_type": f.fact_type.value,
                    "entities": f.canonical_entities,
                    "confidence": round(f.confidence, 3),
                    "importance": round(f.importance, 3),
                    "observation_date": f.observation_date,
                    "referenced_date": f.referenced_date,
                    "lifecycle": f.lifecycle.value,
                    "access_count": f.access_count,
                    "project": projects.get(f.fact_id, ""),
                })
            if not items:
                return {
                    "success": False,
                    "error": (
                        "no fact matched "
                        + ", ".join(repr(fid) for fid in not_found)
                        + f" in profile {pid!r}"
                    ),
                    "results": [], "count": 0, "not_found": not_found,
                }
            return {
                "success": True, "results": items, "count": len(items),
                "not_found": not_found,
            }
        except Exception as exc:
            logger.exception("fetch failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    @admits(OperationKind.RECALL)
    async def list_recent(limit: int = CANONICAL_LIST_LIMIT, kind: str = "",
                          profile_id: str = "", tags: "str | list[str]" = "",
                          tags_match: str = "all") -> dict:
        """List most recently stored memories, newest first.

        ``kind`` (4.1.19 WP8) keeps only memories whose kind — the same nine
        values ``remember``'s ``kind`` parameter takes — equals this value,
        including memories SLM only mapped from their legacy type. Refused
        (``INVALID_KIND``) before anything is retrieved if it does not parse.

        ``profile_id`` reads another profile (empty = the active one); remote
        access sets it to the key's profile, whatever this computer is using.
        A name that is not an existing profile is refused, never silently
        treated as empty (4.1.22).

        ``tags`` / ``tags_match`` (4.1.22): only memories carrying the tags,
        newest first, exactly as ``recall`` takes them.
        """
        from superlocalmemory.core import tag_query
        from superlocalmemory.core.kind_query import (
            InvalidKind,
            engine_display_min_confidence,
            list_recent_facts,
            resolve_kind,
        )
        from superlocalmemory.storage.memory_kinds import kind_fields
        try:
            parsed_kind = resolve_kind(kind)
        except InvalidKind as exc:
            return {"success": False, "code": "INVALID_KIND", "retryable": False,
                    "error": str(exc)}
        try:
            engine = get_engine()
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            # v3.6.12 (search-2): push the limit into the query — was loading the
            # ENTIRE facts table (deserializing every 768-float embedding) just
            # to return the top N. get_all_facts preserves created_at DESC order.
            # Read once: used for BOTH the --kind filter above and labelling
            # each item below, so a fact cannot pass the filter at one
            # threshold and be labelled (confirmed/suggested/legacy) at another.
            _display_min_confidence = engine_display_min_confidence(engine)
            _truncated: list[bool] = []
            _tags = tag_query.TagFilter.of(tags, tags_match)
            facts = list_recent_facts(
                engine._db, pid, limit, parsed_kind,
                display_min_confidence=_display_min_confidence,
                truncated=_truncated, tag_filter=_tags,
            )
            facts = visible_facts(engine._db, pid, facts)
            items = []
            for f in facts:
                items.append({
                    "fact_id": f.fact_id,
                    "content": f.content[:120],
                    "fact_type": f.fact_type.value,
                    "created_at": f.created_at,
                    "session_id": f.session_id,
                    **kind_fields(f, display_min_confidence=_display_min_confidence),
                })
            result = {"success": True, "results": items, "count": len(items)}
            # 4.1.19 L2-13/M2: see the matching note in search() above.
            if _truncated and _truncated[0]:
                result["kind_filter_truncated"] = True
            return tag_query.with_report(result, engine._db, pid, _tags)
        except Exception as exc:
            logger.exception("list_recent failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_status(profile_id: str = "") -> dict:
        """Get memory system status: fact count, entity count, mode, profile, db size.

        ``profile_id`` reports another profile (empty = the active one); the
        counts are then that profile's, read from the store.
        """
        try:
            # Same source the HTTP surface reads, imported here rather than at
            # module scope: that module costs ~260ms and MCP starts over stdio.
            from superlocalmemory.server.routes.helpers import SLM_VERSION

            import asyncio
            import os

            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )
            from superlocalmemory.mcp.request_profile import tool_profile

            # The daemon's /status describes its active profile, so a named
            # profile is counted from the store below instead.
            named = not isinstance(profile_id, str) or bool(profile_id.strip())
            if not named and await asyncio.to_thread(is_daemon_running):
                daemon_status = await asyncio.to_thread(
                    daemon_request,
                    "GET",
                    "/status",
                )
                if isinstance(daemon_status, dict) and daemon_status.get("profile"):
                    return {
                        "success": True,
                        "mode": daemon_status.get("mode", "unknown"),
                        "provider": daemon_status.get("provider", "none"),
                        "profile": daemon_status["profile"],
                        "base_dir": daemon_status.get("base_dir", ""),
                        "db_path": daemon_status.get("db_path", ""),
                        "db_size_mb": float(daemon_status.get("db_size_mb", 0.0)),
                        "fact_count": int(daemon_status.get("fact_count", 0)),
                        "entity_count": int(daemon_status.get("entity_count", 0)),
                        "edge_count": int(daemon_status.get("edge_count", 0)),
                        "profile_generation": int(
                            daemon_status.get("profile_generation", 0)
                        ),
                        "version": SLM_VERSION,
                        "projection_queue_depth": int(
                            daemon_status.get("projection_queue_depth", 0)
                        ),
                    }

            engine = get_engine()
            pid, refused = tool_profile(engine, profile_id)
            if refused:
                return refused
            fact_count = engine._db.get_fact_count(pid)
            entities = engine._db.execute(
                "SELECT COUNT(*) AS c FROM canonical_entities WHERE profile_id = ?",
                (pid,),
            )
            entity_count = int(dict(entities[0])["c"]) if entities else 0
            edges = engine._db.execute(
                "SELECT COUNT(*) AS c FROM graph_edges WHERE profile_id = ?",
                (pid,),
            )
            edge_count = int(dict(edges[0])["c"]) if edges else 0

            db_size_mb = 0.0
            db_path = engine._db.db_path
            if db_path.exists():
                db_size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)

            # additive canonical key set — provider/base_dir/db_path added.
            # All pre-existing keys are preserved (zero removals).
            cfg = engine._config
            return {
                "success": True,
                "mode": cfg.mode.value,
                "provider": cfg.llm.provider or "none",
                "profile": pid,
                "base_dir": str(cfg.base_dir),
                "db_path": str(db_path),
                "db_size_mb": db_size_mb,
                "fact_count": fact_count,
                "entity_count": entity_count,
                "edge_count": edge_count,
                "profile_generation": 0,
                "version": SLM_VERSION,
                "projection_queue_depth": _projection_queue_depth(engine._db),
            }
        except Exception as exc:
            logger.exception("get_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def build_graph() -> dict:
        """Rebuild knowledge graph edges for all facts in the active profile."""
        try:
            engine = get_engine()
            pid = await _runtime_profile(get_engine)
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-build-memory-graph",
                profile_id=pid,
            )
            facts = engine._db.get_all_facts(pid)
            edge_count = 0
            for fact in facts:
                if engine._graph_builder:
                    engine._graph_builder.build_edges(fact, pid)
                    edge_count += 1
            authorization.complete()
            return {
                "success": True,
                "facts_processed": len(facts),
                "edges_built": edge_count,
            }
        except Exception as exc:
            logger.exception("build_graph failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.PROFILE_SWITCH)
    async def switch_profile(profile_id: str) -> dict:
        """Switch the active memory profile. All operations scope to this profile."""
        try:
            import asyncio

            engine = get_engine()
            old = engine.profile_id
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-switch-profile",
                profile_id=profile_id,
                content_preview=f"{old} -> {profile_id}",
            )
            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )

            generation = 0
            # Only the profile_id explicitly confirmed by this process
            # (daemon-acknowledged + locally-validated, or locally
            # validated directly) is ever synced into engine state.
            confirmed_profile_id = None
            if await asyncio.to_thread(is_daemon_running):
                result = await asyncio.to_thread(
                    daemon_request,
                    "POST",
                    f"/api/profiles/{profile_id}/switch",
                )
                if not result or not result.get("success"):
                    return {
                        "success": False,
                        "error": "resident daemon rejected the profile switch",
                    }
                acknowledged = str(result.get("active_profile", ""))
                if not acknowledged or acknowledged != profile_id:
                    return {
                        "success": False,
                        "error": "resident daemon acknowledged a different profile",
                    }
                # Local consistency guard (SEC-H-01): the daemon's HTTP
                # acknowledgement alone is not sufficient — this MCP
                # process must also confirm the profile exists in its
                # own local DB handle before syncing local state to it.
                # Mirrors the existence check the no-daemon branch already
                # performs below.
                local_rows = engine._db.execute(
                    "SELECT 1 FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                if not local_rows:
                    return {
                        "success": False,
                        "error": (
                            f"resident daemon acknowledged profile "
                            f"'{acknowledged}' but it does not exist in "
                            f"this process's local profile store"
                        ),
                    }
                generation = int(result.get("generation", 0))
                # Sync target is the DAEMON-CONFIRMED value, never the raw
                # caller-supplied profile_id, even though they are equal
                # here by construction (checked above).
                confirmed_profile_id = acknowledged
            else:
                rows = engine._db.execute(
                    "SELECT 1 FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                if not rows:
                    return {
                        "success": False,
                        "error": f"Profile '{profile_id}' does not exist.",
                    }
                from superlocalmemory.server.profile_runtime import (
                    persist_active_profile,
                )

                persistence = persist_active_profile(profile_id)
                try:
                    engine.profile_id = profile_id
                    engine._config.active_profile = profile_id
                except BaseException:
                    engine.profile_id = old
                    engine._config.active_profile = old
                    persistence.rollback()
                    raise
                confirmed_profile_id = profile_id

            if not confirmed_profile_id:
                # Defensive: should be unreachable — every path above
                # either returns an error or sets confirmed_profile_id.
                return {
                    "success": False,
                    "error": "profile switch could not be confirmed",
                }

            # Synchronize this MCP process only after confirmation
            # (daemon-acknowledged + locally-validated, or directly
            # locally-validated in the no-daemon branch above).
            engine.profile_id = confirmed_profile_id
            engine._config.active_profile = confirmed_profile_id

            # v3.6.12 (search-3): recall/delete run in a separate worker
            # subprocess that caches its engine (and profile_id) at init. Recycle
            # it so the NEXT recall uses the new profile instead of the stale one.
            try:
                from superlocalmemory.core.worker_pool import WorkerPool
                WorkerPool.shared().shutdown()
            except Exception:
                logger.debug("worker-pool recycle on profile switch skipped")

            authorization.complete()
            return {
                "success": True,
                "previous_profile": old,
                "current_profile": profile_id,
                "generation": generation,
            }
        except Exception as exc:
            logger.exception("switch_profile failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def backup_status() -> dict:
        """Get backup system status, last backup time, and available backup files."""
        try:
            engine = get_engine()
            from superlocalmemory.infra.backup import BackupManager
            bm = BackupManager(
                db_path=engine._db.db_path,
                base_dir=engine._config.base_dir,
            )
            return {"success": True, **bm.get_status()}
        except Exception as exc:
            logger.exception("backup_status failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def memory_used(profile_id: str = "") -> dict:
        """Get memory usage breakdown by fact type and lifecycle state.

        ``profile_id`` reports another profile (empty = the active one).
        """
        try:
            engine = get_engine()
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            facts = engine._db.get_all_facts(pid)
            by_type: dict[str, int] = {}
            by_lifecycle: dict[str, int] = {}
            for f in facts:
                by_type[f.fact_type.value] = by_type.get(f.fact_type.value, 0) + 1
                by_lifecycle[f.lifecycle.value] = (
                    by_lifecycle.get(f.lifecycle.value, 0) + 1
                )
            return {
                "success": True,
                "total_facts": len(facts),
                "by_type": by_type,
                "by_lifecycle": by_lifecycle,
                "profile": pid,
            }
        except Exception as exc:
            logger.exception("memory_used failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_learned_patterns(pattern_type: str = "", limit: int = 20,
                                   profile_id: str = "") -> dict:
        """Get learned behavioral patterns (interests, refinements, archival habits).

        ``profile_id`` reads another profile (empty = the active one).
        """
        try:
            engine = get_engine()
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            ptype = pattern_type if pattern_type else None
            patterns = store.get_patterns(
                pid, pattern_type=ptype, limit=limit,
            )
            return {"success": True, "patterns": patterns, "count": len(patterns)}
        except Exception as exc:
            logger.exception("get_learned_patterns failed")
            return {"success": False, "error": str(exc)}

    @server.tool()
    @admits(OperationKind.CORRECT)
    async def correct_pattern(pattern_id: str, correction: str, profile_id: str = "") -> dict:
        """Correct or annotate a learned behavioral pattern to improve retrieval.

        ``profile_id`` corrects another profile's pattern (empty = the active one).
        """
        try:
            engine = get_engine()
            pid, refused = await _call_profile(get_engine, profile_id)
            if refused:
                return refused
            authorization = authorize_mcp_mutation(
                engine,
                "update",
                mutation_source="mcp-correct-pattern",
                profile_id=pid,
                fact_id=pattern_id,
                content_preview=correction,
            )
            from superlocalmemory.learning.behavioral import BehavioralPatternStore
            store = BehavioralPatternStore(engine._db.db_path)
            # The store's write is record_pattern; the pattern key travels in
            # its data. (Before 4.1.21 this called a method the store does not
            # have, so every correction failed.)
            store.record_pattern(
                pid,
                pattern_type="correction",
                data={"pattern_key": pattern_id, "correction": correction},
            )
            authorization.complete()
            return {"success": True, "pattern_id": pattern_id}
        except Exception as exc:
            logger.exception("correct_pattern failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(destructiveHint=True))
    @admits(OperationKind.FORGET)
    async def delete_memory(fact_id: str, agent_id: str = "mcp_client",
                            profile_id: str = "") -> dict:
        """Delete a specific memory by exact fact ID.

        Security note: This is a destructive operation. All deletions are
        logged with the calling agent_id for audit trail. Use get_status or
        list_recent to find fact_ids before deleting.

        Args:
            fact_id: Exact fact ID to delete (from recall or list_recent results).
            agent_id: Identifier of the calling agent (logged for audit).
            profile_id: The profile the memory belongs to (empty = the active
                one). The active profile is not moved.
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        try:
            import asyncio
            import urllib.parse

            from superlocalmemory.cli.daemon import (
                daemon_request,
                is_daemon_running,
            )
            from superlocalmemory.mcp.request_profile import (
                requested_profile,
                routing_needs_daemon_error,
            )

            named = requested_profile(profile_id)
            if await asyncio.to_thread(is_daemon_running):
                from superlocalmemory.mcp.remote_visibility import with_view

                path = "/api/memories/" + urllib.parse.quote(fact_id, safe="")
                if named:
                    path += "?profile_id=" + urllib.parse.quote(named, safe="")
                # A remote app is marked, so the daemon deletes only what that app may see.
                path = with_view(path)
                result = await asyncio.to_thread(_routed_daemon_call, "DELETE", path)
                if isinstance(result, dict) and result.get("code"):
                    return result
                if isinstance(result, dict) and result.get("success"):
                    # The daemon's DELETE route announces it (once, for every
                    # surface); announcing here too showed every MCP delete twice.
                    return {
                        "success": True, "deleted": fact_id,
                        "agent_id": agent_id,
                    }
                return {
                    "success": False,
                    "retryable": True,
                    "error": "resident daemon rejected the delete operation",
                }
            if named:
                # The local worker serves only the active profile.
                return routing_needs_daemon_error()

            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "delete_memory",
                "fact_id": fact_id,
                # Informational IDE/client label only.  The worker derives its
                # authorization actor from the private local capability.
                "source_agent_id": agent_id,
            })
            if result.get("ok"):
                logger.info("Memory deleted: %s by agent: %s", fact_id[:16], agent_id)
                _emit_event("memory.deleted", {
                    "fact_id": fact_id,
                    "agent_id": agent_id,
                }, source_agent=agent_id)
                return {"success": True, "deleted": fact_id, "agent_id": agent_id}
            return {"success": False, "error": result.get("error", "Delete failed")}
        except Exception as exc:
            logger.exception("delete_memory failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(idempotentHint=True))
    @admits(OperationKind.CORRECT)
    async def update_memory(
        fact_id: str, content: str, agent_id: str = "mcp_client",
        profile_id: str = "",
    ) -> dict:
        """Update the content of a specific memory by exact fact ID.

        Security note: All updates are logged with the calling agent_id.
        The fact_id must belong to the profile updated: ``profile_id``, or the
        active profile when it is empty. The active profile is not moved.

        Args:
            fact_id: Exact fact ID to update.
            content: New content for the memory (cannot be empty).
            agent_id: Identifier of the calling agent (logged for audit).
        """
        # v3.6.10: resolve "mcp_client" sentinel → URL path (HTTP) or env var (stdio)
        if agent_id == "mcp_client":
            from superlocalmemory.mcp.agent_context import get_current_agent_id
            agent_id = get_current_agent_id()
        try:
            if not content or not content.strip():
                return {"success": False, "error": "content cannot be empty"}
            import asyncio
            import urllib.parse

            from superlocalmemory.cli.daemon import is_daemon_running
            from superlocalmemory.mcp.request_profile import (
                requested_profile,
                routing_needs_daemon_error,
            )

            named = requested_profile(profile_id)
            if await asyncio.to_thread(is_daemon_running):
                path = "/api/memories/" + urllib.parse.quote(fact_id, safe="")
                body = {"content": content.strip(), **({"profile_id": named} if named else {})}
                # A refusal (unknown profile, a correction already open) comes back
                # with its code and retryable False; only no answer is retryable.
                result = await asyncio.to_thread(_routed_daemon_call, "PATCH", path, body)
                if isinstance(result, dict) and result.get("code"):
                    return result
                if isinstance(result, dict) and result.get("success"):
                    return {
                        "success": True,
                        "predecessor_fact_id": result.get("predecessor_fact_id", fact_id),
                        "successor_fact_id": result.get("successor_fact_id"),
                        "correction_case": result.get("correction_case"),
                        "review_required": bool(result.get("review_required", False)),
                    }
                return {
                    "success": False,
                    "retryable": True,
                    "error": "resident daemon rejected the update operation",
                }
            if named:
                # The local worker serves only the active profile.
                return routing_needs_daemon_error()

            from superlocalmemory.core.worker_pool import WorkerPool
            pool = WorkerPool.shared()
            result = pool._send({
                "cmd": "update_memory",
                "fact_id": fact_id,
                "content": content.strip(),
                "source_agent_id": agent_id,
            })
            if result.get("ok"):
                logger.info("Memory updated: %s by agent: %s", fact_id[:16], agent_id)
                return {
                    "success": True,
                    "predecessor_fact_id": result.get("predecessor_fact_id", fact_id),
                    "successor_fact_id": result.get("successor_fact_id"),
                    "correction_case": result.get("correction_case"),
                    "review_required": bool(result.get("review_required", False)),
                }
            return {"success": False, "error": result.get("error", "Update failed")}
        except Exception as exc:
            logger.exception("update_memory failed")
            return {"success": False, "error": str(exc)}

    @server.tool(annotations=ToolAnnotations(idempotentHint=True))
    @admits(OperationKind.CORRECT)
    async def review_correction(
        case_id: str,
        action: str,
        expected_version: int,
        event_valid_until: str | None = None,
        profile_id: str = "",
    ) -> dict:
        """Apply, reject, or roll back a review-gated correction case.

        The active daemon derives reviewer identity from its local
        authenticated MCP boundary.  Clients provide a case address, a CAS
        version, and an optional reviewer-approved event-time boundary.
        ``profile_id`` names the profile the case belongs to -- the one a
        ``remember(..., profile_id=..., replaces=...)`` was saved to; empty =
        the active profile. Routing never moves the active-profile pointer.
        """
        if action not in {"apply", "reject", "rollback"}:
            return {"success": False, "error": "action must be apply, reject, or rollback"}
        if not isinstance(expected_version, int) or isinstance(expected_version, bool):
            return {"success": False, "error": "expected_version must be an integer"}
        if expected_version < 0:
            return {"success": False, "error": "expected_version must be non-negative"}
        try:
            import asyncio
            import urllib.parse

            from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

            if not await asyncio.to_thread(is_daemon_running):
                return {
                    "success": False,
                    "retryable": True,
                    "error": "correction review requires the resident canonical daemon",
                }
            payload: dict[str, object] = {"expected_version": expected_version}
            if event_valid_until is not None:
                payload["event_valid_until"] = event_valid_until
            if (profile_id or "").strip():
                # Only when set, so a legacy call stays byte-identical.
                payload["profile_id"] = profile_id.strip()
            path = "/api/corrections/" + urllib.parse.quote(case_id, safe="") + "/" + action
            result = await asyncio.to_thread(daemon_request, "POST", path, payload)
            if isinstance(result, dict) and result.get("success"):
                return result
            return {
                "success": False,
                "retryable": True,
                "error": "resident daemon rejected the correction review",
            }
        except Exception:
            logger.exception("review_correction failed")
            return {"success": False, "retryable": True, "error": "correction review unavailable"}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def list_corrections(limit: int = 100, profile_id: str = "") -> dict:
        """List correction cases for a human or host reviewer.

        ``profile_id`` lists another profile's cases (empty = the active
        profile), for a client routed to its own profile with ``remember``.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            return {"success": False, "error": "limit must be an integer from 1 to 500"}
        try:
            import asyncio

            from superlocalmemory.cli.daemon import daemon_request, is_daemon_running

            if not await asyncio.to_thread(is_daemon_running):
                return {
                    "success": False,
                    "retryable": True,
                    "error": "correction review requires the resident canonical daemon",
                }
            import urllib.parse

            path = f"/api/corrections?limit={limit}"
            if (profile_id or "").strip():
                path += "&profile_id=" + urllib.parse.quote(profile_id.strip(), safe="")
            from superlocalmemory.mcp.remote_visibility import with_view

            result = await asyncio.to_thread(daemon_request, "GET", with_view(path))
            if isinstance(result, dict) and result.get("success"):
                return result
            return {
                "success": False,
                "retryable": True,
                "error": "resident daemon rejected correction listing",
            }
        except Exception:
            logger.exception("list_corrections failed")
            return {"success": False, "retryable": True, "error": "correction listing unavailable"}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    async def get_attribution() -> dict:
        """Get system attribution: author, version, license, and provenance metadata."""
        return {
            "success": True,
            "product": "SuperLocalMemory V4",
            "author": "Varun Pratap Bhardwaj",
            "organization": "Qualixar",
            "license": "AGPL-3.0-or-later",
            "urls": {
                "product": "https://superlocalmemory.com",
                "author": "https://varunpratap.com",
                "organization": "https://qualixar.com",
            },
        }


# -- Helpers ------------------------------------------------------------------

def _format_results(results) -> list[dict]:
    """Convert RetrievalResult list to serialisable dicts."""
    items: list[dict] = []
    for r in results:
        items.append({
            "fact_id": r.fact.fact_id,
            "content": r.fact.content,
            "score": round(r.score, 3),
            "confidence": round(r.confidence, 3),
            "relevance_score": round(
                getattr(r, "relevance_score", r.score) or 0.0, 3
            ),
            "ranking_score": getattr(r, "ranking_score", None),
            "memory_confidence": round(
                getattr(r, "memory_confidence", r.confidence) or 0.0, 3
            ),
            "rank_position": int(getattr(r, "rank_position", 0) or 0),
            "trust_score": round(r.trust_score, 3),
            "fact_type": r.fact.fact_type.value,
            "channel_scores": {
                k: round(v, 3) for k, v in r.channel_scores.items()
            },
        })
    return items
