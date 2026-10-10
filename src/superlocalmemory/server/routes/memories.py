# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Memory Routes (AGPL-3.0-or-later).
Routes: /api/memories, /api/graph, /api/search, /api/clusters, /api/clusters/{id}
Uses V3 MemoryEngine for store/recall. Falls back to direct DB for list/graph.
"""
from anyio.from_thread import run as _on_loop  # body of a sync route
import json
import logging
import re
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from superlocalmemory.core.config import BROWSE_PAGE_SIZE
from superlocalmemory.storage.database import (
    visible_fact_clause_for_connection,
)

from .helpers import (
    SearchRequest,
    dict_factory,
    get_active_profile,
    get_db_connection,
    get_engine_lazy,
)

logger = logging.getLogger("superlocalmemory.routes.memories")
router = APIRouter()
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")

# v3.8.3: GENEROUS latency budget for recall. SLM's value is quality recall
# under heavy multi-agent load, so semantic recall is given ample time to
# finish — the keyword fallback is a LAST-RESORT safety net for a genuine hang
# (e.g. a wedged embedder), NOT an aggressive speed cutoff. Only if recall
# exceeds this budget do we serve the fast keyword search so the caller ALWAYS
# gets a result instead of hanging forever. Tune with SLM_SEARCH_RECALL_TIMEOUT_S.
# The dashboard's fetch timeout is set ABOVE this so the browser waits for the
# quality result rather than aborting early.
_DEFAULT_RECALL_BUDGET_S = 25.0


def _search_recall_timeout_s() -> float:
    import os
    try:
        v = float(os.environ.get("SLM_SEARCH_RECALL_TIMEOUT_S", ""))
        return v if v > 0 else _DEFAULT_RECALL_BUDGET_S
    except (TypeError, ValueError):
        return _DEFAULT_RECALL_BUDGET_S


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log the full traceback server-side; return a generic message.

    Returning ``str(e)`` to HTTP clients leaks DB schema (column/constraint
    names), the data-directory filesystem path, and possibly config internals —
    a GDPR Art. 32 gap. Call this only from inside an ``except`` block so
    ``logger.exception`` captures the active traceback automatically.
    """
    logger.exception("memories route error")
    return HTTPException(status_code=500, detail=detail)


def _get_engine(request: Request):
    """Get V3 engine from app state, initializing lazily on first call."""
    return get_engine_lazy(request.app.state)


def _canonical_mutation_runtime(request: Request):
    """Return the daemon-owned mutation boundary or fail before queueing work."""
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if runtime is None or not runtime.ready:
        raise HTTPException(503, detail="canonical mutation writer is not ready; retry shortly")
    return runtime


def _mutation_idempotency_key(request: Request) -> str:
    """Accept a client retry key without making one mandatory for the dashboard."""
    key = request.headers.get("X-Idempotency-Key", "").strip() or str(uuid.uuid4())
    if not _IDEMPOTENCY_KEY.fullmatch(key):
        raise HTTPException(
            422,
            detail="X-Idempotency-Key must contain 1-256 safe characters",
        )
    return key


def _canonical_mutation_error(exc: Exception, detail: str) -> HTTPException:
    """Map typed mutation failures without leaking SQLite or filesystem detail."""
    from superlocalmemory.core.remember_runtime import (
        CanonicalMutationConflict,
        CanonicalRememberUnavailable,
        MutationTargetMissing,
    )

    if isinstance(exc, MutationTargetMissing):
        # Generic on purpose: the caller has already passed authorization, and
        # the specific routes say what was missing.
        return HTTPException(404, detail="Not found")
    if isinstance(exc, CanonicalMutationConflict):
        return HTTPException(409, detail=str(exc))
    if isinstance(exc, CanonicalRememberUnavailable):  # contention: nothing changed
        return HTTPException(503, headers={"Retry-After": "3"}, detail=(
            "the memory store is busy finishing other writes; nothing was changed. "
            "Retry in a few seconds"))
    return _internal_error(detail)


def _mutation_runtime_or_missing_fact(
    request: Request, engine, profile_id: str, fact_id: str,
):
    """Retain the public 404 for a missing fact without creating a local writer."""
    runtime = getattr(request.app.state, "canonical_remember_runtime", None)
    if runtime is not None and runtime.ready:
        return runtime
    rows = engine._db.execute(
        "SELECT 1 FROM atomic_facts WHERE fact_id = ? AND profile_id = ?",
        (fact_id, profile_id),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Memory not found")
    raise HTTPException(503, detail="canonical mutation writer is not ready; retry shortly")


def _admit_http_mutation(
    request: Request, operation: str, *, admission_kind: "OperationKind | None" = None,
    profile: str | None = None,
) -> None:
    """Route a memory HTTP mutation through OperationPolicyRegistry.evaluate().

    Called from _authorize_memory_mutation after RBAC passes. Raises HTTP 403
    if the policy registry denies the actor. Maps "delete" → FORGET,
    "update" → CORRECT (owner/admin only — this is a content *correction*,
    which is why ``/replaces`` stays on CORRECT). Uses the server-derived
    principal and its roles on ``profile`` (default: the active profile) --
    the same profile RBAC just checked, so a role held elsewhere never stands
    in for it.

    ``admission_kind`` lets a caller that passes operation="update" for its
    RBAC permission (WRITE) and hook name (unchanged, so the trust-gate
    pre-hook still fires) override which OperationKind the policy registry
    actually evaluates. Memory-kinds' set/confirm routes use this: they are
    documented as a WRITE-level operation on a fact the active profile owns
    (consistent with ``remember(kind=...)``), not the owner/admin-only
    CORRECT contract a content edit requires — so they pass
    ``admission_kind=OperationKind.REMEMBER`` instead of letting "update"
    default to CORRECT.
    """
    from fastapi import HTTPException as _HTTPException

    from superlocalmemory.core.actor_context import Transport
    from superlocalmemory.core.admission import AdmissionDenied, admit, resolve_actor
    from superlocalmemory.core.operation_request import OperationKind
    from superlocalmemory.server.rbac_enforce import (
        resolve_actor_roles,
        resolve_principal,
    )

    from superlocalmemory.core.admission import _company_mode_active

    # Both switches, the same rule the other entry points use. Reading
    # config.toml alone left this path treating a workspace as personal after
    # the dashboard toggle had turned per-user access on -- so a write refused
    # everywhere else was admitted here.
    deployment = getattr(request.app.state, "deployment", None)
    if deployment is None:
        from superlocalmemory.core.admission import _resolve_deployment

        deployment = _resolve_deployment()
    company = _company_mode_active(deployment)
    tier = "enterprise" if company else "personal"
    mode = "company" if company else "local"

    principal_info = resolve_principal(request)
    principal = str(principal_info.get("user_id") or "")
    actor_roles = resolve_actor_roles(request, profile=profile)
    actor = resolve_actor(
        Transport.HTTP,
        tier=tier,
        mode=mode,
        principal=principal,
        roles=actor_roles,
    )
    if admission_kind is not None:
        kind = admission_kind
    else:
        kind = OperationKind.FORGET if operation == "delete" else OperationKind.CORRECT
    try:
        admit(kind, actor, mode=mode)
    except AdmissionDenied as exc:
        raise _HTTPException(
            status_code=403,
            detail=f"Operation denied: {exc.decision.reason}",
        ) from exc


class _UnknownRoutedProfile(LookupError):
    """A request routed to a profile that does not exist."""

    def __init__(self, profile_id: str) -> None:
        super().__init__(profile_id)
        self.profile_id = profile_id


def _unknown_profile_response(profile_id: str):
    """The same 404 body POST /remember gives for an unknown routed profile."""
    from starlette.responses import JSONResponse

    from superlocalmemory.server.routed_profile import unknown_profile_body

    return JSONResponse(unknown_profile_body(profile_id), status_code=404)


def _announce(event_type: str, payload: dict, actor: str) -> None:
    """Tell the live event stream about a delete or a correction.

    Announced here, by the daemon route every surface uses (dashboard, CLI,
    MCP), so each change is seen once whichever surface made it. Best-effort:
    the event bus never affects the answer.
    """
    from superlocalmemory.server.unified_daemon import _emit_event

    _emit_event(event_type, payload, source_agent=actor or "dashboard")


def _routed_mutation_error(exc: Exception, profile: str | None, detail: str):
    """The answer to a failed routed mutation: the routed profile was deleted
    between the route's check and the writer's own is an unknown profile, not
    a server error. Anything else maps as for an active-profile mutation."""
    from superlocalmemory.core.remember_runtime import UnknownMutationProfile

    if profile is not None and isinstance(exc, (UnknownMutationProfile, _UnknownRoutedProfile)):
        return _unknown_profile_response(profile)
    raise _canonical_mutation_error(exc, detail)


def _routed_profile(value) -> str | None:
    """A request's ``profile_id``: None means the active profile; not text is a 422."""
    from superlocalmemory.server.routed_profile import RoutedProfileError, routed_profile_id

    try:
        return routed_profile_id(value)
    except RoutedProfileError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


def _authorize_memory_mutation(
    request: Request,
    operation: str,
    fact_id: str,
    *,
    content_preview: str = "",
    run_pre_hook: bool = True,
    admission_kind: "OperationKind | None" = None,
    profile: str | None = None,
):
    """Authenticate a mutation, optionally gating route-owned direct SQL.

    ``admission_kind`` is forwarded to ``_admit_http_mutation`` unchanged; see
    its docstring. It does not affect the RBAC permission (still keyed off
    ``operation``) or the hook name run below (still ``operation``) — only
    which ``OperationKind`` the policy registry evaluates.

    ``profile`` is a per-request routed profile, or None for the active one.
    Routed, the caller's role and the operation policy are checked on THAT
    profile, and only then is its existence revealed (``_UnknownRoutedProfile``),
    so a caller without access cannot probe which profiles exist.
    """
    from superlocalmemory.server.write_identity import require_write_actor

    actor_id = require_write_actor(
        request,
        getattr(request.app.state, "daemon_descriptor", None),
        actor_kind="dashboard",
    )
    # RBAC (C3): on top of machine auth, enforce the caller's role on the
    # profile acted on. delete → DELETE; every other mutation → WRITE.
    from superlocalmemory.access.rbac import Permission as _Perm
    from superlocalmemory.server.rbac_enforce import require_permission as _rbac_require
    _rbac_require(
        request,
        _Perm.DELETE if operation == "delete" else _Perm.WRITE,
        profile=profile,
    )
    # Phase 1: admission gateway — policy registry decision for this route.
    _admit_http_mutation(
        request, operation, admission_kind=admission_kind, profile=profile,
    )
    engine = _get_engine(request)
    if engine is None:
        raise HTTPException(503, detail="Engine not initialized")
    if profile is not None:
        if not engine._db.execute(
            "SELECT 1 AS one FROM profiles WHERE profile_id = ?", (profile,),
        ):
            raise _UnknownRoutedProfile(profile)
        logger.info("per-request profile routing: %s %s profile=%s",
                    request.method, request.url.path, profile)
    profile_id = profile or engine.profile_id
    context = {
        "operation": operation,
        "agent_id": actor_id,
        "source_agent_id": "dashboard",
        "profile_id": profile_id,
        "fact_id": fact_id,
    }
    if content_preview:
        context["content_preview"] = content_preview[:100]
    if run_pre_hook:
        try:
            engine._hooks.run_pre(operation, context)
        except Exception as exc:
            logger.warning("Dashboard %s authorization rejected: %s", operation, exc)
            raise HTTPException(403, detail="Write authorization rejected") from exc
    return engine, profile_id, context


def _preview(content: str | None) -> str:
    """Truncate content for preview display."""
    if not content:
        return ""
    return content[:100] + "..." if len(content) > 100 else content


def _scope_where_clause(scope: str | None, active_profile: str) -> tuple[str, list]:
    """The v3 "which rows is this view allowed to see" WHERE fragment + params.

    Shared by ``get_memories`` and ``get_memory_kind_counts`` so the counts a
    kind chip shows are drawn from exactly the rows that filtering by that
    chip would then list — a second hand-written copy of this is how the
    withheld-rows leak (``visible_fact_clause_for_connection``, see the
    comment above it in ``get_memories``) happened in the first place.
    """
    esc = active_profile.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    shared_pat = f'%"{esc}"%'
    if scope == "global":
        return "scope = 'global'", []
    if scope == "shared":
        return "scope = 'shared' AND shared_with LIKE ? ESCAPE '\\'", [shared_pat]
    if scope == "all":
        return (
            "(profile_id = ? OR scope = 'global' "
            "OR (scope = 'shared' AND shared_with LIKE ? ESCAPE '\\'))"
        ), [active_profile, shared_pat]
    return "profile_id = ?", [active_profile]


def _has_table(cursor, name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cursor.fetchone() is not None
    except Exception:
        return False


def _fetch_graph_data(
    cursor, profile: str, use_v3: bool, min_importance: int, max_nodes: int,
) -> tuple[list, list, list]:
    """Fetch graph nodes, links, clusters from V3 or V2 schema."""
    if use_v3:
        # Recency-first: get the most recent nodes, then find their edges.
        # LEFT JOIN fact_importance for graph metrics (v3.4.1 — additive only).
        cursor.execute("""
            SELECT af.fact_id as id, af.content, af.fact_type as category,
                   af.confidence as importance, af.session_id as project_name,
                   af.created_at,
                   fi.pagerank_score, fi.community_id, fi.degree_centrality
            FROM atomic_facts af
            LEFT JOIN fact_importance fi
                ON af.fact_id = fi.fact_id AND fi.profile_id = ?
            WHERE af.profile_id = ? AND af.confidence >= ?
            ORDER BY af.created_at DESC
            LIMIT ?
        """, (profile, profile, min_importance / 10.0, max_nodes))
        nodes = cursor.fetchall()

        node_ids = {n['id'] for n in nodes}

        # Fetch edges between these nodes
        if node_ids:
            ph = ','.join('?' * len(node_ids))
            id_list = list(node_ids)
            cursor.execute(f"""
                SELECT source_id as source, target_id as target,
                       weight, edge_type as relationship_type
                FROM graph_edges
                WHERE profile_id = ?
                  AND source_id IN ({ph}) AND target_id IN ({ph})
                ORDER BY weight DESC
            """, [profile] + id_list + id_list)
            all_links = cursor.fetchall()
        else:
            all_links = []

        links = all_links
        for n in nodes:
            n['entities'] = []
            n['content_preview'] = _preview(n.get('content'))
            # v3.4.1: Default graph metrics when fact_importance has no data
            if n.get('pagerank_score') is None:
                n['pagerank_score'] = 0.0
            if n.get('community_id') is None:
                n['community_id'] = 0
            if n.get('degree_centrality') is None:
                n['degree_centrality'] = 0.0

        # Filter edges to only those between displayed nodes
        node_ids = {n['id'] for n in nodes}
        links = [lk for lk in all_links
                 if lk['source'] in node_ids and lk['target'] in node_ids]

        # Compute clusters from memory_scenes
        clusters = []
        try:
            cursor.execute("""
                SELECT scene_id, theme, fact_ids_json
                FROM memory_scenes WHERE profile_id = ?
            """, (profile,))
            for row in cursor.fetchall():
                fact_ids = []
                try:
                    fact_ids = json.loads(row.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                # Only include clusters that overlap with displayed nodes
                overlap = [fid for fid in fact_ids if fid in node_ids]
                if overlap:
                    clusters.append({
                        'cluster_id': row['scene_id'],
                        'size': len(fact_ids),
                        'visible_size': len(overlap),
                        'theme': row.get('theme', ''),
                    })
        except Exception:
            pass

        return nodes, links, clusters

    # V2 fallback
    try:
        cursor.execute("""
            SELECT m.id, m.content, m.summary, m.category, m.cluster_id,
                   m.importance, m.project_name, m.created_at, m.tags, gn.entities
            FROM memories m LEFT JOIN graph_nodes gn ON m.id = gn.memory_id
            WHERE m.importance >= ? AND m.profile = ?
            ORDER BY m.importance DESC, m.updated_at DESC LIMIT ?
        """, (min_importance, profile, max_nodes))
    except Exception:
        cursor.execute("""
            SELECT id, content, summary, category, cluster_id, importance,
                   project_name, created_at, tags, NULL as entities
            FROM memories WHERE importance >= ? AND profile = ?
            ORDER BY importance DESC, updated_at DESC LIMIT ?
        """, (min_importance, profile, max_nodes))
    nodes = cursor.fetchall()
    for n in nodes:
        ent = n.get('entities')
        n['entities'] = json.loads(ent) if ent else []
        n['content_preview'] = _preview(n.get('content'))
    ids = [n['id'] for n in nodes]
    links = _fetch_edges_v2(cursor, ids)
    try:
        cursor.execute("""
            SELECT cluster_id, COUNT(*) as size, AVG(importance) as avg_importance
            FROM memories WHERE cluster_id IS NOT NULL AND profile = ?
            GROUP BY cluster_id
        """, (profile,))
        clusters = cursor.fetchall()
    except Exception:
        clusters = []
    return nodes, links, clusters


def _fetch_edges_v3(cursor, profile: str, fact_ids: list) -> list:
    if not fact_ids:
        return []
    ph = ','.join('?' * len(fact_ids))
    try:
        cursor.execute(f"""
            SELECT source_id as source, target_id as target,
                   weight, edge_type as relationship_type
            FROM graph_edges WHERE profile_id = ?
              AND source_id IN ({ph}) AND target_id IN ({ph})
            ORDER BY weight DESC
        """, [profile] + fact_ids + fact_ids)
        return cursor.fetchall()
    except Exception:
        return []


def _fetch_edges_v2(cursor, memory_ids: list) -> list:
    if not memory_ids:
        return []
    ph = ','.join('?' * len(memory_ids))
    try:
        cursor.execute(f"""
            SELECT source_memory_id as source, target_memory_id as target,
                   weight, relationship_type, shared_entities
            FROM graph_edges
            WHERE source_memory_id IN ({ph}) AND target_memory_id IN ({ph})
            ORDER BY weight DESC
        """, memory_ids + memory_ids)
        links = cursor.fetchall()
        for lk in links:
            se = lk.get('shared_entities')
            if se:
                try:
                    lk['shared_entities'] = json.loads(se)
                except Exception:
                    lk['shared_entities'] = []
        return links
    except Exception:
        return []


@router.get("/api/memories")
def get_memories(
    request: Request,
    category: Optional[str] = None,
    kind: Optional[str] = Query(
        None,
        description=(
            "One of the nine memory kinds (core/storage/memory_kinds.py), "
            "e.g. 'decision' or 'rule'. v3 stores only — a legacy (v2) store "
            "has no memory_kind column and ignores this filter."
        ),
    ),
    project_name: Optional[str] = None,
    cluster_id: Optional[int] = None,
    min_importance: Optional[int] = None,
    tags: Optional[str] = None,
    limit: int = Query(BROWSE_PAGE_SIZE, ge=1, le=200),
    offset: int = Query(0, ge=0),
    filter: Optional[str] = Query(
        None,
        description="Named filter: 'high_reward' | 'being_forgotten'",
    ),
    scope: Optional[str] = Query(
        None,
        description=(
            "Scope view (v3 only): 'shared' (shared with this profile), "
            "'global' (global memories), 'all' (this profile + global + shared). "
            "Default None = this profile only (unchanged isolation)."
        ),
    ),
):
    """List memories with optional filtering and pagination.

    S9-DASH-07: ``filter`` enables dashboard "learning-visible" views:

    * ``high_reward``: facts cited by ``action_outcomes`` with
      ``reward >= 0.7`` in the last 30 days. Surfaces what the ranker
      is actually learning from.
    * ``being_forgotten``: facts in ``archive_status='archived'`` OR
      with ``lifecycle='cold'`` AND no positive reward in 60 days.
      Makes "memory decay" tangible to the operator.
    """
    conn = None
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        use_v3 = _has_table(cursor, 'atomic_facts')

        if use_v3:
            # Scope view clause. Default (scope=None) is this-profile-only —
            # identical isolation to before. Other values widen the view to
            # global and/or shared-with-this-profile memories.
            scope_where, scope_params = _scope_where_clause(scope, active_profile)

            # Withheld rows are excluded from BOTH the page and the total. On
            # the author's store this list served 24 model-authored summaries
            # in its first 50 rows and reported 5,218 memories against 3,919
            # real ones -- the count a user reads, inflated by exactly the
            # 1,299 rows 4.0.10 withheld.
            visible = visible_fact_clause_for_connection(conn)
            # project_name: NOT session_id (that is a different field — see
            # mcp/tools_core.py's remember(), which stores them separately in
            # the same metadata dict). The project a memory was saved under
            # (remember(project=...), `slm remember --project`) lives in
            # memories.metadata_json->>'project' (core/project_identity.py
            # storable_project()); retrieval.project_scope.stored_projects()
            # reads it the same way for project-aware recall. This mirrors
            # that lookup as a correlated subquery instead of a JOIN so the
            # unqualified column names below (content, category, created_at...)
            # can't collide with the legacy v2 `memories` table's own columns
            # of the same name.
            # fact_type is selected TWICE on purpose: once aliased "category"
            # (unchanged, pre-existing display field) and once under its own
            # name, because storage.memory_kinds.kind_fields() reads a
            # "fact_type" key for its legacy-kind fallback — it does not know
            # about this route's "category" alias. memory_kind/_source/
            # _confidence are the M052 columns kind_fields reads for the
            # confirmed/suggested branches.
            query = (
                "SELECT fact_id as id, memory_id, content, fact_type as category, "
                "fact_type, memory_kind, memory_kind_source, memory_kind_confidence, "
                "confidence as importance, access_count, "
                "created_at, created_at as updated_at, "
                "(SELECT CASE WHEN json_valid(m.metadata_json) "
                " THEN json_extract(m.metadata_json, '$.project') END "
                " FROM memories m WHERE m.memory_id = atomic_facts.memory_id) "
                "as project_name, scope, shared_with "
                f"FROM atomic_facts WHERE {scope_where}{visible}"
            )
            params = list(scope_params)
            count_base = (
                "SELECT COUNT(*) as total FROM atomic_facts "
                f"WHERE {scope_where}{visible}"
            )
        else:
            query = """
                SELECT id, content, summary, category, project_name, project_path,
                       importance, cluster_id, depth, access_count, parent_id,
                       created_at, updated_at, last_accessed, tags, memory_type
                FROM memories WHERE profile = ?
            """
            params = [active_profile]
            count_base = "SELECT COUNT(*) as total FROM memories WHERE profile = ?"

        # count params must mirror the base WHERE params (scope-aware for v3).
        count_params = list(params)

        if category:
            if use_v3:
                query += " AND fact_type = ?"
            else:
                query += " AND category = ?"
            params.append(category)
            count_base += " AND category = ?" if not use_v3 else " AND fact_type = ?"
            count_params.append(category)
        if project_name:
            if use_v3:
                # Same metadata_json->>'project' lookup as the SELECT above,
                # as an EXISTS so it works unchanged in the COUNT query too.
                _project_filter = (
                    " AND EXISTS (SELECT 1 FROM memories m "
                    "WHERE m.memory_id = atomic_facts.memory_id "
                    "AND json_valid(m.metadata_json) "
                    "AND json_extract(m.metadata_json, '$.project') = ?)"
                )
                query += _project_filter
                count_base += _project_filter
            else:
                query += " AND project_name = ?"
                count_base += " AND project_name = ?"
            params.append(project_name)
            count_params.append(project_name)
        if cluster_id is not None and not use_v3:
            query += " AND cluster_id = ?"
            params.append(cluster_id)
            count_base += " AND cluster_id = ?"
            count_params.append(cluster_id)
        if min_importance:
            if use_v3:
                query += " AND confidence >= ?"
                params.append(min_importance / 10.0)
            else:
                query += " AND importance >= ?"
                params.append(min_importance)
        if tags and not use_v3:
            tag_list = [t.strip() for t in tags.split(',')]
            for tag in tag_list:
                query += " AND tags LIKE ?"
                params.append(f'%{tag}%')

        # S9-DASH-07: named filters — "high_reward" and "being_forgotten".
        # Only supported on the v3 (atomic_facts) path — v2 fallback
        # ignores the flag silently.
        if filter and use_v3:
            if filter == "high_reward":
                query += (
                    " AND fact_id IN ("
                    "  SELECT DISTINCT json_each.value"
                    "  FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "  WHERE action_outcomes.reward >= 0.7"
                    "    AND datetime(action_outcomes.settled_at) >= "
                    "        datetime('now', '-30 day')"
                    ")"
                )
                count_base += (
                    " AND fact_id IN ("
                    "  SELECT DISTINCT json_each.value"
                    "  FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "  WHERE action_outcomes.reward >= 0.7"
                    "    AND datetime(action_outcomes.settled_at) >= "
                    "        datetime('now', '-30 day')"
                    ")"
                )
            elif filter == "being_forgotten":
                # Cold / archived + no recent positive reward.
                query += (
                    " AND ("
                    "  archive_status = 'archived' OR "
                    "  (lifecycle = 'cold' AND fact_id NOT IN ("
                    "    SELECT DISTINCT json_each.value"
                    "    FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "    WHERE action_outcomes.reward >= 0.5"
                    "      AND datetime(action_outcomes.settled_at) >= "
                    "          datetime('now', '-60 day')"
                    "  ))"
                    ")"
                )
                count_base += (
                    " AND ("
                    "  archive_status = 'archived' OR "
                    "  (lifecycle = 'cold' AND fact_id NOT IN ("
                    "    SELECT DISTINCT json_each.value"
                    "    FROM action_outcomes, json_each(action_outcomes.fact_ids_json)"
                    "    WHERE action_outcomes.reward >= 0.5"
                    "      AND datetime(action_outcomes.settled_at) >= "
                    "          datetime('now', '-60 day')"
                    "  ))"
                    ")"
                )

        kind_truncated = False
        if kind and use_v3:
            # The nine kinds aren't a SQL-expressible predicate — a row's
            # DISPLAYED kind depends on confirmed/suggested/confidence-
            # threshold/legacy-fallback precedence (kind_fields), the exact
            # same rule `slm list`/recall use (core/kind_query.py). So this
            # grows the SAME WHERE-clause fetch window core.kind_query's own
            # `_windowed_kind_fetch` uses, classifies each row in Python, and
            # paginates the filtered result — rather than re-deriving the
            # precedence as a one-off SQL CASE that could drift from theirs.
            from superlocalmemory.core.kind_query import (
                WINDOWED_FETCH_HARD_CAP,
                InvalidKind,
                engine_display_min_confidence,
                resolve_kind,
            )
            from superlocalmemory.retrieval.kind_filter import overfetch_limit
            from superlocalmemory.storage.memory_kinds import kind_fields

            try:
                resolved_kind = resolve_kind(kind)
            except InvalidKind as exc:
                raise HTTPException(400, detail=str(exc)) from exc

            if resolved_kind is None:
                memories, total = [], 0
            else:
                threshold = engine_display_min_confidence(_get_engine(request))
                needed = offset + limit
                window = min(max(overfetch_limit(needed), needed), WINDOWED_FETCH_HARD_CAP)
                matched: list = []
                while True:
                    cursor.execute(
                        query + " ORDER BY created_at DESC LIMIT ?", params + [window],
                    )
                    batch = cursor.fetchall()
                    for row in batch:
                        row.update(kind_fields(row, display_min_confidence=threshold))
                    matched = [r for r in batch if r.get("memory_kind") == resolved_kind]
                    if len(matched) >= needed or len(batch) < window:
                        kind_truncated = (
                            len(matched) < needed
                            and len(batch) >= window
                            and window >= WINDOWED_FETCH_HARD_CAP
                        )
                        break
                    if window >= WINDOWED_FETCH_HARD_CAP:
                        kind_truncated = True
                        break
                    window = min(window * 4, WINDOWED_FETCH_HARD_CAP)
                total = len(matched)
                memories = matched[offset:offset + limit]
        else:
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            memories = cursor.fetchall()

            cursor.execute(count_base, count_params)
            total = cursor.fetchone()['total']

            if use_v3:
                from superlocalmemory.core.kind_query import engine_display_min_confidence
                from superlocalmemory.storage.memory_kinds import kind_fields

                threshold = engine_display_min_confidence(_get_engine(request))
                for row in memories:
                    row.update(kind_fields(row, display_min_confidence=threshold))

        response = {
            "memories": memories, "total": total,
            "limit": limit, "offset": offset,
            "has_more": (offset + limit) < total,
        }
        if kind and use_v3:
            # Honest about the bound: more matches may exist past the window
            # this looked at (same contract as core.kind_query.list_recent_facts'
            # own `truncated` out-param) — never silently under-reported.
            response["kind_filter_truncated"] = kind_truncated
        return response

    except HTTPException:
        # An explicit, already-correct response (e.g. the 400 above for an
        # unparseable `kind`) — must not be flattened into a generic 500.
        raise
    except Exception:
        raise _internal_error("Database error")
    finally:
        if conn is not None:  # also on the 400 and 500 paths
            conn.close()


@router.get("/api/memories/kind-counts")
def get_memory_kind_counts(
    request: Request,
    scope: Optional[str] = Query(
        None, description="Same 'shared'|'global'|'all' scope view as /api/memories.",
    ),
):
    """How many visible memories display as each of the nine kinds.

    Backs the Memories -> All memories filter chips. One pass over the
    (lightweight — four columns, no content) visible rows classifies each
    with the exact same ``kind_fields`` precedence ``GET /api/memories``'s
    own ``kind=`` filter uses, instead of the dashboard firing one
    ``?category=X&limit=1`` request per legacy fact_type (which could only
    ever count the four legacy buckets, never the nine kinds that fan out
    from them).
    """
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        if not _has_table(cursor, 'atomic_facts'):
            # A legacy (v2) store has no memory_kind column at all.
            conn.close()
            return {"counts": {}, "truncated": False}

        from superlocalmemory.core.kind_query import (
            WINDOWED_FETCH_HARD_CAP,
            engine_display_min_confidence,
        )
        from superlocalmemory.storage.memory_kinds import kind_fields

        scope_where, scope_params = _scope_where_clause(scope, active_profile)
        visible = visible_fact_clause_for_connection(conn)
        cursor.execute(
            "SELECT fact_type, memory_kind, memory_kind_source, memory_kind_confidence "
            f"FROM atomic_facts WHERE {scope_where}{visible} "
            "ORDER BY created_at DESC LIMIT ?",
            scope_params + [WINDOWED_FETCH_HARD_CAP + 1],
        )
        rows = cursor.fetchall()
        conn.close()

        truncated = len(rows) > WINDOWED_FETCH_HARD_CAP
        rows = rows[:WINDOWED_FETCH_HARD_CAP]

        threshold = engine_display_min_confidence(_get_engine(request))
        counts: dict[str, int] = {}
        for row in rows:
            fields = kind_fields(row, display_min_confidence=threshold)
            kind_value = fields["memory_kind"]
            if kind_value is None:  # "untyped" — no kind and no mappable legacy type
                continue
            counts[kind_value] = counts.get(kind_value, 0) + 1

        return {"counts": counts, "truncated": truncated}

    except Exception:
        raise _internal_error("Database error")


@router.get("/api/graph")
def get_graph(
    request: Request,
    max_nodes: int = Query(100, ge=10, le=10000),
    min_importance: int = Query(1, ge=1, le=10),
):
    """Get knowledge graph data for D3.js force-directed visualization."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        use_v3 = _has_table(cursor, 'atomic_facts')

        nodes, links, clusters = _fetch_graph_data(
            cursor, active_profile, use_v3, min_importance, max_nodes,
        )

        conn.close()

        return {
            "nodes": nodes, "links": links, "clusters": clusters,
            "metadata": {
                "node_count": len(nodes), "edge_count": len(links),
                "cluster_count": len(clusters) if clusters else 0,
                "filters_applied": {"max_nodes": max_nodes, "min_importance": min_importance},
            },
        }

    except Exception:
        raise _internal_error("Graph error")


@router.post("/api/search")
async def search_memories(request: Request, body: SearchRequest):
    """Semantic search using the daemon's in-process engine.

    v3.4.61: Replaced WorkerPool.shared() (subprocess-based, cold-starts on
    every request, always >15s) with the daemon's own engine that is already
    loaded and warm. WorkerPool.shared() was legacy from v3.4.32 before the
    unified daemon architecture. Using the daemon engine matches what the /recall
    HTTP endpoint does and shares its warm SQLite page cache, bringing dashboard
    search from >15s timeout to <1s warm.

    Falls back to direct DB LIKE search if engine is unavailable.

    ``kind`` (L3-20) narrows to one of the nine memory kinds, the same
    filter MCP's ``search`` and ``list_recent`` already take. Refused
    (422, INVALID_KIND) before any retrieval if it does not parse.
    """
    from superlocalmemory.core.kind_query import InvalidKind, resolve_kind
    try:
        parsed_kind = resolve_kind(getattr(body, "kind", ""))
    except InvalidKind as exc:
        from superlocalmemory.server.kind_error import invalid_kind_http
        raise invalid_kind_http(exc)
    # T-window: prefer an explicit ``window`` spec; otherwise derive a range
    # from the legacy date_from/date_to pair when both are set. 4.1.20 (R5):
    # checked here, before anything runs — an unreadable one is a 400, never a
    # search that silently ignores it.
    _window = getattr(body, "window", None) or ""
    if not _window and getattr(body, "date_from", None) and getattr(body, "date_to", None):
        _window = f"{body.date_from}..{body.date_to}"
    from superlocalmemory.server.time_filter_error import checked_window_or_400
    _window = checked_window_or_400(_window)
    if not isinstance(_window, str):
        return _window
    from superlocalmemory.core.recall_gate import begin_recall, end_recall
    begin_recall()
    try:
        # Use the daemon engine directly — already loaded, shares warm cache.
        # v3.4.63: engine.recall() is synchronous/blocking (~2-10s). Calling it
        # directly in an async route blocks the ASGI event loop — Chrome detects
        # a stalled connection and aborts with "signal is aborted without reason"
        # before the response arrives. Fix: run in a thread-pool executor so the
        # event loop stays alive to send keepalive frames.
        # v3.8.2: fast=True — the dashboard search BOX is a snappy retrieval
        # list (all six local channels + reranker), never the internal agentic
        # LLM round, which would reintroduce the multi-second hang this endpoint
        # is regression-tested against (test_search_fast_param_and_profile_isolation).
        # The human-facing LLM synthesis lives on separate paths that are NOT the
        # search list: the "ask" memory-chat (/api/v3/chat/stream, Ollama Mode B)
        # and the precomputed knowledge-cluster summaries (core.community_summary,
        # Mode B/C). So search stays fast; synthesis is where the LLM adds value.
        import asyncio
        import time as _time
        engine = _get_engine(request)
        if engine is not None:
            loop = asyncio.get_running_loop()
            t0 = _time.monotonic()
            # v3.8.3: bound the synchronous recall. Under a concurrent
            # maintenance pass or a busy embedder it can run tens of seconds
            # and the browser aborts the fetch. If it exceeds the budget we
            # fall through to the fast keyword search below, so the dashboard
            # ALWAYS returns instead of failing with an abort.
            # A kind filter narrows AFTER retrieval (below), so ask for more
            # than ``body.limit`` up front — the same over-fetch contract
            # core.kind_query's list/search helpers use for the other two
            # doors — or a kind-filtered search could come back with fewer
            # than ``limit`` results even when enough actually exist.
            from superlocalmemory.retrieval.kind_filter import overfetch_limit
            _search_limit = overfetch_limit(body.limit) if parsed_kind else body.limit
            # Tagged as a dashboard test, so it never counts as agent traffic.
            from superlocalmemory.core.answer_check_history import call_as_dashboard
            _recall_future = loop.run_in_executor(
                None,
                lambda: call_as_dashboard(
                    engine.recall, body.query, limit=_search_limit, fast=True,
                    window=_window or None,
                    # Name the surface. A recall with no name leaves no record
                    # an outcome can be matched to, and a search typed into the
                    # dashboard is one continuous thread of use, not a
                    # conversation turn.
                    session_id=f"dashboard:{get_active_profile()}",
                ),
            )
            # A run_in_executor thread cannot be cancelled, and wait_for() on it
            # blocks until the thread finishes (defeating the timeout). So poll
            # the future without blocking the event loop and give up at the
            # deadline — the orphaned recall completes in the background and its
            # result is discarded. This is what bounds dashboard-search latency.
            _budget = _search_recall_timeout_s()
            _deadline = loop.time() + _budget
            while not _recall_future.done() and loop.time() < _deadline:
                await asyncio.sleep(0.05)
            if _recall_future.done():
                response = _recall_future.result()
            else:
                # Ensure the orphaned future's eventual result/exception is
                # retrieved so asyncio doesn't log "never retrieved".
                _recall_future.add_done_callback(
                    lambda f: (f.cancelled() or f.exception())
                )
                logger.warning(
                    "search_memories: semantic recall exceeded %.0fs budget for "
                    "%r — serving keyword fallback",
                    _budget, (body.query or "")[:80],
                )
                response = None
            if response is not None:
                elapsed_ms = round((_time.monotonic() - t0) * 1000, 1)
                from superlocalmemory.core.kind_query import engine_display_min_confidence
                from superlocalmemory.server.recall_serializer import (
                    recall_response_metadata,
                    serialize_recall_response,
                )
                results, no_confident_match = serialize_recall_response(
                    response,
                    limit=_search_limit,
                    per_fact_max=300,
                    total_max=max(300, body.limit * 300),
                    display_min_confidence=engine_display_min_confidence(engine),
                )
                if parsed_kind:
                    results = [r for r in results
                              if r.get("memory_kind") == parsed_kind][:body.limit]
                return {
                    "query": body.query,
                    "results": results,
                    "total": len(results),
                    "query_type": getattr(response, "query_type", "semantic"),
                    "retrieval_time_ms": elapsed_ms,
                    "no_confident_match": no_confident_match,
                    **recall_response_metadata(response),
                }
            # recall timed out — fall through to the fast keyword search.

        # Fallback: direct DB text search (engine not ready OR recall over budget)
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()
        # Degraded-lexical fallback: a plain equality filter, not the
        # over-fetch-then-trim windowing core.kind_query's helpers use for
        # list/search's primary paths (this is a raw connection, not a
        # DatabaseManager, and this path only runs when the main engine is
        # unavailable or over its time budget). Stores without the M052 kind
        # columns (has_kind_columns False) ignore the filter rather than
        # erroring on an unknown column.
        has_kind_columns = any(
            row.get("name") == "memory_kind" for row in cursor.execute(
                "PRAGMA table_info(atomic_facts)").fetchall()
        )
        apply_kind_filter = bool(parsed_kind) and has_kind_columns
        kind_clause = " AND memory_kind = ?" if apply_kind_filter else ""
        kind_params = (parsed_kind,) if apply_kind_filter else ()
        # memory_kind_source/_confidence and the bare fact_type (selected
        # again, alongside the existing "fact_type as category" alias) are
        # what kind_fields() below needs for its confirmed/suggested/legacy
        # precedence — the same five fields the primary (engine) path's
        # serialize_recall_response() already attaches. Without them this
        # fallback could only ever show the raw memory_kind column, with no
        # legacy-fact_type fallback for the ~pre-4.1.19 majority of rows
        # that have none, and degraded mode would show kinds differently
        # from — not "the same way as" — normal recall.
        kind_select = ", memory_kind, memory_kind_source, memory_kind_confidence, fact_type" if has_kind_columns else ""
        # Same word-based match as the daemon's /recall fallback, so the
        # dashboard, CLI and MCP find the same rows (server/recall_fallback.py).
        from superlocalmemory.server.recall_fallback import (
            abandoned_channel_metadata, keyword_match_sql)
        sql = keyword_match_sql(body.query, column="content")
        rows = [] if sql is None else cursor.execute(f"""
            SELECT fact_id, content, confidence as memory_confidence,
                   fact_type as category, created_at{kind_select}
            FROM atomic_facts WHERE profile_id = ? AND {sql[0]}{kind_clause}
            ORDER BY {sql[2]}, confidence DESC LIMIT ?
        """, (active_profile, *sql[1], *kind_params, *sql[3], body.limit)).fetchall()
        conn.close()

        if has_kind_columns:
            from superlocalmemory.core.kind_query import engine_display_min_confidence
            from superlocalmemory.storage.memory_kinds import kind_fields

            _threshold = engine_display_min_confidence(engine)
            for row in rows:
                row.update(kind_fields(row, display_min_confidence=_threshold))

        results = [{
            **row, "score": None, "relevance_score": None, "ranking_score": None,
            "confidence": row.get("memory_confidence"), "rank_position": position,
        } for position, row in enumerate(rows, start=1)]

        return {
            "query": body.query, "results": results, "total": len(results),
            "query_type": "text_search", "retrieval_time_ms": 0,
            "retrieval_mode": "degraded_lexical", **abandoned_channel_metadata(),
            "score_contract_version": "2",
            "calibration_status": "uncalibrated", "calibration_id": None,
            "answer_confidence": None,
            "abstained": not bool(results), "answerability": "unjudged",  # never checked
            "answerability_reason": "unavailable" if results else "no_results",
            "abstention_reason": None if results else "no_candidates",
            "no_confident_match": False,
        }

    except Exception:
        raise _internal_error("Search error")
    finally:
        end_recall()


@router.get("/api/clusters")
def get_clusters(request: Request):
    """Get cluster information with member counts and statistics."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        profile = get_active_profile()
        unclustered = 0

        # V3 schema: memory_scenes stores fact_ids_json (JSON array)
        if _has_table(cursor, 'memory_scenes'):
            cursor.execute("""
                SELECT scene_id as cluster_id, theme, fact_ids_json,
                       entity_ids_json, created_at as first_memory
                FROM memory_scenes WHERE profile_id = ?
                ORDER BY created_at DESC
            """, (profile,))
            raw_scenes = cursor.fetchall()
            clusters = []
            for scene in raw_scenes:
                fact_ids = []
                try:
                    fact_ids = json.loads(scene.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                entity_ids = []
                try:
                    entity_ids = json.loads(scene.get('entity_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                clusters.append({
                    'cluster_id': scene['cluster_id'],
                    'member_count': len(fact_ids),
                    'categories': scene.get('theme', ''),
                    'summary': scene.get('theme', ''),
                    'first_memory': scene.get('first_memory', ''),
                    'top_entities': entity_ids[:5],
                })
            # Filter out empty clusters
            clusters = [c for c in clusters if c['member_count'] > 0]
            clusters.sort(key=lambda c: c['member_count'], reverse=True)

            # Count facts not in any scene
            all_scene_fact_ids = set()
            for scene in raw_scenes:
                try:
                    ids = json.loads(scene.get('fact_ids_json', '[]') or '[]')
                    all_scene_fact_ids.update(ids)
                except (json.JSONDecodeError, TypeError):
                    pass
            total_facts = cursor.execute(
                "SELECT COUNT(*) as c FROM atomic_facts WHERE profile_id = ?",
                (profile,),
            ).fetchone()['c']
            unclustered = total_facts - len(all_scene_fact_ids)
        else:
            # V2 fallback
            try:
                cursor.execute("""
                    SELECT cluster_id, COUNT(*) as member_count,
                           AVG(importance) as avg_importance,
                           GROUP_CONCAT(DISTINCT category) as categories
                    FROM memories WHERE cluster_id IS NOT NULL AND profile = ?
                    GROUP BY cluster_id ORDER BY member_count DESC
                """, (profile,))
                clusters = [dict(r, top_entities=[]) for r in cursor.fetchall()]
            except Exception:
                clusters = []
            try:
                cursor.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE cluster_id IS NULL AND profile = ?",
                    (profile,),
                )
                unclustered = cursor.fetchone()['c']
            except Exception:
                unclustered = 0

        conn.close()
        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "unclustered_count": unclustered,
        }
    except Exception:
        raise _internal_error("Cluster error")


@router.get("/api/clusters/{cluster_id}")
def get_cluster_detail(
    request: Request,
    cluster_id: str,
    limit: int = Query(BROWSE_PAGE_SIZE, ge=1, le=200),
):
    """Get detailed view of a specific cluster (scene)."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        profile = get_active_profile()

        if _has_table(cursor, 'memory_scenes'):
            # Get fact IDs from the scene's JSON array
            cursor.execute(
                "SELECT fact_ids_json, theme FROM memory_scenes "
                "WHERE scene_id = ? AND profile_id = ?",
                (cluster_id, profile),
            )
            scene_row = cursor.fetchone()
            if scene_row:
                fact_ids = []
                try:
                    fact_ids = json.loads(scene_row.get('fact_ids_json', '[]') or '[]')
                except (json.JSONDecodeError, TypeError):
                    pass
                if fact_ids:
                    ph = ','.join('?' * min(len(fact_ids), limit))
                    cursor.execute(f"""
                        SELECT fact_id as id, content, fact_type as category,
                               confidence as importance, created_at
                        FROM atomic_facts
                        WHERE profile_id = ? AND fact_id IN ({ph})
                        ORDER BY confidence DESC
                    """, [profile] + fact_ids[:limit])
                else:
                    cursor.execute("SELECT 1 WHERE 0")  # empty result
            else:
                cursor.execute("SELECT 1 WHERE 0")  # empty result
        else:
            cursor.execute("""
                SELECT id, content, summary, category, project_name, importance, created_at, tags
                FROM memories WHERE cluster_id = ? AND profile = ?
                ORDER BY importance DESC, created_at DESC LIMIT ?
            """, (cluster_id, profile, limit))
        members = cursor.fetchall()
        conn.close()
        if not members:
            raise HTTPException(status_code=404, detail="Cluster not found")
        # Generate cluster summary
        # v3.7.8: previously routed through WorkerPool.shared(), a subprocess
        # cache never recycled on a profile switch (CRITICAL cross-profile
        # leak — up to 120s stale). Use the daemon's own resident,
        # lease-protected engine's config instead; the summarizer only
        # consumes already profile-filtered `texts` above, so this closes
        # the stale-engine window without changing behavior.
        summary = ""
        try:
            texts = [m.get("content", "")[:200] for m in members[:10] if m.get("content")]
            if texts:
                from superlocalmemory.core.summarizer import Summarizer
                from superlocalmemory.server.routes.helpers import get_engine_lazy

                # v3.4.64: Do NOT call runtime.operation() synchronously in an
                # async route — if a transition is pending, acquire_operation()
                # blocks the event loop thread, deadlocking the drain.  The
                # middleware already holds an operation lease for this request;
                # the engine is stable for the full duration of the handler.
                engine = get_engine_lazy(request.app.state)
                if engine is not None:
                    summarizer = Summarizer(engine._config)
                    summary = summarizer.summarize_cluster(
                        [{"content": t} for t in texts]
                    ) or ""
        except Exception:
            pass

        return {
            "cluster_info": {"cluster_id": cluster_id, "total_members": len(members)},
            "summary": summary,
            "members": members,
            "connections": [],
        }
    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Cluster detail error")


@router.get("/api/memories/{memory_id}/facts")
def get_memory_facts(request: Request, memory_id: str):
    """Get original memory text with all its child atomic facts.

    v3.7.8: previously routed through WorkerPool.shared(), a subprocess
    engine cached at process init and never recycled on a profile switch
    (CRITICAL cross-profile leak — served the OLD profile's facts for up to
    120s after a switch). Uses the daemon's own resident, lease-protected
    engine instead, exactly like the /recall route.
    """
    try:
        from superlocalmemory.server.routes.helpers import get_engine_lazy

        # v3.4.64: Do NOT call runtime.operation() synchronously in an async
        # route — blocks the event loop when a transition is pending.  The
        # middleware already holds an operation lease; the engine and its
        # profile_id are stable for the full duration of this request.
        engine = get_engine_lazy(request.app.state)
        if engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        active_profile = engine.profile_id
        mem_map = engine._db.get_memory_content_batch(
            [memory_id], active_profile, include_global=True, include_shared=True,
        )
        original = mem_map.get(memory_id, "")
        facts = engine._db.get_facts_by_memory_id(memory_id, active_profile)
        fact_list = [
            {
                "fact_id": f.fact_id,
                "content": f.content,
                "fact_type": (
                    f.fact_type.value if hasattr(f.fact_type, "value")
                    else str(f.fact_type)
                ),
                "confidence": round(f.confidence, 3),
                "created_at": f.created_at,
            }
            for f in facts
        ]
        return {
            "ok": True,
            "memory_id": memory_id,
            "original_content": original,
            "facts": fact_list,
            "fact_count": len(fact_list),
        }
    except HTTPException:
        raise
    except Exception:
        raise _internal_error()


@router.get("/api/memories/{memory_id}/detail")
def get_memory_detail(request: Request, memory_id: str):
    """Full memory row + all child atomic facts (for dashboard modal)."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        cursor.execute(
            "SELECT memory_id, content, session_id, speaker, role, "
            "session_date, created_at, metadata_json "
            "FROM memories WHERE memory_id = ? AND profile_id = ?",
            (memory_id, active_profile),
        )
        mem = cursor.fetchone()
        if not mem:
            conn.close()
            raise HTTPException(status_code=404, detail="Memory not found")

        cursor.execute(
            "SELECT fact_id, content, fact_type, confidence, importance, "
            "access_count, created_at, entities_json "
            "FROM atomic_facts WHERE memory_id = ? AND profile_id = ? "
            "ORDER BY created_at ASC",
            (memory_id, active_profile),
        )
        facts = cursor.fetchall()
        conn.close()

        try:
            mem["metadata"] = json.loads(mem.pop("metadata_json") or "{}")
        except Exception:
            mem["metadata"] = {}
        for f in facts:
            try:
                f["entities"] = json.loads(f.pop("entities_json") or "[]")
            except Exception:
                f["entities"] = []

        return {
            "memory": mem,
            "facts": facts,
            "fact_count": len(facts),
        }
    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Detail error")


@router.get("/api/facts/{fact_id}")
def get_fact_detail(request: Request, fact_id: str):
    """Single atomic fact detail (for fact popup)."""
    try:
        conn = get_db_connection()
        conn.row_factory = dict_factory
        cursor = conn.cursor()
        active_profile = get_active_profile()

        cursor.execute(
            "SELECT f.fact_id, f.memory_id, f.content, f.fact_type, "
            "f.confidence, f.importance, f.access_count, f.created_at, "
            "f.entities_json, f.canonical_entities_json, f.session_id, "
            "m.content AS source_memory_content "
            "FROM atomic_facts f "
            "LEFT JOIN memories m ON f.memory_id = m.memory_id "
            "WHERE f.fact_id = ? AND f.profile_id = ?",
            (fact_id, active_profile),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Fact not found")
        try:
            row["entities"] = json.loads(row.pop("entities_json") or "[]")
        except Exception:
            row["entities"] = []
        try:
            row["canonical_entities"] = json.loads(
                row.pop("canonical_entities_json") or "[]"
            )
        except Exception:
            row["canonical_entities"] = []
        row["code_links"] = _code_links_for_fact(fact_id)
        return row
    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Fact detail error")


def _code_links_for_fact(fact_id: str) -> list[dict]:
    """Code entities this fact mentions, from the code graph.

    Fail-open by design: this is a display extra on a detail popup. No code graph
    built, bridge switched off, or database missing all mean "no section shown",
    never an error on the fact itself. A user who has never touched the code
    graph must not see a failure because of a feature they do not use.

    Reads code_graph.db, which is a separate database from memory.db and is never
    opened by the recall path — so nothing here can affect recall.
    """
    try:
        from superlocalmemory.code_graph.config import CodeGraphConfig

        cfg = CodeGraphConfig.load()
        if not (cfg.enabled and cfg.bridge_enabled):
            return []
        db_path = cfg.get_db_path()
        if not db_path.exists():
            return []

        import sqlite3

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.row_factory = dict_factory
            rows = conn.execute(
                "SELECT cml.link_type, cml.confidence, cml.is_stale, "
                "       cml.enriched_description, "
                "       gn.name, gn.qualified_name, gn.kind, gn.file_path "
                "FROM code_memory_links cml "
                "LEFT JOIN graph_nodes gn ON gn.node_id = cml.code_node_id "
                "WHERE cml.slm_fact_id = ? "
                "ORDER BY cml.confidence DESC, gn.qualified_name",
                (fact_id,),
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "name": r.get("name") or "",
                "qualified_name": r.get("qualified_name") or "",
                "kind": r.get("kind") or "",
                "file_path": r.get("file_path") or "",
                "link_type": r.get("link_type") or "mentions",
                "confidence": r.get("confidence"),
                "is_stale": bool(r.get("is_stale")),
                "description": r.get("enriched_description") or "",
            }
            for r in rows
            # A link whose node is gone (LEFT JOIN produced no row) is a stale
            # pointer, not something to render as a blank entry.
            if r.get("qualified_name")
        ]
    except Exception:
        logger.debug("code links unavailable for %s", fact_id, exc_info=True)
        return []


@router.delete("/api/memories/{fact_id}")
def delete_memory(request: Request, fact_id: str, profile_id: str = ""):
    """Delete a specific memory (atomic fact) by ID.

    ``profile_id`` names the profile the memory belongs to, authorized like a
    routed remember (role and policy on THAT profile, before its existence is
    revealed); without it, the active profile.
    """
    profile = _routed_profile(profile_id)
    try:
        engine, target_profile, hook_context = _authorize_memory_mutation(
            request, "delete", fact_id, run_pre_hook=False, profile=profile,
        )
    except _UnknownRoutedProfile as exc:
        return _unknown_profile_response(exc.profile_id)
    try:
        from superlocalmemory.core.mutations import delete_fact_authorized

        result = delete_fact_authorized(
            engine,
            fact_id,
            trusted_actor_id=hook_context["agent_id"],
            source_agent_id="dashboard",
            canonical_runtime=_mutation_runtime_or_missing_fact(
                request, engine, target_profile, fact_id,
            ),
            idempotency_key=_mutation_idempotency_key(request),
            profile_id=target_profile,
        )
        if not result.get("ok"):
            if result.get("retryable"):
                raise HTTPException(
                    status_code=503,
                    detail="Erasure incomplete (projection residue); retry shortly",
                )
            raise HTTPException(status_code=404, detail="Memory not found")
        _announce("memory.deleted", {"fact_id": fact_id, "profile_id": target_profile},
                  hook_context["agent_id"])
        return {
            "success": True,
            "deleted": fact_id,
            "erasure_verified": bool(result.get("erasure_verified", False)),
            "erasure_state": result.get("erasure_state", "FAILED"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        return _routed_mutation_error(exc, profile, "Delete error")


@router.post("/api/memories/{fact_id}/forget")
def forget_memory(request: Request, fact_id: str):
    """S9-DASH-08: soft-forget a fact — flip archive_status='archived'.

    Non-destructive: the row stays in ``atomic_facts`` for audit and
    can be un-archived later. Default recall paths filter it out.
    The fact's payload is ALSO copied into ``memory_archive`` so a
    future ``slm restore`` can bring it back.
    """
    engine, active_profile, hook_context = _authorize_memory_mutation(
        request, "delete", fact_id,
        run_pre_hook=False,
    )
    try:
        engine._hooks.run_pre("delete", hook_context)
        result = _canonical_mutation_runtime(request).archive_fact(
            active_profile,
            fact_id,
            idempotency_key=_mutation_idempotency_key(request),
        )
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail="Memory not found")
        engine._hooks.run_post("delete", hook_context)
        return {"success": True, "fact_id": fact_id, "archived_at": result["archived_at"]}
    except HTTPException:
        raise
    except Exception as exc:
        raise _canonical_mutation_error(exc, "Forget error")


@router.post("/api/memories/{fact_id}/merge")
def merge_memory(request: Request, fact_id: str):
    """S9-DASH-08: merge this fact into another (keep the other).

    Body: ``{into: <kept_fact_id>}``.

    Writes a ``memory_merge_log`` row (M011) for provenance and marks
    the loser's ``merged_into`` column. The loser is archived so it
    no longer appears in default recall. The winner is untouched.
    """
    engine, active_profile, hook_context = _authorize_memory_mutation(
        request, "delete", fact_id,
        run_pre_hook=False,
    )
    try:
        body = _on_loop(request.json)
        kept = str((body or {}).get("into", "")).strip()
        if not kept:
            raise HTTPException(400, "Body field 'into' is required")
        # S9-AUDIT: cap length defensively — fact_ids are UUID-v4 36 chars.
        if len(kept) > 200:
            raise HTTPException(400, "'into' exceeds 200-char limit")
        if kept == fact_id:
            raise HTTPException(400, "Cannot merge a fact into itself")
        engine._hooks.run_pre("delete", hook_context)
        result = _canonical_mutation_runtime(request).merge_fact(
            active_profile,
            fact_id,
            kept,
            idempotency_key=_mutation_idempotency_key(request),
        )
        if not result.get("ok"):
            raise HTTPException(404, "Both fact_ids must exist in the active profile")
        engine._hooks.run_post("delete", hook_context)
        return {
            "success": True,
            "merged": fact_id,
            "into": kept,
            "merged_at": result["merged_at"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _canonical_mutation_error(exc, "Merge error")


@router.patch("/api/memories/{fact_id}", status_code=202)
def edit_memory(request: Request, fact_id: str):
    """Propose an immutable, review-required correction for one memory.

    A ``profile_id`` in the body names the profile the memory belongs to,
    authorized like a routed remember; without it, the active profile.
    """
    profile = None
    try:
        body = _on_loop(request.json)
        new_content = (body.get("content") or "").strip()
        if not new_content:
            raise HTTPException(status_code=400, detail="content is required")
        profile = _routed_profile(body.get("profile_id"))
        # Prepared before anything uses it: the hook preview, the stored
        # successor, the event and the response all carry the prepared text.
        from superlocalmemory.memory_core import prepare_user_text

        new_content = prepare_user_text(
            getattr(getattr(request.app.state, "engine", None), "_config", None),
            new_content,
        ).text
        engine, target_profile, hook_context = _authorize_memory_mutation(
            request,
            "update",
            fact_id,
            content_preview=new_content,
            run_pre_hook=False,
            profile=profile,
        )
        from superlocalmemory.core.mutations import update_fact_authorized
        result = update_fact_authorized(  # embeds, may wait: never on the request loop
            engine,
            fact_id,
            new_content,
            trusted_actor_id=hook_context["agent_id"],
            source_agent_id="dashboard",
            canonical_runtime=_canonical_mutation_runtime(request),
            idempotency_key=_mutation_idempotency_key(request),
            profile_id=target_profile,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=404, detail="Memory not found")
        if result.get("unchanged"):
            return {"success": True, "fact_id": fact_id, "content": new_content, "unchanged": True}
        correction = result["correction_case"]
        _announce("memory.updated", {
            "fact_id": fact_id, "successor_fact_id": result["successor_fact_id"],
            "case_id": correction.get("case_id"), "profile_id": target_profile,
            "status": "proposed", "content_preview": new_content[:120],
        }, hook_context["agent_id"])
        return {
            "success": True,
            "fact_id": fact_id,
            "predecessor_fact_id": result["predecessor_fact_id"],
            "successor_fact_id": result["successor_fact_id"],
            "correction_case": correction,
            "review_required": True,
            "status": "proposed",
        }
    except HTTPException:
        raise
    except Exception as exc:
        return _routed_mutation_error(exc, profile, "Edit error")


@router.post("/api/corrections/{case_id}/{action}")
def review_correction(request: Request, case_id: str, action: str):
    """Apply, reject, or roll back a correction case.

    The caller authenticates through the daemon boundary.  It cannot select a
    fact scope or trust tier; the canonical writer rechecks those in its one
    SQLite transaction.  It may name a ``profile_id``, authorized exactly like
    a routed remember: its role and the correction policy on THAT profile,
    before the profile's existence is revealed.  Without one, the case is
    looked up in the active profile.
    """
    profile = None
    try:
        body = _on_loop(request.json)
        if action not in {"apply", "reject", "rollback"}:
            raise HTTPException(422, detail="action must be apply, reject, or rollback")
        expected_version = body.get("expected_version") if isinstance(body, dict) else None
        if not isinstance(expected_version, int) or isinstance(expected_version, bool):
            raise HTTPException(422, detail="expected_version must be a non-negative integer")
        if expected_version < 0:
            raise HTTPException(422, detail="expected_version must be a non-negative integer")
        event_valid_until = body.get("event_valid_until") if isinstance(body, dict) else None
        if event_valid_until is not None and not isinstance(event_valid_until, str):
            raise HTTPException(422, detail="event_valid_until must be an RFC3339 timestamp")
        if event_valid_until is not None and action != "apply":
            raise HTTPException(422, detail="event_valid_until is permitted only for apply")
        profile = _routed_profile(body.get("profile_id") if isinstance(body, dict) else None)
        engine, target_profile, hook_context = _authorize_memory_mutation(
            request, "update", case_id, run_pre_hook=False, profile=profile,
        )
        result = _canonical_mutation_runtime(request).transition_correction(
            target_profile,
            case_id,
            action=action,
            expected_version=expected_version,
            actor_id=hook_context["agent_id"],
            event_valid_until=event_valid_until,
            idempotency_key=_mutation_idempotency_key(request),
        )
        if not result.get("ok"):
            raise HTTPException(404, detail="Correction case not found")
        if action in {"apply", "rollback"}:
            from superlocalmemory.core.mutations import purge_profile_context_cache

            purge_profile_context_cache(engine, target_profile)
        engine._hooks.run_post("update", hook_context)
        _announce("memory.updated", {"case_id": case_id, "profile_id": target_profile,
                                     "status": action}, hook_context["agent_id"])
        return {"success": True, "correction_case": result}
    except HTTPException:
        raise
    except _UnknownRoutedProfile as exc:
        return _unknown_profile_response(exc.profile_id)
    except Exception as exc:
        from superlocalmemory.core.remember_runtime import (
            CaseNotInProfile,
            UnknownMutationProfile,
        )

        # Deleted between the route's check and the writer's own.
        if isinstance(exc, UnknownMutationProfile) and profile is not None:
            return _unknown_profile_response(profile)
        if isinstance(exc, CaseNotInProfile):
            raise HTTPException(404, detail="Correction case not found") from exc
        raise _canonical_mutation_error(exc, "Correction review error")


def _correction_case_response(case) -> dict[str, object]:
    """Return review metadata only; correction ledgers never contain fact text."""
    return {
        "case_id": case.case_id,
        "profile_id": case.profile_id,
        "scope": case.scope,
        "predecessor_fact_id": case.predecessor_fact_id,
        "successor_fact_id": case.successor_fact_id,
        "reason_code": case.reason_code,
        "status": case.status,
        "version": case.version,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "reviewed_at": case.reviewed_at,
        "applied_at": case.applied_at,
        "system_effective_at": case.system_effective_at,
        "event_valid_from": case.event_valid_from,
        "event_valid_until": case.event_valid_until,
    }


def _correction_store_for(engine, active_profile: str):
    from superlocalmemory.storage.correction_cases import CorrectionCaseStore

    return CorrectionCaseStore(
        engine._db.db_path,
        is_profile_active=lambda candidate: candidate == active_profile,
        # Read operations never invoke this callback; writes use the daemon's
        # canonical runtime, which derives the authenticated actor separately.
        is_actor_trusted=lambda _actor: False,
    )


@router.get("/api/corrections")
def list_corrections(request: Request, limit: int = 100, profile_id: str = ""):
    """List bounded review metadata for one profile: the routed one when
    ``profile_id`` names it (authorized like a routed review), else the active one."""
    try:
        profile = _routed_profile(profile_id)
        engine, target_profile, _context = _authorize_memory_mutation(
            request, "update", "correction-list", run_pre_hook=False, profile=profile,
        )
        cases = _correction_store_for(engine, target_profile).list_cases(
            target_profile, limit=limit)
        from superlocalmemory.server.routes.overtaken import overtaken_for  # a user action closed
        return {"success": True, "corrections": [_correction_case_response(case) for case in cases],
                "overtaken": overtaken_for(engine, target_profile, limit)}
    except HTTPException:
        raise
    except _UnknownRoutedProfile as exc:
        return _unknown_profile_response(exc.profile_id)
    except Exception as exc:
        raise _canonical_mutation_error(exc, "Correction list error")


@router.get("/api/corrections/{case_id}")
def get_correction(request: Request, case_id: str, profile_id: str = ""):
    """Get one correction case without exposing raw memory text, from the
    routed profile when ``profile_id`` names it, else from the active one."""
    try:
        profile = _routed_profile(profile_id)
        engine, target_profile, _context = _authorize_memory_mutation(
            request, "update", case_id, run_pre_hook=False, profile=profile,
        )
        case = _correction_store_for(engine, target_profile).get_case(case_id)
        return {"success": True, "correction": _correction_case_response(case)}
    except HTTPException:
        raise
    except _UnknownRoutedProfile as exc:
        return _unknown_profile_response(exc.profile_id)
    except Exception as exc:
        from superlocalmemory.storage.correction_cases import (
            CorrectionAuthorizationError,
            CorrectionNotFoundError,
        )

        # Another profile's case is "not found" here, exactly like a missing
        # one, so a lookup cannot tell which ids exist elsewhere.
        if isinstance(exc, (CorrectionNotFoundError, CorrectionAuthorizationError)):
            raise HTTPException(404, detail="Correction case not found") from exc
        raise _canonical_mutation_error(exc, "Correction lookup error")


_VALID_SCOPES = ("personal", "shared", "global")


@router.patch("/api/memories/{fact_id}/scope")
def set_memory_scope(request: Request, fact_id: str):
    """Set a memory's scope (personal | shared | global) + shared_with.

    Body: {"scope": "shared", "shared_with": "alice,bob"}. shared_with accepts a
    comma-separated string or a list; it is stored as a JSON array so the
    _scope_where LIKE match works. Mutation-authorized, and the fact must
    belong to the active profile (a caller cannot re-scope another profile's
    fact). This is the write side of multi-scope sharing from the dashboard.
    """
    try:
        body = _on_loop(request.json)
        scope = (body.get("scope") or "").strip().lower()
        if scope not in _VALID_SCOPES:
            raise HTTPException(400, detail=f"scope must be one of {_VALID_SCOPES}")

        raw_shared = body.get("shared_with", [])
        if isinstance(raw_shared, str):
            shared_list = [s.strip() for s in raw_shared.split(",") if s.strip()]
        elif isinstance(raw_shared, list):
            shared_list = [str(s).strip() for s in raw_shared if str(s).strip()]
        else:
            shared_list = []
        if scope == "shared" and not shared_list:
            raise HTTPException(
                400, detail="shared scope requires at least one profile in shared_with")
        # global/personal never carry a shared_with list.
        if scope != "shared":
            shared_list = []

        engine, active_profile, hook_context = _authorize_memory_mutation(
            request, "update", fact_id, run_pre_hook=False,
        )
        if scope in {"shared", "global"}:
            from superlocalmemory.access.rbac import Permission
            from superlocalmemory.server.rbac_enforce import require_permission

            require_permission(request, Permission.SHARE, profile=active_profile)
        engine._hooks.run_pre("update", hook_context)
        result = _canonical_mutation_runtime(request).set_fact_scope(
            active_profile,
            fact_id,
            scope,
            shared_list,
            idempotency_key=_mutation_idempotency_key(request),
        )
        if not result.get("ok"):
            raise HTTPException(404, detail="Memory not found in this profile")
        engine._hooks.run_post("update", hook_context)
        return {
            "success": True, "fact_id": fact_id, "scope": scope,
            "shared_with": shared_list, "active_profile": active_profile,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _canonical_mutation_error(exc, "Scope update error")
