# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""MCP surface for the readable summary layer (issue #113).

WHY THIS EXISTS
---------------
4.0.6 shipped the summary generators with no caller at all. 4.0.7 added
``slm summary``, which fixed it for a person at a terminal and left agents with
nothing — the changelog said the defect was "no command, tool or endpoint" and
only the command was built. This is the tool half.

It matters more than the CLI: the natural consumer of "what did I work on
yesterday" is the agent holding the conversation, not a human running a command.

CONTRACT
--------
Read-only, profile-scoped, and honest about coverage. Every response carries
``coverage`` and ``source_fact_ids``, so a caller can tell a summary of 4% of a
session from a summary of all of it, and can drill back to the memories it came
from. Callers must not present a partial summary as complete; the field exists
precisely so they do not have to guess.

NOT ON THE HOT PATH. Summaries read memory.db directly and are invoked on
demand; nothing here runs during remember or recall.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from typing import Any, Callable

from superlocalmemory.core.admission import admits
from superlocalmemory.core.operation_request import OperationKind
from superlocalmemory.infra.data_root import state_path
from superlocalmemory.mcp.tool_hints import READ_ONLY

logger = logging.getLogger("superlocalmemory.mcp.summaries")

#: Accepted values for the ``kind`` argument.
_KINDS = ("day", "project", "session", "community", "page")

#: Q9 (2026-10-06): a community can hold thousands of member ids. Explicit
#: drill-down (this tool) may return them all, but a caller that passes no
#: ``limit`` still gets a bounded page rather than an accidental multi-MB
#: response — raise it with an explicit ``limit`` when more is wanted.
_DEFAULT_COMMUNITY_MEMBER_LIMIT = 500


def _result_payload(result: Any) -> dict[str, Any]:
    """Shape a SummaryResult for the wire.

    ``coverage`` and ``source_fact_ids`` are non-negotiable parts of the
    response: a summary that cannot be traced back, or that hides how much it
    covered, is the opaque generic summary issue #113 asked us not to build.
    """
    return {
        "success": True,
        "kind": result.kind,
        "profile_id": result.profile_id,
        "summary": result.content,
        "coverage": result.coverage,
        "generated_by": result.generated_by,
        "source_fact_ids": result.source_fact_ids,
        "source_count": len(result.source_fact_ids),
        "metadata": result.metadata,
    }


def _error(message: str, **extra: Any) -> dict[str, Any]:
    out = {"success": False, "error": message}
    out.update(extra)
    return out


def _community_drill_down(
    engine: Any, profile_id: str, community_id: int, limit: int, offset: int,
    hidden_of: Any = None,
) -> Any | None:
    """Full (paged) membership of one community — the Q9 explicit drill-down.

    ``thematic_context`` on recall/session_init carries only a small,
    relevance-bounded SAMPLE of a community's member ids (see
    ``RetrievalEngine._community_context``); this reads the complete stored
    list those ids come from. Not on the hot path — a direct, on-demand DB
    read, exactly like the other summary kinds in this module.
    """
    from superlocalmemory.core.community_summary import CommunitySummaryBuilder
    from superlocalmemory.summaries.base import (
        COVERAGE_FULL,
        GENERATED_BY_EXTRACTIVE,
        SummaryResult,
    )

    db = getattr(engine, "db", None) or getattr(engine, "_db", None)
    if db is None:
        return None
    row = CommunitySummaryBuilder(db).get_summary(profile_id, community_id)
    if row is None:
        return None
    try:
        all_ids = json.loads(row.get("fact_ids_json") or "[]")
    except (ValueError, TypeError):
        all_ids = []
    if hidden_of is not None and hidden_of(all_ids):
        # The summary text was written from every member: a remote caller that may not
        # see one of them is told there is no such community.
        return None
    total = len(all_ids)
    page_limit = limit if isinstance(limit, int) and limit > 0 else _DEFAULT_COMMUNITY_MEMBER_LIMIT
    page_offset = max(0, offset) if isinstance(offset, int) else 0
    page = all_ids[page_offset:page_offset + page_limit]
    return SummaryResult(
        kind="community",
        profile_id=profile_id,
        content=row.get("summary", ""),
        source_fact_ids=page,
        coverage=COVERAGE_FULL,
        generated_by=GENERATED_BY_EXTRACTIVE,
        metadata={
            "community_id": community_id,
            "keywords": row.get("keywords", ""),
            "member_count": int(row.get("fact_count") or total),
            "offset": page_offset,
            "limit": page_limit,
            "has_more": page_offset + page_limit < total,
        },
    )


def _compiled_page(engine: Any, profile_id: str, target: str, hidden_of: Any = None) -> Any | None:
    """A compiled page (WS-6, D-18): every sentence a word-for-word span of a memory it cites
    or a fixed template over slot values, re-checked now. With no ``target``, the list of pages.

    ``None`` when there is no such page (or a remote caller may not see a memory it cites).
    """
    from superlocalmemory.compiler import pages
    from superlocalmemory.summaries.base import (
        COVERAGE_FULL,
        COVERAGE_PARTIAL,
        GENERATED_BY_EXTRACTIVE,
        SummaryResult,
    )

    db = getattr(engine, "db", None) or getattr(engine, "_db", None)
    if db is None:
        return None
    if not (target or "").strip():
        listed = pages.list_pages(db, profile_id)
        if hidden_of is not None:  # a remote caller is not shown titles of pages it cannot read
            listed = []
        lines = [f"{p['title']} ({p['kind'].replace('_', ' ')}, id {p['page_id']})" for p in listed]
        return SummaryResult(
            kind="page", profile_id=profile_id,
            content="\n".join(lines) or "No compiled pages yet.", source_fact_ids=[],
            coverage=COVERAGE_FULL, generated_by=GENERATED_BY_EXTRACTIVE,
            metadata={"pages": listed, "next": "call again with target=<page id> to read one"})
    page = pages.show_page(db, profile_id, target.strip())
    if page is None:
        return None
    cited = sorted({c["fact_id"] for s in page["sentences"] for c in s["citations"]})
    if hidden_of is not None and hidden_of(cited):
        return None
    text = "\n".join(s["text"] for s in page["sentences"])
    if page["withheld"]["count"]:
        text += f"\n({page['withheld']['count']} sentences withheld.)"
    return SummaryResult(
        kind="page", profile_id=profile_id, content=f"{page['title']}\n{text}",
        source_fact_ids=cited,
        coverage=COVERAGE_PARTIAL if page["withheld"]["count"] else COVERAGE_FULL,
        generated_by=GENERATED_BY_EXTRACTIVE,
        metadata={"page_id": page["page_id"], "page_kind": page["kind"], "origin": page["origin"],
                  "stale": page["stale"], "withheld": page["withheld"], "notice": page["notice"],
                  "sentences": [{"text": s["text"], "facts": [c["fact_id"] for c in s["citations"]],
                                 "type": s["type"]} for s in page["sentences"]]})


def register_summary_tools(server: Any, get_engine: Callable[[], Any]) -> None:
    """Register the read-only summary tool."""

    @server.tool(title="Summarize Memories", annotations=READ_ONLY)
    @admits(OperationKind.RECALL)
    async def get_memory_summary(
        kind: str = "day",
        target: str = "",
        profile_id: str = "",
        limit: int = 0,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Summarise your memories: a day, a project, a session, or a community.

        Args:
            kind: "day", "project", "session", "community", or "page".
            target: For "day", an ISO date, "today" or "yesterday" (default
                today, in this computer's time zone). For "project", a directory
                path (default: none — supply one). For "session", the session
                id; leave it empty to get ``recent_sessions`` to choose from.
                For "community", the ``community_id`` a recall or session_init
                response gave you in ``thematic_context``. For "page", a page
                id (leave it empty to list the compiled pages).
            profile_id: The profile to summarise (empty = the active one).
            limit, offset: "community" only — page through
                ``source_fact_ids`` when it is large (default: first 500).
                Ignored for every other kind.

        Returns a summary plus ``coverage`` and ``source_fact_ids``. Coverage is
        not decoration: session data is sparse — roughly 4% of facts carry a
        session id — so a session summary is usually partial. Do not present a
        partial summary as a complete record of what happened.

        "page" (4.2.30) reads a compiled page: a current-state, story or topic page
        built only from word-for-word sentences of the memories it cites and fixed
        templates over slot values, each sentence re-checked when you ask. It says
        how many sentences it withheld and is never a memory itself; read it for
        an overview, and use ``recall`` for a question.

        "community" is the drill-down for ``thematic_context``: a recall or
        session_init response carries only a small relevance-bounded SAMPLE of
        a community's member ids (to keep every response small); call this,
        with that response's ``community_id``, to page through the rest.

        No language model is required; summaries are extractive unless the
        profile runs a local or cloud model, in which case that writes them.
        """
        kind = (kind or "day").strip().lower()
        if kind not in _KINDS:
            return _error(
                f"unknown summary kind {kind!r}; expected one of {', '.join(_KINDS)}"
            )

        from superlocalmemory.mcp.request_profile import requested_profile, tool_profile

        engine = get_engine()
        try:
            named = requested_profile(profile_id)
        except ValueError as exc:
            return _error(str(exc))
        if named:
            profile_id, refused = tool_profile(engine, named)
            if refused:
                return refused
        else:
            profile_id = getattr(engine, "profile_id", "default")

        # A remote caller's summaries leave out what it may not see (pictures, pages,
        # folder files). None for a caller on this computer: nothing changes for it.
        from superlocalmemory.mcp.remote_visibility import current_view, hidden_fact_ids

        hidden_of = None
        if current_view():
            def hidden_of(ids, _db=engine._db, _pid=profile_id):  # noqa: F811
                return hidden_fact_ids(_db, _pid, list(ids))

        if kind == "page":  # compiled pages (WS-6): read through the engine's db, like community
            try:
                result = _compiled_page(engine, profile_id, target, hidden_of)
            except Exception as exc:
                logger.warning("page read failed (%s): %s", target, exc)
                return _error(f"page read failed: {exc}", kind=kind)
            if result is None:
                return _error(f"no page {(target or '').strip()!r} for this profile", kind=kind)
            return _result_payload(result)

        # "community" reads through the live engine's own db handle (exactly
        # like session_init's community_context), not a path-opened
        # connection — the file-existence check below is for the other three
        # kinds, which open memory.db themselves.
        if kind == "community":
            if not (target or "").strip().lstrip("-").isdigit():
                return _error("kind='community' requires target=<community_id>")
            try:
                result = _community_drill_down(
                    engine, profile_id, int(target.strip()), limit, offset, hidden_of,
                )
            except Exception as exc:
                logger.warning("community drill-down failed (%s): %s", target, exc)
                return _error(f"community drill-down failed: {exc}", kind=kind)
            if result is None:
                return _error(
                    f"no community {target.strip()!r} for this profile",
                    community_id=int(target.strip()),
                )
            return _result_payload(result)

        db_path = state_path("memory.db")
        if not db_path.exists():
            return _error("no memory database found", db_path=str(db_path))

        # The engine's config drives Mode B/C enrichment. Passing None would
        # silently force the extractive path for every caller regardless of
        # mode — the exact bug the CLI shipped with in 4.0.7.
        config = getattr(engine, "config", None)

        try:
            if kind == "day":
                from superlocalmemory.summaries import generate_daily_reflection
                from superlocalmemory.summaries.base import local_offset_minutes

                day = (target or "").strip() or date.today().isoformat()
                if day == "today":
                    day = date.today().isoformat()
                elif day == "yesterday":
                    day = (date.today() - timedelta(days=1)).isoformat()
                try:
                    date.fromisoformat(day)
                except ValueError:
                    return _error(f"target {day!r} is not a date; use YYYY-MM-DD, "
                                  "'today' or 'yesterday'")
                # "today" is this computer's today, so bucket by its time zone.
                result = generate_daily_reflection(
                    db_path, day, profile_id, config,
                    tz_offset_minutes=local_offset_minutes(day), hidden_of=hidden_of)

            elif kind == "project":
                from superlocalmemory.summaries import generate_project_work_log

                if not (target or "").strip():
                    return _error("kind='project' requires target=<project path>")
                result = generate_project_work_log(
                    db_path, target.strip(), profile_id, config,
                    hidden_of=hidden_of,
                )

            elif kind == "session":
                from superlocalmemory.summaries import generate_session_summary

                if not (target or "").strip():
                    # Say which ids exist: an agent has no other way to learn
                    # them, and a bare refusal is a dead end.
                    from superlocalmemory.summaries.sessions import list_recent_sessions

                    # Its counts include every memory of a session, so a remote caller
                    # is not shown the list: it names the session it wants.
                    return _error("kind='session' requires target=<session id>",
                                  recent_sessions=[] if hidden_of is not None
                                  else list_recent_sessions(db_path, profile_id))
                result = generate_session_summary(
                    db_path, target.strip(), profile_id, config, hidden_of=hidden_of,
                )
        except Exception as exc:
            logger.warning("summary generation failed (%s/%s): %s", kind, target, exc)
            return _error(f"summary generation failed: {exc}", kind=kind)

        return _result_payload(result)
