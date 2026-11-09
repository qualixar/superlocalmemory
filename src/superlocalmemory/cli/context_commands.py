# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §8

"""CLI — ``slm context prestage`` and ``slm connect --disable`` (LLD-05).

Kept in a focused module (instead of bloating cli/commands.py past its cap).
"""

from __future__ import annotations

import json as _json
import os
from argparse import Namespace
from pathlib import Path

from superlocalmemory.hooks.antigravity_adapter import AntigravityAdapter
from superlocalmemory.hooks.context_payload import (
    RecallFn,
    build_payload,
    format_decisions,
    format_entities,
    format_memories,
    format_topics,
)
from superlocalmemory.hooks.copilot_adapter import CopilotAdapter
from superlocalmemory.hooks.cross_platform_connector import (
    CrossPlatformConnector,
)
from superlocalmemory.hooks.cursor_adapter import CursorAdapter
from superlocalmemory.mcp.tools_context import prestage_context


def _daemon_recall(q: str, limit: int, _profile: str, *, project: str = "") -> list[dict]:
    """Recall through a RUNNING daemon, as ``{"text", "score"}`` rows.

    Never starts one: building a context file is not a reason to launch the
    memory service. Not a question either, so it is never judged. ``project``
    is recall's project filter (#150). The active profile serves it, as before.
    """
    from superlocalmemory.cli.daemon import _get_port, is_daemon_running
    from superlocalmemory.core.answer_check_scope import context_load
    from superlocalmemory.mcp._daemon_proxy import DaemonPoolProxy

    if not is_daemon_running():
        return []
    with context_load():
        raw = DaemonPoolProxy(port=_get_port()).recall(
            q, limit=limit, project=project or "", answer_check=False)
    if not isinstance(raw, dict) or raw.get("ok") is False:
        return []
    return [{"text": r.get("content", ""), "score": float(r.get("score") or 0.0)}
            for r in raw.get("results", []) if isinstance(r, dict)]


def _get_recall_fn() -> RecallFn:
    """Recall through the running daemon; nothing when none is running.

    This used to call ``Engine.get_shared()``, which does not exist, so the
    import failed silently and every context file was built from no memories.
    """
    def _fn(q: str, limit: int, profile_id: str, *, project: str = "") -> list[dict]:
        try:
            return _daemon_recall(q, limit, profile_id, project=project)
        except Exception:  # noqa: BLE001 - a context file is advisory
            return []
    return _fn


def _default_sync_log_db() -> Path:
    explicit = os.environ.get("SLM_MEMORY_DB", "").strip()
    if explicit:
        return Path(explicit)
    from superlocalmemory.infra.data_root import state_path
    return state_path("memory.db")


def build_default_adapters(
    *, base_dir: Path | None = None,
    recall_fn: RecallFn | None = None,
    sync_log_db: Path | None = None,
    include_project_scope: bool = True,
) -> list:
    """Adapters that write SLM context into each host's rules files.

    The working directory is only a project when someone is standing in one.
    The daemon passes ``include_project_scope=False``: it serves every project
    at once, and taking the project from its working directory sent the
    project-scoped adapters to whatever directory it happened to start in.
    Under launchd that is "/", so they tried to write /.cursor and /.agent —
    2,904 read-only failures in one daemon's log. The root of the disk is never
    a project, whoever calls this.
    """
    base = Path(base_dir or Path.cwd())
    resolved = base.resolve()
    if resolved == Path(resolved.anchor):
        include_project_scope = False
    recall = recall_fn or _get_recall_fn()
    db = sync_log_db or _default_sync_log_db()
    adapters: list = []
    if include_project_scope:
        adapters.append(CursorAdapter(scope="project", base_dir=base,
                                      sync_log_db=db, recall_fn=recall))
    adapters.append(CursorAdapter(scope="global", base_dir=Path.home(),
                                  sync_log_db=db, recall_fn=recall))
    if include_project_scope:
        adapters.append(AntigravityAdapter(scope="workspace", base_dir=base,
                                           sync_log_db=db, recall_fn=recall))
    adapters.append(AntigravityAdapter(scope="global", base_dir=Path.home(),
                                       sync_log_db=db, recall_fn=recall))
    if include_project_scope:
        adapters.append(CopilotAdapter(base_dir=base, sync_log_db=db,
                                       recall_fn=recall))
    return adapters


# ---------------------------------------------------------------------------
# `slm connect` (LLD-05 mode)
# ---------------------------------------------------------------------------


def cmd_connect_cross_platform(args: Namespace) -> None:
    """LLD-05 mode: orchestrate cross-platform adapters."""
    adapters = build_default_adapters()
    connector = CrossPlatformConnector(adapters)

    disable_target = getattr(args, "disable", None)
    if disable_target:
        ok = connector.disable(disable_target)
        msg = {"adapter": disable_target, "disabled": ok}
        if getattr(args, "json", False):
            print(_json.dumps(msg))
        else:
            print(f"{'Disabled' if ok else 'Not found'}: {disable_target}")
        return

    if getattr(args, "dry_run", False):
        statuses = connector.detect()
        out = [
            {"name": s.name, "active": s.active, "target": s.target_path}
            for s in statuses
        ]
        if getattr(args, "json", False):
            print(_json.dumps({"detected": out}))
        else:
            for row in out:
                mark = "[+]" if row["active"] else "[-]"
                print(f"  {mark} {row['name']:25s} {row['target']}")
        return

    results = connector.connect()
    if getattr(args, "json", False):
        print(_json.dumps({"results": results}))
    else:
        for name, res in results.items():
            print(f"  {name}: {res}")


# ---------------------------------------------------------------------------
# `slm context prestage`
# ---------------------------------------------------------------------------


def _render_markdown(payload) -> str:
    lines = [
        "# --- SLM Context ---",
        f"_Version {payload.version} — {payload.generated_at}_",
        "",
        "## Topics",
        format_topics(payload),
        "",
        "## Entities",
        format_entities(payload),
        "",
        "## Recent decisions",
        format_decisions(payload),
        "",
        "## Project memories",
        format_memories(payload),
    ]
    return "\n".join(lines)


def cmd_context(args: Namespace) -> None:
    """Dispatcher for ``slm context <subcommand>``."""
    sub = getattr(args, "subcommand", None) or "prestage"
    if sub != "prestage":
        print(f"Unknown subcommand: {sub}")
        return

    query = getattr(args, "query", "") or ""
    limit = int(getattr(args, "limit", 5))
    profile_id = getattr(args, "profile_id", "default")

    if getattr(args, "tool", False):
        # Direct MCP-like tool response.
        result = prestage_context(
            query, limit=limit, profile_id=profile_id,
            recall_fn=_get_recall_fn(),
        )
        print(_json.dumps(result, indent=2))
        return

    # Build a full markdown / JSON payload for CLI consumption.
    # #150: the working directory is the project, so each section asks for
    # that project's memories rather than canned global words.
    payload = build_payload(
        profile_id, "project", Path.cwd(),
        recall_fn=_get_recall_fn(),
        project=str(Path.cwd()),
    )
    if getattr(args, "json", False):
        print(_json.dumps({
            "query": query,
            "topics": list(payload.topics),
            "entities": list(payload.entities),
            "decisions": list(payload.recent_decisions),
            "memories": list(payload.project_memories),
            "generated_at": payload.generated_at,
        }))
        return
    print(_render_markdown(payload))


__all__ = (
    "build_default_adapters",
    "cmd_context",
    "cmd_connect_cross_platform",
)
