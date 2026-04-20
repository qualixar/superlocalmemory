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
from typing import Callable

from superlocalmemory.hooks.antigravity_adapter import AntigravityAdapter
from superlocalmemory.hooks.copilot_adapter import CopilotAdapter
from superlocalmemory.hooks.cursor_adapter import CursorAdapter
from superlocalmemory.hooks.context_payload import (
    RecallFn,
    build_payload,
    format_decisions,
    format_entities,
    format_memories,
    format_topics,
)
from superlocalmemory.hooks.cross_platform_connector import (
    CrossPlatformConnector,
)
from superlocalmemory.mcp.tools_context import prestage_context


def _noop_recall(_q: str, _limit: int, _profile: str) -> list[dict]:
    """Placeholder recall when the daemon/engine isn't reachable."""
    return []


def _get_recall_fn() -> RecallFn:
    """Wire the real recall engine if available, else no-op."""
    try:
        from superlocalmemory.core.engine import Engine  # pragma: no cover
        eng = Engine.get_shared()
        def _fn(q: str, limit: int, profile_id: str) -> list[dict]:
            try:
                return eng.recall(q, limit=limit) or []  # type: ignore
            except Exception:
                return []
        return _fn
    except Exception:
        return _noop_recall


def _default_sync_log_db() -> Path:
    return Path(os.environ.get("SLM_MEMORY_DB",
                               str(Path.home() / ".superlocalmemory"
                                   / "memory.db")))


def build_default_adapters(
    *, base_dir: Path | None = None,
    recall_fn: RecallFn | None = None,
    sync_log_db: Path | None = None,
) -> list:
    base = Path(base_dir or Path.cwd())
    recall = recall_fn or _get_recall_fn()
    db = sync_log_db or _default_sync_log_db()
    adapters: list = []
    adapters.append(CursorAdapter(scope="project", base_dir=base,
                                  sync_log_db=db, recall_fn=recall))
    adapters.append(CursorAdapter(scope="global", base_dir=Path.home(),
                                  sync_log_db=db, recall_fn=recall))
    adapters.append(AntigravityAdapter(scope="workspace", base_dir=base,
                                       sync_log_db=db, recall_fn=recall))
    adapters.append(AntigravityAdapter(scope="global", base_dir=Path.home(),
                                       sync_log_db=db, recall_fn=recall))
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
    payload = build_payload(
        profile_id, "project", Path.cwd(),
        recall_fn=_get_recall_fn(),
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
