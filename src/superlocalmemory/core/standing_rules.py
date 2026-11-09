# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The standing rules and decisions a session starts with.

Only kinds a person or the saving agent confirmed are read (LLD invariant I6):
a model's suggestion never decides what every session is told. A fact that was
archived, quarantined or marked replaced is left out - a replaced decision is
exactly what must not be loaded. A rule matters whether or not it was read
recently, so warm and cold facts count; only archived ones do not.

Reads only. Returns [] on a store without the kind columns, or on any error.

4.2.14: one entry per memory. Since 4.2.5 every sentence of a memory is also a
fact of its own and inherits the memory's kind, so one seven-sentence rule
filled seven of the ten slots with fragments. The memory's whole-text fact
stands for it (else its longest fact). Over the cap, rules of this project come
first, then the ones the session's own recall found relevant, then the newest -
the cap is unchanged, it is only filled by relevance instead of by age.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from superlocalmemory.storage.memory_kinds import CONFIRMED_SOURCES, MemoryKind

logger = logging.getLogger(__name__)

#: Caps per session: enough to state a working agreement, small enough that
#: the injected block stays readable.
MAX_RULES = 10
MAX_DECISIONS = 5


@dataclass(frozen=True, slots=True)
class StandingFact:
    fact_id: str
    content: str
    kind: str
    importance: float
    access_count: int
    memory_id: str = ""


def _has_table(db: Any, name: str) -> bool:
    rows = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return bool(rows)


def confirmed_facts(
    db: Any, profile_id: str, kind: MemoryKind, limit: int, project_path: str = "",
    relevance: Mapping[str, float] | None = None,
) -> list[StandingFact]:
    """Current confirmed facts, one per memory, best first, at most ``limit``.

    ``relevance`` maps a memory id to the score the session's recall gave it."""
    try:
        if limit <= 0 or not db.has_memory_kind_columns():
            return []
        # Every candidate is read, as 4.2.13 did for a project: a read window
        # could fill with other projects' newer rules before this one's.
        rows = [dict(r) for r in _confirmed_rows(db, profile_id, kind)]
        rows = _project_rows(rows, project_path)
        candidate_ids = [str(row["fact_id"]) for row in rows]
        withheld = (
            db.get_nonapplied_correction_successor_ids(candidate_ids, profile_id)
            if candidate_ids
            else set()
        )
        if not isinstance(withheld, set):
            logger.debug("Standing %s lookup skipped: malformed admission result", kind.value)
            return []
    except Exception as exc:  # noqa: BLE001 - session start must never fail on this
        logger.debug("Standing %s lookup skipped: %s", kind.value, type(exc).__name__)
        return []
    chosen = _one_per_memory([r for r in rows if str(r["fact_id"]) not in withheld])
    ranked = _ranked(chosen, project_path, relevance or {})
    return [_standing(row, kind) for row in ranked[:limit]]


def standing_facts(
    db: Any, profile_id: str, exclude: frozenset[str] = frozenset(), *, project_path: str = "",
    relevance: Mapping[str, float] | None = None,
) -> list[StandingFact]:
    """Standing rules first, then active decisions; ``exclude`` (already shown,
    e.g. pinned) does not use up a slot."""
    out: list[StandingFact] = []
    for kind, cap in ((MemoryKind.RULE, MAX_RULES), (MemoryKind.DECISION, MAX_DECISIONS)):
        found = confirmed_facts(db, profile_id, kind, cap + len(exclude), project_path,
                                relevance)
        out.extend([f for f in found if f.fact_id not in exclude][:cap])
    return out


def _one_per_memory(rows: list[dict]) -> list[dict]:
    """The fact that stands for each memory: its whole-text fact, else its
    longest. A fact with no memory stands for itself. Keeps newest-first order."""
    best: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        key = str(row.get("memory_id") or row["fact_id"])
        if key not in best:
            order.append(key)
            best[key] = row
        elif _stands_better(row, best[key]):
            best[key] = row
    return [best[key] for key in order]


def _stands_better(row: dict, current: dict) -> bool:
    whole = str(row.get("memory_content") or "")
    row_whole = bool(whole) and row["content"] == whole
    current_whole = bool(whole) and current["content"] == whole
    if row_whole != current_whole:
        return row_whole
    return len(str(row["content"] or "")) > len(str(current["content"] or ""))


def _ranked(rows: list[dict], project_path: str, relevance: Mapping[str, float]) -> list[dict]:
    """This project's own first, then by the session recall's score; ties keep
    the newest-first order the rows arrived in (``sorted`` is stable)."""
    from superlocalmemory.core.project_identity import project_key

    wanted = project_key(project_path) if project_path else None

    def key(row: dict) -> tuple[int, float]:
        own = wanted is not None and _row_project(row) == wanted
        return (0 if own else 1, -float(relevance.get(str(row.get("memory_id") or ""), 0.0)))

    return sorted(rows, key=key)


def _row_project(row: dict) -> Any:
    from superlocalmemory.core.project_identity import project_key

    try:
        meta = json.loads(row.get("metadata_json") or "{}")
        return project_key(meta.get("project") or meta.get("project_path"))
    except (ValueError, TypeError, AttributeError):
        return None


def _standing(row: dict, kind: MemoryKind) -> StandingFact:
    return StandingFact(
        str(row["fact_id"]),
        str(row["content"] or ""),
        kind.value,
        float(row.get("importance") or 0.0),
        int(row.get("access_count") or 0),
        str(row.get("memory_id") or ""),
    )


def enabled(config: Any) -> bool:
    """``memory_kinds.standing_rules_in_session`` (default on)."""
    section = getattr(config, "memory_kinds", None)
    value = getattr(section, "standing_rules_in_session", True)
    return value is not False


__all__ = [
    "MAX_DECISIONS",
    "MAX_RULES",
    "StandingFact",
    "confirmed_facts",
    "enabled",
    "standing_facts",
]


def _project_rows(rows: list, project_path: str) -> list:
    from superlocalmemory.core.project_identity import project_key

    if not project_path:
        return rows
    wanted = project_key(project_path)
    admitted = []
    for row in rows:
        try:
            meta = json.loads(dict(row).get("metadata_json") or "{}")
            project = project_key(meta.get("project") or meta.get("project_path"))
            if project is None or project == wanted:
                admitted.append(row)
        except (ValueError, TypeError, AttributeError):
            continue
    return admitted


def _confirmed_rows(db, profile_id, kind):
    sources = sorted(s.value for s in CONFIRMED_SOURCES)
    replaced = ""
    if _has_table(db, "fact_temporal_validity"):
        replaced = (
            " AND NOT EXISTS (SELECT 1 FROM fact_temporal_validity v "
            "WHERE v.fact_id = f.fact_id AND v.system_expired_at IS NOT NULL)"
        )
    return db.execute(
        "SELECT f.fact_id, f.memory_id, f.content, f.importance, f.access_count, "
        "m.metadata_json, m.content AS memory_content "
        "FROM atomic_facts f LEFT JOIN memories m ON m.memory_id=f.memory_id "
        "AND m.profile_id=f.profile_id "
        "WHERE f.profile_id = ? AND f.memory_kind = ? "
        f"AND f.memory_kind_source IN ({','.join('?' * len(sources))}) "
        "AND COALESCE(f.lifecycle, 'active') != 'archived' "
        "AND COALESCE(f.quarantined, 0) = 0"
        + replaced
        + " ORDER BY f.created_at DESC, f.rowid DESC",
        (profile_id, kind.value, *sources),
    )
