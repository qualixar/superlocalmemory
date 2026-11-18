# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Region provenance (WS-8, flag ``media.regions``): which page and which box a fact's text came from.

When a picture or a scanned page is read, the reader reports the box of each line it recognised. They are
kept (``media_regions`` in ``media.db``, fractions of the width and height, so a thumbnail and the original
agree) next to a redacted snippet of the words. Later, ``slm why`` and the memory drawer ask: which of the
regions of the picture this memory came from does this fact quote? and show it in words ("from page 3, top
left"). A fact made from a caption or from a person's own note matches no region and shows none.

Reads are read-only and fail open: a missing file, a missing table or an old build gives an empty list.
Words here are for a person; no raw ids are shown.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: Regions kept for one picture or page (routing-only: a page of dense text has a few hundred lines).
MAX_REGIONS = 400
#: Words of a region kept to match a fact against (routing-only: enough to find the line again).
SNIPPET_CHARS = 160
#: Most places shown for one fact (routing-only: a page size).
MAX_SHOWN = 3
#: A fact "quotes" a region when at least this share of the region's words are in the fact (routing-only).
MATCH_SHARE = 0.6
#: Words shorter than this are not used to match (routing-only).
MIN_WORD = 3
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_WHOLE = 0.85  # a box this big in both directions is "the whole page"


def _unit(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, v)) if math.isfinite(v) else None


def clean_regions(raw: Iterable[Any], redact: Any = None) -> list[dict[str, Any]]:
    """Valid regions only: a box of four fractions with positive area, a snippet, a label.

    ``redact`` (text -> text) is the same redaction the memory's own text went through, so a snippet
    never holds more than the memory does. Anything malformed is dropped; nothing raises.
    """
    out: list[dict[str, Any]] = []
    for item in list(raw or [])[:MAX_REGIONS]:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (_unit(v) for v in box)
        if None in (x0, y0, x1, y1) or x1 <= x0 or y1 <= y0:  # type: ignore[operator]
            continue
        text = str(item.get("text") or item.get("snippet") or "")  # cleaning twice is harmless
        try:
            text = redact(text) if redact is not None else text
        except Exception:  # noqa: BLE001 - a snippet that cannot be vetted is not kept
            continue
        text = re.sub(r"\s+", " ", text).strip()[:SNIPPET_CHARS]
        out.append({"bbox": [x0, y0, x1, y1], "snippet": text, "label": str(item.get("label") or "")[:40]})
    return out


def _words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD.findall(text or "")) if len(w) >= MIN_WORD}


def place_words(page_no: int, bbox: list[float] | tuple[float, ...]) -> str:
    """Where a box is, in words: ``page 3, top left`` or ``the top left of the picture``."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    page = int(page_no) > 0
    if (x1 - x0) >= _WHOLE and (y1 - y0) >= _WHOLE:
        return f"page {int(page_no)}" if page else "the whole picture"
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    vertical = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    horizontal = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "centre")
    if vertical == "middle" and horizontal == "centre":
        spot = "middle"
    elif vertical == "middle":
        spot = horizontal
    elif horizontal == "centre":
        spot = vertical
    else:
        spot = f"{vertical} {horizontal}"
    return f"page {int(page_no)}, {spot}" if page else f"{spot} of the picture"


def match_fact(rows: list[dict[str, Any]], fact_text: str, limit: int = MAX_SHOWN) -> list[dict[str, Any]]:
    """The regions whose words the fact quotes, best first, as ``{"page", "bbox", "label", "where"}``."""
    fact = _words(fact_text)
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        words = _words(str(row.get("snippet") or ""))
        if not words:
            continue
        share = len(words & fact) / len(words)
        if share >= MATCH_SHARE:
            scored.append((share, row))
    scored.sort(key=lambda t: (-t[0], t[1].get("page_no", 0), t[1].get("y0", 0.0)))
    shown: list[dict[str, Any]] = []
    for _, row in scored[:limit]:
        box = [row["x0"], row["y0"], row["x1"], row["y1"]]
        shown.append({"page": int(row.get("page_no") or 0) or None, "bbox": box, "label": row.get("label") or "",
                      "where": "from " + place_words(int(row.get("page_no") or 0), box)})
    return shown


def media_db_next_to(conn: sqlite3.Connection) -> Path | None:
    """``media.db`` in the folder of the database ``conn`` has open, when there is one."""
    try:
        for _, name, file in conn.execute("PRAGMA database_list"):
            if name == "main" and file:
                found = Path(file).parent / "media.db"
                return found if found.is_file() else None
    except sqlite3.Error:
        return None
    return None


def for_fact(conn: sqlite3.Connection, memory_id: str, fact_text: str) -> list[dict[str, Any]]:
    """Where a fact's text sat on its picture or page. Read-only; ``[]`` when there is nothing to show."""
    path = media_db_next_to(conn)
    if path is None or not memory_id:
        return []
    try:
        ro = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return []
    try:
        ro.row_factory = sqlite3.Row
        rows = [dict(r) for r in ro.execute(
            "SELECT r.page_no, r.x0, r.y0, r.x1, r.y1, r.label, r.snippet FROM media_regions r"
            " WHERE r.media_id IN (SELECT media_id FROM media_items WHERE anchor_memory_id = ?1"
            "   UNION SELECT d.media_id FROM doc_pages d, json_each(d.memory_ids_json) j"
            "     WHERE j.value = ?1 AND d.media_id IS NOT NULL) LIMIT ?2", (memory_id, MAX_REGIONS * 4))]
    except sqlite3.Error:
        return []  # no such table (an older file) or a locked file: nothing to show
    finally:
        ro.close()
    return match_fact(rows, fact_text)


def record(store: Any, profile_id: str, media_id: str, page_no: int, raw: Iterable[Any], *, engine: str,
           redact: Any = None) -> int:
    """Keep the regions of one picture or page. Fail-open: a failure is logged and the save goes on."""
    try:
        rows = clean_regions(raw, redact)
        return store.put_regions(media_id, profile_id, page_no, rows, engine=engine) if rows else 0
    except Exception as exc:  # noqa: BLE001 - provenance never blocks a save
        logger.warning("regions of a picture were not kept (%s)", type(exc).__name__)
        return 0


__all__ = ["MAX_REGIONS", "MAX_SHOWN", "clean_regions", "for_fact", "match_fact", "place_words", "record"]
