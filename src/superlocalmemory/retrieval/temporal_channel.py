# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Temporal Retrieval Channel (3-Date Model).

Searches by referenced_date (NOT just created_at like V1).
Returns empty when query has no temporal signal (no recency noise).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import TYPE_CHECKING

from dateutil.parser import ParserError, parse as dateutil_parse

from superlocalmemory.encoding.temporal_parser import TemporalParser
from superlocalmemory.storage.database import _scope_where

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)

_MAX_PROXIMITY_DAYS: float = 365.0


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    # v3.5.0 perf: stored dates are ISO-8601, so try the C-level
    # datetime.fromisoformat (~1µs) FIRST. dateutil.parser.parse is ~100x
    # slower (~100µs) and the temporal channel parses up to 4 dates per event
    # across thousands of events — that was ~2.6s of the recall. dateutil is
    # now only the fallback for non-ISO strings.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    try:
        return dateutil_parse(s)
    except (ParserError, ValueError, OverflowError, TypeError):
        return None


def _proximity_score(q: datetime, e: datetime) -> float:
    """Gaussian proximity: same day=1.0, 30d=0.61, 90d=0.11."""
    dist = abs((q - e).total_seconds()) / 86400.0
    if dist > _MAX_PROXIMITY_DAYS:
        return 0.0
    return math.exp(-(dist * dist) / (2.0 * 30.0 * 30.0))


class TemporalChannel:
    """Date-aware retrieval using the 3-date temporal model."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def search(self, query: str, profile_id: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Search for temporally relevant facts.

        Two strategies:
        1. Date proximity: scores events by date closeness to query date.
        2. Entity-temporal: filters events by entity name in query,
           returns ALL their temporal facts (metadata-first approach).

        Returns empty only when query has no temporal signal AND no
        entity-temporal matches.
        """
        parser = TemporalParser()
        dates = parser.extract_dates_from_text(query)
        query_dt = _parse_iso(dates.get("referenced_date"))
        if query_dt is None:
            query_dt = self._try_parse(query)

        # Strategy 1: Entity-temporal metadata search
        # "When did Alice...?" → find all temporal events for Alice
        entity_results = self._entity_temporal_search(query, profile_id)

        # Strategy 2: Date proximity search
        if query_dt is None and not entity_results:
            return []

        events = self._load_events(profile_id)
        scored: dict[str, float] = {}

        # Include entity-temporal results with high base score
        for fid, score in entity_results:
            scored[fid] = max(scored.get(fid, 0.0), score)

        if query_dt is not None:
            for ev in events:
                best = 0.0
                ref = _parse_iso(ev.get("referenced_date"))
                if ref is not None:
                    best = max(best, _proximity_score(query_dt, ref))

                obs = _parse_iso(ev.get("observation_date"))
                if obs is not None:
                    best = max(best, _proximity_score(query_dt, obs) * 0.8)

                i_start = _parse_iso(ev.get("interval_start"))
                i_end = _parse_iso(ev.get("interval_end"))
                if i_start and i_end:
                    if i_start <= query_dt <= i_end:
                        best = max(best, 1.0)
                    else:
                        best = max(best, max(
                            _proximity_score(query_dt, i_start),
                            _proximity_score(query_dt, i_end),
                        ) * 0.9)

                if best > 0.0:
                    fid = ev["fact_id"]
                    scored[fid] = max(scored.get(fid, 0.0), best)

        results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _entity_temporal_search(
        self, query: str, profile_id: str,
    ) -> list[tuple[str, float]]:
        """Metadata-first: find temporal events for entities mentioned in query.

        "When did Alice do X?" → SQL filter by entity_id for Alice → return
        all temporal facts about Alice. High precision for entity+time queries.
        """
        import re
        _PROPER_RE = re.compile(r"\b([A-Z][a-z]+)\b")
        names = [m.group(1) for m in _PROPER_RE.finditer(query)]
        # Also try title-cased version for lowercase queries
        if not names:
            names = [m.group(1) for m in _PROPER_RE.finditer(query.title())]
        # Filter out common words from title-casing
        _stop = {"What", "When", "Where", "Who", "Which", "How", "Does", "Did",
                 "The", "That", "This", "There", "Then", "Have", "Has", "Had",
                 "About", "After", "Before", "From", "With", "Would", "Could",
                 "Should", "Will", "Because", "Also", "Just", "Like", "Know",
                 "Think", "Tell", "Said"}
        names = [n for n in names if n not in _stop]
        if not names:
            return []

        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        where, params = _scope_where(
            profile_id,
            include_global=bool(getattr(self, "include_global", False)),
            include_shared=bool(getattr(self, "include_shared", False)),
            prefix="te",
        )

        for name in names[:3]:  # Limit to first 3 entity mentions
            # Resolve the entity and event in one scope-filtered query. Looking
            # up the entity only in the requester's profile made global events
            # owned by another profile undiscoverable before authorization was
            # even evaluated.
            rows = self._db.execute(
                "SELECT te.fact_id FROM temporal_events AS te "
                "JOIN canonical_entities AS ce ON ce.entity_id = te.entity_id "
                f"WHERE {where} AND LOWER(ce.canonical_name) = LOWER(?)",
                (*params, name),
            )
            for row in rows:
                fid = dict(row)["fact_id"]
                if fid not in seen:
                    seen.add(fid)
                    # Rank by position (first events more likely relevant) instead
                    # of flat 0.85 which loses discrimination
                    rank_score = 0.85 - len(seen) * 0.02
                    results.append((fid, max(0.3, rank_score)))

        return results

    def _load_events(self, profile_id: str) -> list[dict]:
        where, params = _scope_where(
            profile_id,
            include_global=bool(getattr(self, "include_global", False)),
            include_shared=bool(getattr(self, "include_shared", False)),
        )
        rows = self._db.execute(
            "SELECT fact_id, observation_date, referenced_date, "
            "interval_start, interval_end "
            f"FROM temporal_events WHERE {where}",
            (*params,),
        )
        return [dict(r) for r in rows]

    @staticmethod
    def _try_parse(text: str) -> datetime | None:
        """Fuzzy date parse with safety guards.

        dateutil fuzzy=True is exponential on long non-date text.
        Guard: only attempt on short strings (< 60 chars) that contain
        at least one digit (dates always have numbers).
        """
        if len(text) > 60 or not any(c.isdigit() for c in text):
            return None
        try:
            return dateutil_parse(text, fuzzy=True)
        except (ParserError, ValueError, OverflowError):
            return None
