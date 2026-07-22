# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Progressive abstraction (Wave Q3) — the top persona tier + drill-down.

Completes the abstraction hierarchy on the ONE principled backbone (rather
than a fourth divergent clustering):

    atoms (atomic_facts)
      -> entity communities        (Wave Q backbone)
        -> community summaries      (Wave Q2)
          -> persona roll-up        (this module)

The persona is one bounded roll-up per profile that consumes the top community
summaries. It is recall-GATED (never auto-injected into the hot recall path —
avoids the V3.4.40 summary-pollution regression) and SIZE-bounded. Drill-down
(``get_sources``) walks the hierarchy back down to the source atoms, matching
the market bar for summary->source provenance (Zep-style).

Runs in the background consolidation lane after community summaries.
Fail-open throughout; recompute replaces a profile's row.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from typing import Any

from superlocalmemory.core.community_summary import CommunitySummaryBuilder

logger = logging.getLogger(__name__)

_PERSONA_MAX_CHARS = 2048


class ProgressiveAbstraction:
    """Build + persist the persona tier; provide hierarchy drill-down."""

    def __init__(
        self,
        db: Any,
        summarizer: Any = None,
        max_communities_in_persona: int = 8,
        persona_max_chars: int = _PERSONA_MAX_CHARS,
        max_keywords: int = 12,
    ) -> None:
        self._db = db
        self._summarizer = summarizer
        self._max_communities = max(1, int(max_communities_in_persona))
        self._persona_max_chars = max(256, int(persona_max_chars))
        self._max_keywords = max(1, int(max_keywords))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def compute_and_store(self, profile_id: str) -> dict[str, Any]:
        summaries = CommunitySummaryBuilder(self._db).get_summaries(profile_id)
        try:
            self._db.execute(
                "DELETE FROM persona_summary WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("persona_summary clear failed: %s", exc)
        if not summaries:
            return {"built": False, "communities_in_persona": 0}

        top = summaries[: self._max_communities]
        summary = self._persona_summary(top)
        keywords = self._merge_keywords(top)
        community_ids = [int(s["community_id"]) for s in top]

        try:
            self._db.execute(
                "INSERT OR REPLACE INTO persona_summary "
                "(profile_id, summary, keywords, community_ids_json, computed_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                (profile_id, summary, keywords, json.dumps(community_ids)),
            )
        except Exception as exc:
            logger.debug("persona_summary write failed: %s", exc)
            return {"built": False, "communities_in_persona": 0}

        return {"built": True, "communities_in_persona": len(top)}

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_persona(self, profile_id: str) -> dict | None:
        try:
            rows = self._db.execute(
                "SELECT * FROM persona_summary WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("get_persona failed: %s", exc)
            return None
        if not rows:
            return None
        d = dict(rows[0])
        try:
            community_ids = json.loads(d.get("community_ids_json") or "[]")
        except (ValueError, TypeError):
            community_ids = []
        return {
            "profile_id": d.get("profile_id", profile_id),
            "summary": d.get("summary", ""),
            "keywords": d.get("keywords", ""),
            "community_ids": community_ids,
            "computed_at": d.get("computed_at", ""),
        }

    def get_sources(self, profile_id: str, node_id: Any) -> dict:
        """Drill-down: a tier node -> its child communities + source atoms.

        node_id == "persona"  -> the persona's member communities + their facts.
        node_id == <community> -> that community's member facts.
        Unknown node -> empty (never raises).
        """
        result: dict[str, Any] = {
            "node_id": node_id, "node_type": "unknown",
            "communities": [], "fact_ids": [],
        }
        try:
            if isinstance(node_id, str) and node_id.lower() == "persona":
                persona = self.get_persona(profile_id)
                cids = persona["community_ids"] if persona else []
                fact_ids: list[str] = []
                seen: set[str] = set()
                for cid in cids:
                    for fid in self._community_fact_ids(profile_id, cid):
                        if fid not in seen:
                            seen.add(fid)
                            fact_ids.append(fid)
                result.update(
                    node_type="persona", communities=list(cids), fact_ids=fact_ids,
                )
                return result

            # Otherwise treat node_id as a community id.
            cid = int(node_id)
            fids = self._community_fact_ids(profile_id, cid)
            result.update(
                node_type="community", communities=[cid], fact_ids=fids,
            )
            return result
        except (ValueError, TypeError):
            return result
        except Exception as exc:
            logger.debug("get_sources failed: %s", exc)
            return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _community_fact_ids(self, profile_id: str, community_id: Any) -> list[str]:
        try:
            rows = self._db.execute(
                "SELECT fact_ids_json FROM community_summaries "
                "WHERE profile_id = ? AND community_id = ?",
                (profile_id, int(community_id)),
            )
        except Exception as exc:
            logger.debug("_community_fact_ids failed: %s", exc)
            return []
        if not rows:
            return []
        try:
            return [str(f) for f in json.loads(dict(rows[0]).get("fact_ids_json") or "[]")]
        except (ValueError, TypeError):
            return []

    def _persona_summary(self, top: list[dict]) -> str:
        if self._summarizer is not None:
            try:
                text = self._summarizer.summarize_cluster(
                    [{"content": s.get("summary", "")} for s in top],
                )
                if text and text.strip():
                    return text.strip()[: self._persona_max_chars]
            except Exception as exc:
                logger.debug("persona summarizer failed (fail-open): %s", exc)
        # Mode A keyword-dense fallback: stitch the top community summaries.
        heads = [s.get("summary", "").strip() for s in top if s.get("summary")]
        base = " ".join(heads) if heads else "No persona yet."
        return base[: self._persona_max_chars]

    def _merge_keywords(self, top: list[dict]) -> str:
        seen: set[str] = set()
        merged: list[str] = []
        for s in top:
            for kw in (s.get("keywords", "") or "").split(","):
                k = kw.strip()
                low = k.lower()
                if k and low not in seen:
                    seen.add(low)
                    merged.append(k)
                    if len(merged) >= self._max_keywords:
                        return ", ".join(merged)
        return ", ".join(merged)
