# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Community summaries (Wave Q2) — one synthesized report per entity community.

Rides on the entity-community backbone (core.entity_community). For each
community it gathers the member entities' facts, EXCLUDES superseded facts
(bi-temporal — market CRIT-3), then produces:

  - a keyword-dense signal (always, Mode A, zero-LLM), and
  - a summary: Mode B/C LLM synthesis via the shared core.Summarizer
    (which itself falls back to a heuristic), else the Mode A keyword-dense
    line. Fail-open — a summarizer error never breaks generation.

Surfacing is on-device-safe: summaries are PRECOMPUTED here in the background
and later matched to a query as a single thematic-context block (Q2b) — never
a GraphRAG-style per-query LLM fan-out (market CRIT-1). member_fact_ids gives
drill-down back to the source atoms (Q3).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from typing import Any

from superlocalmemory.core.entity_community import EntityCommunityBuilder

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "and", "but", "or",
    "not", "no", "this", "that", "these", "those", "it", "its", "they",
    "them", "their", "he", "she", "his", "her", "we", "our", "you", "your",
    "i", "my", "me", "his", "was", "who", "what", "when", "where", "how",
})


class CommunitySummaryBuilder:
    """Generate + persist one summary per entity community (background)."""

    def __init__(
        self,
        db: Any,
        summarizer: Any = None,
        max_communities: int = 50,
        min_facts: int = 2,
        max_facts_per_community: int = 30,
        max_keywords: int = 8,
        summary_max_chars: int = 512,
    ) -> None:
        self._db = db
        self._summarizer = summarizer
        self._max_communities = max(1, int(max_communities))
        self._min_facts = max(1, int(min_facts))
        self._max_facts = max(1, int(max_facts_per_community))
        self._max_keywords = max(1, int(max_keywords))
        self._summary_max_chars = max(64, int(summary_max_chars))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def compute_and_store(self, profile_id: str) -> dict[str, int]:
        communities = EntityCommunityBuilder(self._db).get_communities(profile_id)
        try:
            self._db.execute(
                "DELETE FROM community_summaries WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("community_summaries clear failed: %s", exc)
        if not communities:
            return {"summaries_written": 0, "communities": 0}

        entity_to_cid: dict[str, int] = {
            e: cid for cid, ents in communities.items() for e in ents
        }
        cid_facts = self._gather_facts(profile_id, entity_to_cid)
        name_map = self._entity_names(profile_id, communities)

        written = 0
        ordered = sorted(
            cid_facts.items(), key=lambda kv: len(kv[1]), reverse=True,
        )
        for cid, facts in ordered:
            if written >= self._max_communities:
                break
            seen: set[str] = set()
            vf: list[tuple[str, str]] = []
            for fid, content in facts:
                if fid in seen:
                    continue
                seen.add(fid)
                vf.append((fid, content))
            if len(vf) < self._min_facts:
                continue

            fact_ids = [fid for fid, _ in vf]
            contents = [c for _, c in vf][: self._max_facts]
            entity_ids = list(communities.get(cid, []))
            entity_names = [name_map.get(e, e) for e in entity_ids]
            keywords = self._keywords(contents)
            summary = self._summary(contents, entity_names, keywords)

            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO community_summaries "
                    "(profile_id, community_id, summary, keywords, "
                    " entity_ids_json, fact_ids_json, fact_count, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                    (
                        profile_id, cid, summary, keywords,
                        json.dumps(entity_ids), json.dumps(fact_ids),
                        len(fact_ids),
                    ),
                )
                written += 1
            except Exception as exc:
                logger.debug("community_summaries write failed (%s): %s", cid, exc)

        return {"summaries_written": written, "communities": len(communities)}

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_summaries(self, profile_id: str) -> list[dict]:
        try:
            rows = self._db.execute(
                "SELECT * FROM community_summaries WHERE profile_id = ? "
                "ORDER BY fact_count DESC",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("get_summaries failed: %s", exc)
            return []
        return [dict(r) for r in rows]

    def get_summary(self, profile_id: str, community_id: int) -> dict | None:
        try:
            rows = self._db.execute(
                "SELECT * FROM community_summaries "
                "WHERE profile_id = ? AND community_id = ?",
                (profile_id, int(community_id)),
            )
        except Exception as exc:
            logger.debug("get_summary failed: %s", exc)
            return None
        return dict(rows[0]) if rows else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gather_facts(
        self, profile_id: str, entity_to_cid: dict[str, int],
    ) -> dict[int, list[tuple[str, str]]]:
        """One scan → {community_id -> [(fact_id, content)]}, superseded dropped."""
        try:
            rows = self._db.execute(
                "SELECT fact_id, canonical_entities_json, content "
                "FROM atomic_facts WHERE profile_id = ?",
                (profile_id,),
            )
        except Exception as exc:
            logger.debug("community fact scan failed: %s", exc)
            return {}

        cid_facts: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for row in rows:
            d = dict(row)
            raw = d.get("canonical_entities_json")
            if not raw:
                continue
            try:
                ents = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(ents, list):
                continue
            cids = {
                entity_to_cid[str(e).strip()]
                for e in ents
                if str(e).strip() in entity_to_cid
            }
            if not cids:
                continue
            fid = str(d["fact_id"])
            content = d.get("content") or ""
            for cid in cids:
                cid_facts[cid].append((fid, content))

        all_fids = list({fid for lst in cid_facts.values() for fid, _ in lst})
        invalid: set[str] = set()
        if all_fids:
            try:
                invalid = self._db.get_invalidated_fact_ids(all_fids, profile_id)
            except Exception as exc:
                logger.debug("invalidated-fact lookup failed: %s", exc)
        if invalid:
            cid_facts = {
                cid: [(fid, c) for fid, c in lst if fid not in invalid]
                for cid, lst in cid_facts.items()
            }
        return cid_facts

    def _entity_names(
        self, profile_id: str, communities: dict[int, list[str]],
    ) -> dict[str, str]:
        all_eids = list({e for ents in communities.values() for e in ents})
        name_map: dict[str, str] = {}
        chunk = 900
        for start in range(0, len(all_eids), chunk):
            batch = all_eids[start:start + chunk]
            ph = ",".join("?" for _ in batch)
            try:
                rows = self._db.execute(
                    "SELECT entity_id, canonical_name FROM canonical_entities "
                    f"WHERE profile_id = ? AND entity_id IN ({ph})",
                    (profile_id, *batch),
                )
            except Exception as exc:
                logger.debug("entity-name lookup failed: %s", exc)
                continue
            for r in rows:
                d = dict(r)
                name_map[str(d["entity_id"])] = str(d.get("canonical_name") or "")
        return name_map

    def _keywords(self, contents: list[str]) -> str:
        tokens: list[str] = []
        for text in contents:
            for word in text.lower().split():
                w = word.strip(".,;:!?\"'()[]{}")
                if len(w) > 2 and w not in _STOPWORDS:
                    tokens.append(w)
        top = [w for w, _ in Counter(tokens).most_common(self._max_keywords)]
        return ", ".join(top)

    def _summary(
        self, contents: list[str], entity_names: list[str], keywords: str,
    ) -> str:
        if self._summarizer is not None:
            try:
                text = self._summarizer.summarize_cluster(
                    [{"content": c} for c in contents],
                )
                if text and text.strip():
                    return text.strip()[: self._summary_max_chars]
            except Exception as exc:
                logger.debug("community summarizer failed (fail-open): %s", exc)
        # Mode A keyword-dense fallback.
        names = [n for n in entity_names if n]
        topic = ", ".join(names[:6]) if names else "related memories"
        base = f"Topics: {topic}."
        if keywords:
            base += f" Key terms: {keywords}."
        return base[: self._summary_max_chars]
