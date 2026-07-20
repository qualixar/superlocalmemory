# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Entity Compilation Engine — auto-generates compiled truth per entity.

Builds knowledge summaries using PageRank centrality + Louvain community detection
(Mode A extractive) or local LLM (Mode B). Per-project, per-profile scoping.
2000 character hard limit. Read-only layer — never replaces atomic facts.

Runs after consolidation (every 6 hours or on-demand).

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("superlocalmemory.entity_compiler")

_MAX_COMPILED_TRUTH_CHARS = 2000
_MAX_TIMELINE_ENTRIES = 100


class EntityCompiler:
    """Compiles knowledge summaries for entities from atomic facts.

    Mode A: Extractive (no LLM) — PageRank + Louvain + top sentences
    Mode B: Local LLM via Ollama — prompt with top facts
    """

    def __init__(self, memory_db: str | Path, config=None):
        self._db_path = str(memory_db)
        self._config = config
        self._mode = "a"
        if config:
            mode = getattr(config, 'mode', None)
            if mode:
                self._mode = getattr(mode, 'value', str(mode)).lower()

    def compile_all(self, profile_id: str) -> dict:
        """Compile all entities that have new facts across all projects.

        Returns stats: {compiled: N, skipped: N, errors: N}
        """
        if self._config and not getattr(self._config, 'entity_compilation_enabled', True):
            return {"compiled": 0, "skipped": 0, "errors": 0, "reason": "disabled"}

        stats = {"compiled": 0, "skipped": 0, "errors": 0}
        conn = self._connect()
        try:
            # Get all distinct projects for this profile
            projects = conn.execute(
                "SELECT DISTINCT project_name FROM entity_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
            project_names = [r[0] for r in projects] if projects else [""]

            for project_name in project_names:
                result = self._compile_project(conn, profile_id, project_name)
                stats["compiled"] += result["compiled"]
                stats["skipped"] += result["skipped"]
                stats["errors"] += result["errors"]
        finally:
            conn.close()

        if stats["compiled"] > 0:
            logger.info("Entity compilation: %d compiled, %d skipped, %d errors",
                        stats["compiled"], stats["skipped"], stats["errors"])
        return stats

    def compile_entity(self, profile_id: str, project_name: str,
                       entity_id: str, entity_name: str) -> dict | None:
        """Compile a single entity. Returns compiled truth or None."""
        conn = self._connect()
        try:
            return self._compile_single(conn, profile_id, project_name,
                                         entity_id, entity_name)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _compile_project(self, conn: sqlite3.Connection, profile_id: str,
                          project_name: str) -> dict:
        """Compile all entities needing update in a project."""
        stats = {"compiled": 0, "skipped": 0, "errors": 0}

        # Find entities with new facts since last compilation
        entities = conn.execute("""
            SELECT DISTINCT ce.entity_id, ce.canonical_name, ce.entity_type
            FROM canonical_entities ce
            WHERE ce.profile_id = ?
            AND (
                EXISTS (
                    SELECT 1 FROM atomic_facts af
                    WHERE af.canonical_entities_json LIKE '%' || ce.entity_id || '%'
                      AND af.profile_id = ?
                      AND af.created_at > COALESCE(
                        (SELECT last_compiled_at FROM entity_profiles
                         WHERE entity_id = ce.entity_id
                           AND profile_id = ?
                           AND project_name = ?),
                        '1970-01-01')
                )
                OR NOT EXISTS (
                    SELECT 1 FROM entity_profiles
                    WHERE entity_id = ce.entity_id
                      AND profile_id = ?
                      AND project_name = ?
                      AND last_compiled_at IS NOT NULL
                )
            )
        """, (profile_id, profile_id, profile_id, project_name,
              profile_id, project_name)).fetchall()

        for entity in entities:
            try:
                result = self._compile_single(
                    conn, profile_id, project_name,
                    entity["entity_id"], entity["canonical_name"],
                    entity_type=entity["entity_type"],
                )
                if result:
                    stats["compiled"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as exc:
                logger.debug("Entity compilation error for %s: %s",
                             entity["canonical_name"], exc)
                stats["errors"] += 1

        return stats

    def _compile_single(self, conn: sqlite3.Connection, profile_id: str,
                         project_name: str, entity_id: str, entity_name: str,
                         entity_type: str = "unknown") -> dict | None:
        """Compile one entity. Returns the compiled truth dict or None."""

        # Gather atomic facts for this entity
        facts = conn.execute("""
            SELECT af.fact_id, af.content, af.confidence, af.created_at,
                   fi.pagerank_score, fi.community_id
            FROM atomic_facts af
            LEFT JOIN fact_importance fi ON af.fact_id = fi.fact_id
            WHERE af.canonical_entities_json LIKE ? AND af.profile_id = ?
            ORDER BY fi.pagerank_score DESC NULLS LAST, af.confidence DESC
            LIMIT 50
        """, (f"%{entity_id}%", profile_id)).fetchall()

        if not facts:
            return None

        # Compute PageRank if missing
        has_pagerank = any(f["pagerank_score"] is not None for f in facts)
        if not has_pagerank and len(facts) > 2:
            self._compute_pagerank(conn, [f["fact_id"] for f in facts], profile_id)
            # Re-fetch with scores
            facts = conn.execute("""
                SELECT af.fact_id, af.content, af.confidence, af.created_at,
                       fi.pagerank_score, fi.community_id
                FROM atomic_facts af
                LEFT JOIN fact_importance fi ON af.fact_id = fi.fact_id
                WHERE af.canonical_entities_json LIKE ? AND af.profile_id = ?
                ORDER BY fi.pagerank_score DESC NULLS LAST, af.confidence DESC
                LIMIT 50
            """, (f"%{entity_id}%", profile_id)).fetchall()

        # Generate compiled truth
        if self._mode in ("b", "c") and len(facts) > 3:
            compiled = self._compile_mode_b(entity_name, facts)
            if not compiled:
                compiled = self._compile_mode_a(entity_name, entity_type, facts)
        else:
            compiled = self._compile_mode_a(entity_name, entity_type, facts)

        # Truncate to limit
        compiled = self._truncate(compiled, _MAX_COMPILED_TRUTH_CHARS)

        # Build timeline entry
        now = datetime.now(timezone.utc).isoformat()
        timeline_entry = {
            "date": now,
            "action": "compiled",
            "facts_used": len(facts),
            "mode": self._mode,
        }

        # Load existing timeline
        existing = conn.execute(
            "SELECT timeline, profile_entry_id FROM entity_profiles "
            "WHERE entity_id = ? AND profile_id = ? AND project_name = ?",
            (entity_id, profile_id, project_name),
        ).fetchone()

        timeline = []
        if existing and existing["timeline"]:
            try:
                timeline = json.loads(existing["timeline"])
            except (json.JSONDecodeError, TypeError):
                timeline = []
        timeline.append(timeline_entry)
        # Cap at 100 entries
        if len(timeline) > _MAX_TIMELINE_ENTRIES:
            timeline = timeline[-_MAX_TIMELINE_ENTRIES:]

        fact_ids = [f["fact_id"] for f in facts]
        avg_conf = sum(f["confidence"] or 0.5 for f in facts) / max(len(facts), 1)

        # Upsert
        if existing:
            conn.execute("""
                UPDATE entity_profiles SET
                    compiled_truth = ?, timeline = ?, fact_ids_json = ?,
                    last_compiled_at = ?, compilation_confidence = ?, last_updated = ?
                WHERE entity_id = ? AND profile_id = ? AND project_name = ?
            """, (compiled, json.dumps(timeline), json.dumps(fact_ids),
                  now, round(avg_conf, 3), now,
                  entity_id, profile_id, project_name))
        else:
            entry_id = str(uuid.uuid4())[:16]
            conn.execute("""
                INSERT INTO entity_profiles
                    (profile_entry_id, entity_id, profile_id, project_name,
                     knowledge_summary, compiled_truth, timeline, fact_ids_json,
                     last_compiled_at, compilation_confidence, last_updated)
                VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)
            """, (entry_id, entity_id, profile_id, project_name,
                  compiled, json.dumps(timeline), json.dumps(fact_ids),
                  now, round(avg_conf, 3), now))

        conn.commit()

        return {
            "entity_name": entity_name,
            "compiled_truth": compiled,
            "facts_used": len(facts),
            "confidence": round(avg_conf, 3),
        }

    # -- Mode A: Extractive (no LLM) --

    def _compile_mode_a(self, entity_name: str, entity_type: str,
                         facts: list) -> str:
        """Extract top sentences by PageRank, grouped by community."""
        header = f"{entity_name}"
        if entity_type and entity_type != "unknown":
            header += f" ({entity_type})"
        header += "\n"

        # Group facts by community
        communities: dict[int, list] = {}
        for f in facts:
            cid = f["community_id"] or 0
            communities.setdefault(cid, []).append(f)

        sentences = []
        seen_content = set()
        for cid in sorted(communities.keys()):
            community_facts = communities[cid]
            # Top 3 facts per community
            for fact in community_facts[:3]:
                content = fact["content"]
                # Extract first sentence
                first_sent = content.split(". ")[0].strip()
                if not first_sent.endswith("."):
                    first_sent += "."
                # Dedup by exact match
                normalized = first_sent.lower().strip()
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    sentences.append(first_sent)

        body = " ".join(sentences)
        return header + body

    # -- Mode B: LLM via Ollama --

    def _compile_mode_b(self, entity_name: str, facts: list) -> str | None:
        """Summarize via local LLM (Ollama). Returns None on failure."""
        try:
            import urllib.request
            api_base = "http://localhost:11434"
            if self._config and hasattr(self._config, 'llm'):
                api_base = getattr(self._config.llm, 'api_base', api_base) or api_base
            model = "llama3.2"
            if self._config and hasattr(self._config, 'llm'):
                model = getattr(self._config.llm, 'model', model) or model

            top_facts = "\n".join(
                f"- {f['content']}" for f in facts[:20]
            )
            prompt = (
                f"Summarize these facts about {entity_name} into a concise profile. "
                f"Maximum 2000 characters. Include key relationships, decisions, status. "
                f"Organize by topic, not chronology. Flag contradictions.\n\n"
                f"Facts (by importance):\n{top_facts}"
            )

            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500},
            }).encode()

            req = urllib.request.Request(
                f"{api_base}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            text = result.get("response", "").strip()
            return text if text else None
        except Exception as exc:
            logger.debug("Mode B compilation failed, falling back to Mode A: %s", exc)
            return None

    # -- Helpers --

    def _compute_pagerank(self, conn: sqlite3.Connection,
                           fact_ids: list[str], profile_id: str) -> None:
        """Compute PageRank for a set of facts. Stores in fact_importance."""
        try:
            import networkx as nx
            G = nx.Graph()
            for fid in fact_ids:
                G.add_node(fid)
            # Add edges based on shared entities
            for i, fid1 in enumerate(fact_ids):
                for fid2 in fact_ids[i + 1:]:
                    # Simple heuristic: facts about same entity are connected
                    G.add_edge(fid1, fid2, weight=0.5)

            if len(G.nodes) < 2:
                return

            scores = nx.pagerank(G, alpha=0.85)
            now = datetime.now(timezone.utc).isoformat()

            for fid, score in scores.items():
                conn.execute("""
                    INSERT INTO fact_importance (fact_id, profile_id, pagerank_score, computed_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(fact_id) DO UPDATE SET pagerank_score=excluded.pagerank_score,
                                                       computed_at=excluded.computed_at
                """, (fid, profile_id, round(score, 6), now))
            conn.commit()
        except ImportError:
            logger.debug("NetworkX not available — skipping PageRank")
        except Exception as exc:
            logger.debug("PageRank computation failed: %s", exc)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        """Truncate at sentence boundary within char limit."""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars // 2:
            return truncated[:last_period + 1]
        return truncated.rstrip() + "..."
