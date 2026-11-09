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

        Concurrency fix (v3.8.4): each operation uses a scoped connection via
        memory_read() / memory_write() so the process write lock is never held
        across Ollama / network calls and no long raw connection is kept open.
        """
        from superlocalmemory.storage.memory_write import memory_read

        if self._config and not getattr(self._config, 'entity_compilation_enabled', True):
            return {"compiled": 0, "skipped": 0, "errors": 0, "reason": "disabled"}

        stats = {"compiled": 0, "skipped": 0, "errors": 0}

        with memory_read(self._db_path) as conn:
            projects = conn.execute(
                "SELECT DISTINCT project_name FROM entity_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
            project_names = [r[0] for r in projects] if projects else [""]

        for project_name in project_names:
            result = self._compile_project(profile_id, project_name)
            stats["compiled"] += result["compiled"]
            stats["skipped"] += result["skipped"]
            stats["errors"] += result["errors"]

        if stats["compiled"] > 0:
            logger.info("Entity compilation: %d compiled, %d skipped, %d errors",
                        stats["compiled"], stats["skipped"], stats["errors"])
        return stats

    def compile_entity(self, profile_id: str, project_name: str,
                       entity_id: str, entity_name: str) -> dict | None:
        """Compile a single entity. Returns compiled truth or None."""
        return self._compile_single(profile_id, project_name, entity_id, entity_name)

    def _compile_project(self, profile_id: str, project_name: str) -> dict:
        """Compile all entities needing update in a project.

        Uses scoped memory_read() for the entity query so no connection is
        held across entity compilation iterations.
        """
        from superlocalmemory.storage.memory_write import memory_read

        stats = {"compiled": 0, "skipped": 0, "errors": 0}

        with memory_read(self._db_path) as conn:
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
            # Detach rows from the read connection before closing it
            entities = [dict(e) for e in entities]

        for entity in entities:
            try:
                result = self._compile_single(
                    profile_id, project_name,
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

    def _compile_single(self, profile_id: str,
                         project_name: str, entity_id: str, entity_name: str,
                         entity_type: str = "unknown") -> dict | None:
        """Compile one entity. Returns the compiled truth dict or None.

        Concurrency fix (v3.8.4): reads use memory_read(), writes use
        memory_write().  The Ollama call (Mode B) happens with NO lock held.
        Lock-ordering invariant preserved: get_write_lock (outermost) →
        sqlite BEGIN ... COMMIT.
        """
        from superlocalmemory.storage.memory_write import memory_read, memory_write

        # ── Phase 1: read facts (no write lock) ──────────────────────────────
        with memory_read(self._db_path) as conn:
            facts = conn.execute("""
                SELECT af.fact_id, af.content, af.confidence, af.created_at,
                       fi.pagerank_score, fi.community_id
                FROM atomic_facts af
                LEFT JOIN fact_importance fi ON af.fact_id = fi.fact_id
                WHERE af.canonical_entities_json LIKE ? AND af.profile_id = ?
                ORDER BY fi.pagerank_score DESC NULLS LAST, af.confidence DESC
                LIMIT 50
            """, (f"%{entity_id}%", profile_id)).fetchall()
            facts = [dict(f) for f in facts]

        if not facts:
            return None

        # There was a second PageRank here, and it made recall worse.
        #
        # On finding no score for this entity's facts it built a COMPLETE graph
        # over them -- every pair joined, weight 0.5 -- and ran PageRank on it.
        # PageRank of K_n is uniformly 1/n, so every fact received the identical
        # score and the re-fetch's ORDER BY fell straight through to confidence,
        # which is what had ordered them in the first place. It bought nothing.
        #
        # What it cost: that 1/n landed in the same column the ranker reads as a
        # whole-graph score. On the author's store, ten facts from one 10-fact
        # cluster were written 0.1 each, against a real whole-graph maximum of
        # 0.008744 and median of 0.000214 -- eleven times the largest true score
        # and roughly 470x the median. The hop boost min(1 + pr*2, 2) gave them
        # 1.2 where every real memory got 1.0004, and the ten of them together
        # carried as much PageRank mass as the other 2,988 facts combined (the
        # table summed to 1.9999 instead of 1). They were also written with no
        # community, so the community bias could not see them either.
        #
        # Whole-graph metrics belong to core/graph_metrics, which owns this
        # table. Ordering for compilation is confidence, as it always effectively
        # was.

        # ── Phase 3: generate compiled truth — NO write lock held ────────────
        # Mode B calls Ollama (up to 30 s) — write lock MUST NOT be held here.
        if self._mode in ("b", "c") and len(facts) > 3:
            compiled = self._compile_with_llm(entity_name, facts)
            if not compiled:
                compiled = self._compile_mode_a(entity_name, entity_type, facts)
        else:
            compiled = self._compile_mode_a(entity_name, entity_type, facts)

        compiled = self._truncate(compiled, _MAX_COMPILED_TRUTH_CHARS)

        now = datetime.now(timezone.utc).isoformat()
        timeline_entry = {
            "date": now,
            "action": "compiled",
            "facts_used": len(facts),
            "mode": self._mode,
        }

        fact_ids = [f["fact_id"] for f in facts]
        avg_conf = sum(f["confidence"] or 0.5 for f in facts) / max(len(facts), 1)

        # ── Phase 4: short write — hold write lock for INSERT/UPDATE only ────
        with memory_write(self._db_path) as conn:
            # Re-read inside the write lock for correct timeline merge and to
            # decide INSERT vs UPDATE atomically (prevents lost-update race).
            existing = conn.execute(
                "SELECT timeline, profile_entry_id FROM entity_profiles "
                "WHERE entity_id = ? AND profile_id = ? AND project_name = ?",
                (entity_id, profile_id, project_name),
            ).fetchone()

            timeline: list = []
            if existing and existing["timeline"]:
                try:
                    timeline = json.loads(existing["timeline"])
                except (json.JSONDecodeError, TypeError):
                    timeline = []
            timeline.append(timeline_entry)
            if len(timeline) > _MAX_TIMELINE_ENTRIES:
                timeline = timeline[-_MAX_TIMELINE_ENTRIES:]

            if existing:
                conn.execute("""
                    UPDATE entity_profiles SET
                        compiled_truth = ?, timeline = ?, fact_ids_json = ?,
                        last_compiled_at = ?, compilation_confidence = ?,
                        last_updated = ?
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

    # -- Mode B/C: LLM --

    def _compile_with_llm(self, entity_name: str, facts: list) -> str | None:
        """Mode B: the configured Ollama. Mode C: the configured provider.

        Mode C used to reuse the Ollama path, which posted the facts to
        ``{cloud api_base}/api/generate``. Both paths send through the
        outbound gate, so a host other than this machine never receives a
        credential.
        """
        if self._mode == "c":
            return self._compile_mode_c(entity_name, facts)
        return self._compile_mode_b(entity_name, facts)

    def _compile_mode_c(self, entity_name: str, facts: list) -> str | None:
        """Summarize via the configured cloud provider. None on failure."""
        try:
            from superlocalmemory.llm.backbone import LLMBackbone

            llm_config = getattr(self._config, "llm", None)
            if llm_config is None or not getattr(llm_config, "provider", ""):
                return None
            llm = LLMBackbone(llm_config)
            if not llm.is_available():
                return None
            text = llm.generate(
                prompt=self._compile_prompt(entity_name, facts),
                max_tokens=500, attempts=1,
            )
            return text.strip() or None
        except Exception as exc:
            logger.debug("Mode C compilation failed, falling back to Mode A: %s", exc)
            return None

    @staticmethod
    def _compile_prompt(entity_name: str, facts: list) -> str:
        top_facts = "\n".join(f"- {f['content']}" for f in facts[:20])
        return (
            f"Summarize these facts about {entity_name} into a concise profile. "
            f"Maximum 2000 characters. Include key relationships, decisions, status. "
            f"Organize by topic, not chronology. Flag contradictions.\n\n"
            f"Facts (by importance):\n{top_facts}"
        )

    def _compile_mode_b(self, entity_name: str, facts: list) -> str | None:
        """Summarize via the configured Ollama. Returns None on failure."""
        try:
            import urllib.request

            from superlocalmemory.core import outbound_http
            api_base = "http://localhost:11434"
            if self._config and hasattr(self._config, 'llm'):
                api_base = getattr(self._config.llm, 'api_base', api_base) or api_base
            model = "llama3.2"
            if self._config and hasattr(self._config, 'llm'):
                model = getattr(self._config.llm, 'model', model) or model

            prompt = self._compile_prompt(entity_name, facts)

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
            resp = outbound_http.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            text = result.get("response", "").strip()
            return text if text else None
        except Exception as exc:
            logger.debug("Mode B compilation failed, falling back to Mode A: %s", exc)
            return None

    # -- Helpers --

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
