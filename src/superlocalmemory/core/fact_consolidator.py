# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3.4.11 "Scale-Ready" — Fact Consolidation Engine.

Merges clusters of related facts about the same entity into single
comprehensive summary facts. Original facts move to 'archived' tier
but are NEVER deleted — searchable via deep recall.

Uses Mode B (Ollama LLM) for summarization, with Mode A (extractive)
fallback if LLM is unavailable.

CRITICAL RULES:
  1. NEVER delete original facts
  2. Original facts → lifecycle='archived' (not deleted)
  3. Consolidated fact links back to originals via fact_consolidations table
  4. Only consolidates facts that are already 'warm' or 'cold' tier
  5. Never touches 'active' or 'pinned' facts
  6. All writes per cluster wrapped in SAVEPOINT for atomicity
  7. Entity ID LIKE patterns use JSON-boundary quoting to prevent
     substring false positives

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("superlocalmemory.fact_consolidator")

_MAX_CLUSTER_SIZE = 10   # Max facts to merge into one
_MIN_CLUSTER_SIZE = 3    # Need at least 3 related facts to consolidate
_MAX_CONSOLIDATED_CHARS = 2000


def consolidate_facts(
    db_path: str | Path,
    profile_id: str = "default",
    max_clusters: int = 20,
    dry_run: bool = False,
    config: object | None = None,
) -> dict:
    """Find and consolidate clusters of related facts.

    Mode behavior:
      - Mode A: Extractive only (no LLM). Always available.
      - Mode B: Ollama LLM summarization. Falls back to extractive if Ollama down.
      - Mode C: Cloud LLM (user's configured provider). Falls back to extractive.

    Returns stats: consolidated, clusters_found, facts_archived, errors.
    """
    stats = {
        "clusters_found": 0,
        "consolidated": 0,
        "facts_archived": 0,
        "errors": 0,
        "error_detail": "",
        "mode": "a",
    }

    if config:
        mode = getattr(config, 'mode', None)
        if mode:
            mode_str = getattr(mode, 'value', str(mode)).lower()
            stats["mode"] = mode_str

    conn = sqlite3.connect(str(db_path))
    wal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if wal_mode and wal_mode[0] != "wal":
        logger.warning("WAL mode not active, got: %s", wal_mode[0])
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row

    try:
        clusters = _find_consolidation_clusters(conn, profile_id, max_clusters)
        stats["clusters_found"] = len(clusters)

        for entity_id, entity_name, fact_ids in clusters:
            try:
                result = _consolidate_cluster(
                    conn, profile_id, entity_id, entity_name,
                    fact_ids, dry_run, config,
                )
                if result:
                    stats["consolidated"] += 1
                    stats["facts_archived"] += len(fact_ids)
            except Exception as exc:
                logger.warning(
                    "Consolidation failed for %s: %s",
                    entity_name, exc, exc_info=True,
                )
                stats["errors"] += 1

        if not dry_run:
            conn.commit()

        if stats["consolidated"] > 0:
            logger.info(
                "Fact consolidation: %d clusters merged, %d facts archived",
                stats["consolidated"], stats["facts_archived"],
            )
    except Exception as exc:
        logger.error("Fact consolidation failed: %s", exc, exc_info=True)
        stats["errors"] += 1
        stats["error_detail"] = str(exc)
    finally:
        conn.close()

    return stats


def _find_consolidation_clusters(
    conn: sqlite3.Connection,
    profile_id: str,
    max_clusters: int,
) -> list[tuple[str, str, list[str]]]:
    """Find entities with clusters of warm/cold facts ready for consolidation.

    Uses JSON-boundary quoting on entity_id to prevent substring false positives.
    Both outer count and inner fact query are scoped to profile_id.
    """
    c = conn.cursor()

    # Find entities with many non-active, non-pinned facts
    # Uses '%" entity_id "%' pattern for JSON boundary matching
    entities = c.execute("""
        SELECT ce.entity_id, ce.canonical_name, COUNT(af.fact_id) as fact_count
        FROM canonical_entities ce
        JOIN atomic_facts af
          ON af.canonical_entities_json LIKE '%"' || ce.entity_id || '"%'
         AND af.profile_id = ?
        WHERE ce.profile_id = ?
          AND af.lifecycle IN ('warm', 'cold')
          AND af.fact_id NOT IN (
            SELECT fact_id FROM pinned_facts WHERE profile_id = ?
          )
        GROUP BY ce.entity_id
        HAVING COUNT(af.fact_id) >= ?
        ORDER BY COUNT(af.fact_id) DESC
        LIMIT ?
    """, (profile_id, profile_id, profile_id, _MIN_CLUSTER_SIZE,
          max_clusters)).fetchall()

    clusters = []
    for entity in entities:
        eid = entity["entity_id"]
        facts = c.execute("""
            SELECT af.fact_id FROM atomic_facts af
            WHERE af.canonical_entities_json LIKE ?
              AND af.profile_id = ?
              AND af.lifecycle IN ('warm', 'cold')
              AND af.fact_id NOT IN (
                SELECT fact_id FROM pinned_facts WHERE profile_id = ?
              )
            ORDER BY af.confidence DESC, af.created_at DESC
            LIMIT ?
        """, (f'%"{eid}"%', profile_id, profile_id,
              _MAX_CLUSTER_SIZE)).fetchall()

        fact_ids = [f["fact_id"] for f in facts]
        if len(fact_ids) >= _MIN_CLUSTER_SIZE:
            clusters.append((eid, entity["canonical_name"], fact_ids))

    return clusters


def _consolidate_cluster(
    conn: sqlite3.Connection,
    profile_id: str,
    entity_id: str,
    entity_name: str,
    fact_ids: list[str],
    dry_run: bool,
    config: object | None = None,
) -> dict | None:
    """Merge a cluster of facts into one consolidated fact.

    All writes are wrapped in a SAVEPOINT for atomicity — if any step fails,
    the entire cluster consolidation is rolled back.
    """
    c = conn.cursor()

    # Load fact contents including canonical_entities_json
    placeholders = ",".join("?" * len(fact_ids))
    facts = c.execute(
        f"SELECT fact_id, content, confidence, created_at, canonical_entities_json, "
        f"scope, shared_with "
        f"FROM atomic_facts "
        f"WHERE fact_id IN ({placeholders}) ORDER BY created_at",
        fact_ids,
    ).fetchall()

    if len(facts) < _MIN_CLUSTER_SIZE:
        return None

    summary = _generate_summary(entity_name, facts, config)
    if not summary:
        return None

    if dry_run:
        return {"entity": entity_name, "facts": len(facts), "summary_len": len(summary)}

    # Use SAVEPOINT for atomic multi-step write
    savepoint_name = f"consolidate_{uuid.uuid4().hex[:8]}"
    c.execute(f"SAVEPOINT {savepoint_name}")

    try:
        new_fact_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        avg_confidence = sum(f["confidence"] or 0.5 for f in facts) / len(facts)

        # v3.6.15 multi-scope: a summary must never be MORE visible than its
        # sources, or it would leak a private fact into a shared/global summary.
        # Preserve scope only when the whole cluster agrees; any mix (or shared
        # facts with differing targets) falls back to 'personal' — the most
        # restrictive scope. All-personal clusters (the common case) are
        # unchanged. shared_with is preserved only for a uniform shared cluster.
        _src_scopes = {(f["scope"] or "personal") for f in facts}
        _src_shared = {f["shared_with"] for f in facts}
        if _src_scopes == {"global"}:
            _sum_scope, _sum_shared = "global", None
        elif _src_scopes == {"shared"} and len(_src_shared) == 1:
            _sum_scope, _sum_shared = "shared", facts[0]["shared_with"]
        else:
            _sum_scope, _sum_shared = "personal", None

        # Collect entities from ALL source facts (already in the SELECT)
        all_entities = set()
        raw_entities = set()
        for f in facts:
            cej = f["canonical_entities_json"]
            if cej:
                try:
                    all_entities.update(json.loads(cej))
                except (json.JSONDecodeError, TypeError):
                    pass

        # P0-3 (dedup-complete-01): apply the SAME content-idempotency invariant
        # as storage.database.store_fact — but on THIS cursor so it stays inside
        # the cluster SAVEPOINT. Previously this raw INSERT bypassed dedup, so a
        # consolidated summary identical to an existing live fact created a
        # duplicate row and never reinforced evidence. Now: reinforce-or-insert.
        # (Excludes 'archived' = soft-deleted, mirroring store_fact.)
        _existing = c.execute(
            "SELECT fact_id FROM atomic_facts "
            "WHERE profile_id = ? AND content = ? "
            "AND lifecycle IN ('active', 'warm', 'cold') "
            "ORDER BY created_at LIMIT 1",
            (profile_id, summary),
        ).fetchone()
        if _existing:
            new_fact_id = _existing["fact_id"]
            c.execute(
                "UPDATE atomic_facts "
                "SET evidence_count = evidence_count + ?, access_count = access_count + 1 "
                "WHERE fact_id = ?",
                (len(facts), new_fact_id),
            )
        else:
            c.execute("""
                INSERT INTO atomic_facts
                (fact_id, memory_id, profile_id, content, fact_type,
                 entities_json, canonical_entities_json,
                 confidence, importance, evidence_count, access_count,
                 created_at, lifecycle, scope, shared_with)
                VALUES (?, '', ?, ?, 'semantic', ?, ?, ?, 0.8, ?, 0, ?, 'active', ?, ?)
            """, (
                new_fact_id, profile_id, summary,
                json.dumps(list(all_entities)),
                json.dumps(list(all_entities)),
                round(avg_confidence, 3), len(facts), now,
                _sum_scope, _sum_shared,
            ))

        # Record the consolidation
        consolidation_id = uuid.uuid4().hex[:16]
        c.execute("""
            INSERT INTO fact_consolidations
            (consolidation_id, profile_id, consolidated_fact_id,
             source_fact_ids, strategy, created_at)
            VALUES (?, ?, ?, ?, 'entity_cluster', ?)
        """, (consolidation_id, profile_id, new_fact_id,
              json.dumps(fact_ids), now))

        # Archive the original facts (NEVER delete) through the canonical
        # lifecycle writer so missing retention rows are created too.
        from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone
        set_fact_lifecycle_zone(
            conn, fact_ids, "archive", profile_id=profile_id,
        )

        # P1-4 (graph-integrity-01): archived facts must stop influencing
        # graph-based ranking. The association_edges FK is ON DELETE CASCADE
        # only (no ON UPDATE), so archiving via UPDATE leaves orphaned edges
        # that spreading_activation still reads. Remove edges touching the
        # archived facts, and set their retention zone so ForgettingFilter
        # excludes them. Inside the SAVEPOINT for atomicity.
        c.execute(
            f"DELETE FROM association_edges "
            f"WHERE profile_id = ? "
            f"AND (source_fact_id IN ({placeholders}) "
            f"     OR target_fact_id IN ({placeholders}))",
            (profile_id, *fact_ids, *fact_ids),
        )
        c.execute(f"RELEASE SAVEPOINT {savepoint_name}")

    except Exception:
        c.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        raise

    logger.info(
        "Consolidated %d facts about '%s' → %s (%d chars)",
        len(facts), entity_name, new_fact_id[:8], len(summary),
    )

    return {"entity": entity_name, "facts": len(facts), "new_fact_id": new_fact_id}


def _generate_summary(
    entity_name: str,
    facts: list,
    config: object | None = None,
) -> str | None:
    """Generate a consolidated summary based on the user's configured mode.

    All modes cap output at _MAX_CONSOLIDATED_CHARS.
    """
    mode = "a"
    if config:
        m = getattr(config, 'mode', None)
        if m:
            mode = getattr(m, 'value', str(m)).lower()

    result = None

    if mode == "a":
        result = _summarize_extractive(entity_name, facts)
    elif mode == "b":
        result = _summarize_with_ollama(entity_name, facts, config)
        if not result:
            result = _summarize_extractive(entity_name, facts)
    elif mode == "c":
        result = _summarize_with_cloud_llm(entity_name, facts, config)
        if not result:
            result = _summarize_with_ollama(entity_name, facts, config)
        if not result:
            result = _summarize_extractive(entity_name, facts)
    else:
        result = _summarize_extractive(entity_name, facts)

    # Uniform cap across all modes
    if result and len(result) > _MAX_CONSOLIDATED_CHARS:
        result = result[:_MAX_CONSOLIDATED_CHARS - 3] + "..."

    return result


def _summarize_with_ollama(
    entity_name: str,
    facts: list,
    config: object | None = None,
) -> str | None:
    """Mode B: Summarize using local Ollama LLM."""
    try:
        import urllib.request

        api_base = "http://localhost:11434"
        model = "llama3.2"
        timeout = 30

        if config and hasattr(config, 'llm'):
            api_base = getattr(config.llm, 'api_base', api_base) or api_base
            model = getattr(config.llm, 'model', model) or model
            # v3.6.12 (modeb-4): the LLMConfig field is `timeout_seconds`, not
            # `timeout` — the old read always missed and silently used 30s.
            timeout = getattr(config.llm, 'timeout_seconds', None) or \
                getattr(config.llm, 'timeout', None) or timeout

        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:_MAX_CLUSTER_SIZE])
        prompt = (
            f"Merge these {len(facts)} facts about '{entity_name}' into ONE concise "
            f"summary paragraph. Keep all key information. Maximum 500 words. "
            f"No preamble.\n\nFacts:\n{fact_texts}"
        )

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 600},
        }).encode()

        req = urllib.request.Request(
            f"{api_base}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json.loads(resp.read().decode())
        text = result.get("response", "").strip()
        return text if text and len(text) > 50 else None
    except Exception as exc:
        logger.warning("Ollama summarization failed: %s", exc)
        return None


def _summarize_with_cloud_llm(
    entity_name: str,
    facts: list,
    config: object | None = None,
) -> str | None:
    """Mode C: Summarize using the user's configured cloud LLM provider."""
    if not config or not hasattr(config, 'llm'):
        return None

    llm_config = config.llm
    provider = getattr(llm_config, 'provider', '')
    if not provider:
        return None

    try:
        from superlocalmemory.llm.backbone import LLMBackbone
        llm = LLMBackbone(llm_config)
        if not llm.is_available():
            return None

        fact_texts = "\n".join(f"- {f['content']}" for f in facts[:_MAX_CLUSTER_SIZE])
        prompt = (
            f"Merge these {len(facts)} facts about '{entity_name}' into ONE concise "
            f"summary paragraph. Keep all key information. Maximum 500 words. "
            f"No preamble.\n\nFacts:\n{fact_texts}"
        )

        response = llm.generate(
            prompt=prompt,
            system="You are a precise fact summarizer. Output only the merged summary.",
            max_tokens=600,
            temperature=0.1,
        )
        text = response.strip() if response else None
        return text if text and len(text) > 50 else None
    except Exception as exc:
        logger.warning("Cloud LLM summarization failed: %s", exc)
        return None


def _summarize_extractive(entity_name: str, facts: list) -> str:
    """Extractive summary — all sentences from all facts, deduped.

    Includes ALL sentences from each fact (not just the first one)
    to preserve complete information.
    """
    header = f"{entity_name}: "
    seen = set()
    sentences = []

    for f in facts:
        content = f["content"]
        # Split on sentence boundaries and include ALL sentences
        raw_sentences = [s.strip() for s in content.split(". ") if s.strip()]
        for sent in raw_sentences:
            if not sent.endswith("."):
                sent += "."
            normalized = sent.lower()
            if normalized not in seen:
                seen.add(normalized)
                sentences.append(sent)

    body = " ".join(sentences)
    result = header + body
    if len(result) > _MAX_CONSOLIDATED_CHARS:
        result = result[:_MAX_CONSOLIDATED_CHARS - 3] + "..."
    return result
