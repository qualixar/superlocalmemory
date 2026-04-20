# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — F4.A Stage-8 H-01 fix

"""Behavioural pattern mining for the consolidation worker.

Analyses atomic_facts + signals + co-retrieval + entities to produce
pattern rows consumed by the dashboard, soft-prompt generator, and
recall ranker. Eight families: tech_preference, interest, temporal,
entity_preferences, session_activity, fact_type_distribution,
channel_performance + co_retrieval_clusters, knowledge_structure.

Contract refs: LLD-10 §2, Stage 8 H-01 (file split).
"""

from __future__ import annotations

import json as _json
import logging
import re
import sqlite3
from collections import Counter

logger = logging.getLogger(__name__)

__all__ = ("generate_patterns",)


# Keyword + stopword dictionaries live in a sibling constants module
# so this file stays within the 400-LOC per-file cap.
from superlocalmemory.learning.pattern_miner_constants import (
    TECH_KEYWORDS as _TECH_KEYWORDS,
    STOPWORDS as _STOPWORDS,
)


def generate_patterns(
    memory_db: str,
    learning_db: str,
    profile_id: str,
    dry_run: bool,
) -> int:
    """Mine behavioural patterns and record them in BehavioralPatternStore.

    Returns the count of patterns generated. Safe to call repeatedly —
    the store upserts by ``(profile_id, pattern_type, pattern_key)``.
    """
    try:
        from superlocalmemory.learning.behavioral import BehavioralPatternStore

        conn = sqlite3.connect(memory_db, timeout=10)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row

        facts = conn.execute(
            "SELECT fact_id, content, fact_type, created_at, session_id, "
            "confidence, canonical_entities_json "
            "FROM atomic_facts "
            "WHERE profile_id = ? AND lifecycle = 'active' "
            "ORDER BY created_at DESC",
            (profile_id,),
        ).fetchall()

        if len(facts) < 5:
            conn.close()
            return 0

        store = BehavioralPatternStore(learning_db)
        generated = 0

        generated += _mine_tech_preferences(store, facts, profile_id, dry_run)
        generated += _mine_topic_interests(store, facts, profile_id, dry_run)
        generated += _mine_temporal(store, facts, profile_id, dry_run)
        generated += _mine_entity_preferences(
            store, conn, facts, profile_id, dry_run,
        )
        generated += _mine_session_activity(store, facts, profile_id, dry_run)
        generated += _mine_fact_type_distribution(
            store, facts, profile_id, dry_run,
        )
        generated += _mine_channel_and_coretrieval(
            store, learning_db, profile_id, dry_run,
        )
        generated += _mine_knowledge_structure(
            store, conn, profile_id, dry_run,
        )

        conn.close()

        logger.info(
            "Pattern mining: %d patterns generated for profile %s "
            "from %d facts",
            generated, profile_id, len(facts),
        )
        return generated
    except Exception as exc:
        logger.warning("Pattern generation error: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Family miners — each returns its contribution to the generated count.
# ---------------------------------------------------------------------------


def _mine_tech_preferences(store, facts, profile_id, dry_run) -> int:
    tech_counts: Counter = Counter()
    for f in facts:
        content = dict(f)["content"].lower()
        for keyword, label in _TECH_KEYWORDS.items():
            if keyword in content:
                tech_counts[label] += 1

    gen = 0
    for tech, count in tech_counts.most_common(20):
        if count >= 2 and not dry_run:
            confidence = min(1.0, count / max(len(facts) * 0.1, 10))
            store.record_pattern(
                profile_id=profile_id,
                pattern_type="tech_preference",
                data={"topic": tech, "pattern_key": tech,
                      "value": tech, "key": "tech",
                      "evidence": count},
                success_rate=confidence,
                confidence=confidence,
            )
            gen += 1
    return gen


def _mine_topic_interests(store, facts, profile_id, dry_run) -> int:
    word_counts: Counter = Counter()
    for f in facts:
        words = re.findall(r'\b[a-zA-Z]{4,}\b', dict(f)["content"].lower())
        for w in words:
            if w not in _STOPWORDS:
                word_counts[w] += 1

    gen = 0
    for topic, count in word_counts.most_common(15):
        if count >= 3 and not dry_run:
            confidence = min(1.0, count / max(len(facts) * 0.05, 15))
            store.record_pattern(
                profile_id=profile_id,
                pattern_type="interest",
                data={"topic": topic, "pattern_key": topic,
                      "count": count, "evidence": count},
                success_rate=confidence,
                confidence=confidence,
            )
            gen += 1
    return gen


def _mine_temporal(store, facts, profile_id, dry_run) -> int:
    hour_counts: Counter = Counter()
    for f in facts:
        created = dict(f).get("created_at", "")
        try:
            if "T" in created:
                hour = int(created.split("T")[1][:2])
            elif " " in created:
                hour = int(created.split(" ")[1][:2])
            else:
                continue
            period = (
                "morning" if 6 <= hour < 12 else
                "afternoon" if 12 <= hour < 18 else
                "evening" if 18 <= hour < 22 else "night"
            )
            hour_counts[period] += 1
        except (ValueError, IndexError):
            pass

    gen = 0
    total_hours = sum(hour_counts.values())
    for period, count in hour_counts.most_common():
        if count >= 2 and total_hours > 0 and not dry_run:
            pct = round(count / total_hours * 100)
            store.record_pattern(
                profile_id=profile_id,
                pattern_type="temporal",
                data={"topic": period, "pattern_key": period,
                      "value": f"{period} ({pct}%)",
                      "evidence": count, "key": period,
                      "distribution": dict(hour_counts)},
                success_rate=pct / 100,
                confidence=min(1.0, count / max(total_hours * 0.1, 5)),
            )
            gen += 1
    return gen


def _mine_entity_preferences(
    store, conn: sqlite3.Connection, facts, profile_id, dry_run,
) -> int:
    entity_counts: Counter = Counter()
    for f in facts:
        raw = dict(f).get("canonical_entities_json", "")
        if raw:
            try:
                for ent in _json.loads(raw):
                    entity_counts[ent] += 1
            except (ValueError, TypeError):
                pass

    # v3.4.7: Resolve entity IDs to readable canonical names
    entity_names: dict = {}
    try:
        eid_list = list(entity_counts.keys())
        if eid_list:
            placeholders = ",".join("?" * len(eid_list))
            name_rows = conn.execute(
                f"SELECT entity_id, canonical_name FROM canonical_entities "
                f"WHERE entity_id IN ({placeholders})",
                eid_list,
            ).fetchall()
            entity_names = {
                dict(r)["entity_id"]: dict(r)["canonical_name"]
                for r in name_rows
            }
    except Exception:
        pass

    gen = 0
    # S9-DASH-01: skip entity IDs that don't resolve to a canonical name.
    # Previously the fallback ``entity_names.get(entity, entity)`` leaked
    # raw hex entity_ids (e.g. ``ea701bf01f1ff4df8``) into the dashboard
    # as "preferences". We now (1) require a resolved readable name and
    # (2) defensively drop pure-hex strings that somehow slip through
    # (16-17 hex chars — the canonical_entities primary-key shape).
    _HEX_ID_RE = re.compile(r"^[0-9a-f]{15,20}$")
    for entity, count in entity_counts.most_common(15):
        if count < 3 or dry_run:
            continue
        readable = entity_names.get(entity)
        if not readable:
            logger.debug(
                "_mine_entity_preferences: skipping orphan entity_id=%r "
                "(count=%d) — no row in canonical_entities",
                entity, count,
            )
            continue
        if _HEX_ID_RE.match(readable):
            logger.debug(
                "_mine_entity_preferences: skipping hex-shaped name=%r "
                "— likely an entity_id stored as name", readable,
            )
            continue
        confidence = min(1.0, count / max(len(facts) * 0.05, 10))
        store.record_pattern(
            profile_id=profile_id,
            pattern_type="entity_preferences",
            data={"topic": readable,
                  "pattern_key": f"entity:{readable}",
                  "value": readable, "evidence": count,
                  "source": "entity_frequency"},
            success_rate=confidence,
            confidence=confidence,
        )
        gen += 1
    return gen


def _mine_session_activity(store, facts, profile_id, dry_run) -> int:
    session_counts: Counter = Counter()
    for f in facts:
        sid = dict(f).get("session_id", "")
        if sid:
            session_counts[sid] += 1

    if not session_counts:
        return 0

    avg_facts_per_session = sum(session_counts.values()) / len(session_counts)
    heavy_sessions = [
        s for s, c in session_counts.items()
        if c > avg_facts_per_session * 2
    ]
    if heavy_sessions and not dry_run:
        store.record_pattern(
            profile_id=profile_id,
            pattern_type="session_activity",
            data={"pattern_key": "heavy_session_usage",
                  "value": f"{len(heavy_sessions)} intensive sessions",
                  "evidence": len(heavy_sessions),
                  "avg_facts": round(avg_facts_per_session, 1),
                  "total_sessions": len(session_counts)},
            success_rate=0.8,
            confidence=min(1.0, len(heavy_sessions) / 5),
        )
        return 1
    return 0


def _mine_fact_type_distribution(store, facts, profile_id, dry_run) -> int:
    type_counts: Counter = Counter()
    for f in facts:
        ft = dict(f).get("fact_type", "semantic")
        type_counts[ft] += 1

    total_ft = sum(type_counts.values())
    if total_ft > 0 and not dry_run:
        dominant_type = type_counts.most_common(1)[0]
        pct = round(dominant_type[1] / total_ft * 100)
        store.record_pattern(
            profile_id=profile_id,
            pattern_type="fact_type_distribution",
            data={"pattern_key": "memory_style",
                  "value": f"{dominant_type[0]} dominant ({pct}%)",
                  "evidence": dominant_type[1],
                  "distribution": dict(type_counts)},
            success_rate=pct / 100,
            confidence=min(1.0, dominant_type[1] / 20),
        )
        return 1
    return 0


def _mine_channel_and_coretrieval(
    store, learning_db: str, profile_id: str, dry_run: bool,
) -> int:
    gen = 0
    try:
        learn_conn = sqlite3.connect(learning_db, timeout=10)
        learn_conn.row_factory = sqlite3.Row

        channel_rows = learn_conn.execute(
            "SELECT channel, COUNT(*) AS cnt, "
            "AVG(signal_value) AS avg_signal "
            "FROM learning_feedback "
            "WHERE profile_id = ? "
            "GROUP BY channel ORDER BY cnt DESC",
            (profile_id,),
        ).fetchall()

        for row in channel_rows:
            d = dict(row)
            ch = d.get("channel", "unknown")
            cnt = d.get("cnt", 0)
            avg_sig = round(float(d.get("avg_signal", 0) or 0), 3)
            if cnt >= 5 and not dry_run:
                store.record_pattern(
                    profile_id=profile_id,
                    pattern_type="channel_performance",
                    data={"pattern_key": f"channel:{ch}",
                          "value": f"{ch} ({cnt} hits, {avg_sig} avg)",
                          "evidence": cnt,
                          "avg_signal": avg_sig},
                    success_rate=avg_sig,
                    confidence=min(1.0, cnt / 50),
                )
                gen += 1

        try:
            coret_rows = learn_conn.execute(
                "SELECT fact_a, fact_b, co_access_count "
                "FROM co_retrieval_edges "
                "WHERE profile_id = ? AND co_access_count >= 3 "
                "ORDER BY co_access_count DESC LIMIT 20",
                (profile_id,),
            ).fetchall()
            if coret_rows and not dry_run:
                top_pair = (
                    dict(coret_rows[0]).get("co_access_count", 0)
                    if coret_rows else 0
                )
                store.record_pattern(
                    profile_id=profile_id,
                    pattern_type="co_retrieval_clusters",
                    data={"pattern_key": "co_retrieval_clusters",
                          "value": f"{len(coret_rows)} strong fact pairs",
                          "evidence": len(coret_rows),
                          "top_pair_count": top_pair},
                    success_rate=0.7,
                    confidence=min(1.0, len(coret_rows) / 10),
                )
                gen += 1
        except Exception:
            pass

        learn_conn.close()
    except Exception as exc:
        logger.debug("Signal pattern mining failed: %s", exc)
    return gen


def _mine_knowledge_structure(
    store, conn: sqlite3.Connection, profile_id: str, dry_run: bool,
) -> int:
    try:
        comm_rows = conn.execute(
            "SELECT community_id, COUNT(*) AS cnt "
            "FROM fact_importance "
            "WHERE profile_id = ? AND community_id IS NOT NULL "
            "GROUP BY community_id ORDER BY cnt DESC",
            (profile_id,),
        ).fetchall()
        if comm_rows and not dry_run:
            total_comm = sum(dict(r)["cnt"] for r in comm_rows)
            store.record_pattern(
                profile_id=profile_id,
                pattern_type="knowledge_structure",
                data={"pattern_key": "knowledge_structure",
                      "value": f"{len(comm_rows)} topic communities, "
                               f"{total_comm} classified facts",
                      "evidence": total_comm,
                      "community_count": len(comm_rows)},
                success_rate=0.8,
                confidence=min(1.0, len(comm_rows) / 5),
            )
            return 1
    except Exception:
        pass
    return 0
