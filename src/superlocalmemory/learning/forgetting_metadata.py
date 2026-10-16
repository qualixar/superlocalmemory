# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Canonical inputs to decay, including genuine use and source provenance."""


def columns(db, table):
    return {dict(r)["name"] for r in db.execute(f"PRAGMA table_info({table})")}


def protection_clause(db):
    clauses = []
    if "pinned" in columns(db, "atomic_facts"):
        clauses.append("COALESCE(f.pinned,0)=0")
    if columns(db, "pinned_facts"):
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM pinned_facts p "
            "WHERE p.fact_id=f.fact_id AND p.profile_id=f.profile_id)"
        )
    if {"reward", "settled_at"} <= columns(db, "action_outcomes"):
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM action_outcomes o "
            "WHERE o.profile_id=f.profile_id AND o.reward > .3 "
            "AND julianday(o.settled_at) >= julianday('now','-60 days') "
            "AND EXISTS (SELECT 1 FROM json_each(o.fact_ids_json) j "
            "WHERE j.value=f.fact_id))"
        )
    return "".join(" AND " + c for c in clauses)


def attach_canonical_trust(db, rows, profile_id):
    """Resolve canonical source/direct trust with at most three profile reads."""
    tables = {
        dict(row)["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name IN ('trust_scores','provenance')"
        )
    }
    records, sources = {}, {}
    if "trust_scores" in tables:
        for row in db.execute(
            "SELECT target_type,target_id,trust_score,evidence_count "
            "FROM trust_scores WHERE profile_id=? AND target_type IN ('fact','agent') "
            "ORDER BY rowid",
            (profile_id,),
        ):
            records.setdefault((row["target_type"], row["target_id"]), dict(row))
    if "provenance" in tables:
        for row in db.execute(
            "SELECT fact_id,created_by FROM provenance WHERE profile_id=? ORDER BY rowid",
            (profile_id,),
        ):
            sources.setdefault(row["fact_id"], row["created_by"])
    return [dict(row, trust_score=_fact_trust(row["fact_id"], records, sources)) for row in rows]


def _fact_trust(fact_id, records, sources):
    from superlocalmemory.trust.scorer import _DEFAULT_TRUST, TrustScorer, is_anonymous

    record = records.get(("fact", fact_id))
    if record is None:
        source = sources.get(fact_id, "")
        record = records.get(("agent", source)) if not is_anonymous(source) else None
    if record is None:
        return _DEFAULT_TRUST
    alpha, beta = TrustScorer._decode_beta(record["trust_score"], record["evidence_count"])
    return TrustScorer._compute_trust(alpha, beta)


def protected_fact(db, fact_id, profile_id):
    extra = protection_clause(db)
    if columns(db, "core_memory_blocks"):
        extra += (
            " AND NOT EXISTS (SELECT 1 FROM core_memory_blocks b, "
            "json_each(b.source_fact_ids) j WHERE b.profile_id=f.profile_id "
            "AND j.value=f.fact_id)"
        )
    if columns(db, "ccq_consolidated_blocks"):
        extra += (
            " AND NOT EXISTS (SELECT 1 FROM ccq_consolidated_blocks c, "
            "json_each(c.source_fact_ids) j WHERE c.profile_id=f.profile_id "
            "AND j.value=f.fact_id AND NOT EXISTS (SELECT 1 FROM fact_access_log a "
            "WHERE a.fact_id=f.fact_id AND a.profile_id=f.profile_id "
            "AND a.access_type!='consolidation' "
            "AND julianday(a.accessed_at)>julianday(c.created_at)))"
        )
    return not db.execute(
        "SELECT 1 FROM atomic_facts f WHERE f.fact_id=? AND f.profile_id=?" + extra,
        (fact_id, profile_id),
    )


def metadata_query(db, internal_access, consolidated):
    protection = protection_clause(db)
    sql = (
        "SELECT f.fact_id, f.content AS content, "
        "COALESCE(al.access_count,0) AS access_count, "
        "COALESCE(fi.pagerank_score,0.0) AS importance, "
        "COALESCE(f.evidence_count,0) AS confirmation_count, "
        "COALESCE(f.emotional_valence,0.0) AS emotional_salience, "
        "COALESCE(r.memory_strength,0.0) AS memory_strength, "
        "CASE WHEN julianday(al.latest) > julianday(COALESCE(r.last_accessed_at,f.created_at)) "
        "THEN al.latest ELSE COALESCE(r.last_accessed_at,al.latest,f.created_at) END "
        "AS last_accessed_at FROM atomic_facts f "
        "LEFT JOIN (SELECT fact_id,COUNT(*) AS access_count, "
        "strftime('%Y-%m-%dT%H:%M:%fZ',MAX(julianday(accessed_at))) AS latest "
        "FROM fact_access_log WHERE profile_id=? AND access_type NOT IN "
        + internal_access
        + " GROUP BY fact_id) al ON al.fact_id=f.fact_id "
        "LEFT JOIN fact_importance fi ON fi.fact_id=f.fact_id AND fi.profile_id=? "
        "LEFT JOIN fact_retention r ON r.fact_id=f.fact_id AND r.profile_id=? "
        "WHERE f.profile_id=? AND f.fact_id NOT IN (SELECT j.value "
        "FROM core_memory_blocks b,json_each(b.source_fact_ids) j WHERE b.profile_id=?)"
        + protection
        + consolidated
    )
    return sql
