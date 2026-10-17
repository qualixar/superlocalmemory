# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Exclude legacy live-but-marked damage from recall without rewriting storage.

New extraction failures are quarantined before commit. Older live facts must
also stay out of answers when the source check finds damage; they remain in
the read-only fidelity review listing. Explicit user releases are respected.
Kept results retain their order, scores and evidence. No model is called.
"""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.core.source_fidelity_guard import FLAGGED_PREFIX, RELEASED_PREFIX
from superlocalmemory.encoding.source_fidelity import check_fact_against_source

logger = logging.getLogger(__name__)


def _unverified(db: Any, profile_id: str, fact_ids: list[str]) -> dict[str, str]:
    marks = ",".join("?" for _ in fact_ids)
    rows = list(db.execute(
        "SELECT f.fact_id, f.profile_id, f.content, f.observation_date, "
        "m.content AS source, m.profile_id AS source_profile, m.speaker, m.role "
        "FROM atomic_facts f LEFT JOIN memories m ON m.memory_id = f.memory_id "
        f"WHERE f.fact_id IN ({marks})",
        tuple(fact_ids),
    ))
    if {str(dict(r)["fact_id"]) for r in rows} != set(fact_ids):
        raise LookupError("candidate identity unavailable")
    if any(not dict(r)["source"] or dict(r)["source_profile"] != dict(r)["profile_id"]
           for r in rows):
        raise LookupError("owning source unavailable")
    # Only a release recorded under the fact's own profile counts (checked
    # below), so naming those profiles lets the (profile, type, id) index
    # answer instead of a scan of the whole lineage table.
    profiles = sorted({str(dict(r)["profile_id"]) for r in rows})
    pmarks = ",".join("?" for _ in profiles)
    released = {
        (str(dict(r)["profile_id"]), str(dict(r)["object_id"])) for r in db.execute(
            "SELECT profile_id,object_id FROM derivation_lineage "
            f"WHERE profile_id IN ({pmarks}) AND object_type = 'fact' "
            f"AND object_id IN ({marks}) AND unresolved_reason LIKE ?",
            (*profiles, *fact_ids, RELEASED_PREFIX + "%"),
        )
    }
    released |= _applied_corrections(db, profiles, fact_ids)
    flags: dict[str, str] = {}
    for row in rows:
        data = dict(row)
        fact_id, fact, source = str(data["fact_id"]), data["content"] or "", data["source"] or ""
        if (str(data["profile_id"]), fact_id) in released or not fact or not source:
            continue
        from superlocalmemory.encoding.speaker_spans import with_actor

        report = check_fact_against_source(
            fact, with_actor(source, data["speaker"] or "", data["role"] or "user"),
            reference_date=data["observation_date"],
        )
        if not report.ok:
            flags[fact_id] = FLAGGED_PREFIX + "+".join(report.reasons)
    return flags


def _applied_corrections(db: Any, profiles: list[str], fact_ids: list[str]) -> set:
    """(profile, fact) for each candidate that is an applied correction.

    A correction keeps the memory it corrects, so its source is the text it
    replaced: checked against it, the new text always "changed what was said"
    and the edit was withheld from every recall (4.2.10). An applied
    correction is the person's own say-so, like a release. A store without
    the correction ledger has none.
    """
    pmarks = ",".join("?" for _ in profiles)
    marks = ",".join("?" for _ in fact_ids)
    try:
        rows = db.execute(
            "SELECT profile_id, successor_fact_id FROM correction_cases "
            f"WHERE profile_id IN ({pmarks}) AND successor_fact_id IN ({marks}) "
            "AND status = 'applied'",
            (*profiles, *fact_ids),
        )
    except Exception as exc:  # noqa: BLE001 - only a missing ledger is "none"
        if "no such table" in str(exc).lower():
            return set()
        raise
    return {(str(dict(r)["profile_id"]), str(dict(r)["successor_fact_id"])) for r in rows}


def _source_results(results: list, db: Any) -> list:
    """Keep only candidates mechanically verified against immutable memories.

    Read the source table independently of facts/lineage/checker state. If it
    too is unavailable, none of the candidates can safely be called a source.
    Candidate visibility was already authorized by the retrieval engine;
    matching the owning profile also covers explicitly included shared facts.
    """
    from superlocalmemory.core.source_excerpt import is_source_excerpt

    memory_ids = sorted({r.fact.memory_id for r in results
                         if getattr(r, "fact", None) is not None and r.fact.memory_id})
    if not memory_ids:
        return []
    marks = ",".join("?" for _ in memory_ids)
    try:
        rows = db.execute(
            f"SELECT memory_id,profile_id,content FROM memories WHERE memory_id IN ({marks})",
            tuple(memory_ids),
        )
        sources = {(row["profile_id"], row["memory_id"]): row["content"] for row in rows}
    except Exception:  # noqa: BLE001 - unreadable sources cannot establish fidelity
        return []
    kept = []
    for result in results:
        fact = getattr(result, "fact", None)
        if fact is None:
            continue
        source = sources.get((fact.profile_id, fact.memory_id))
        if source and (fact.content == source or is_source_excerpt(fact.content, source)):
            kept.append(result)
    return kept


def flag_unverified(results: list, db: Any, profile_id: str, *, response: Any = None) -> list:
    """Omit damaged facts; on failure retain only verifiable source projections.

    ``response`` receives a per-recall status for every transport, including
    when withholding leaves no results. Direct list callers also get a warning
    on degraded fidelity. A missing store is unavailable, not a failed check.
    Availability depends on the read interface, not a filesystem path (an
    in-memory store can check sources too). Neither state admits unchecked
    derivatives. No source text or exception message is logged.
    """
    status = "checked"
    try:
        fact_ids = [r.fact.fact_id for r in results if getattr(r, "fact", None) is not None]
        if not fact_ids:
            status = "not_checked"
            return results
        if not callable(getattr(db, "execute", None)):
            status = "unavailable"
            logger.warning("No source store available; unchecked derivatives withheld")
            return []
        flags = _unverified(db, profile_id, fact_ids)
    except Exception as exc:  # noqa: BLE001 - withhold unchecked derivatives
        status = "degraded"
        logger.warning("Source fidelity check failed (%s); unchecked derivatives withheld",
                       type(exc).__name__)
        return _source_results(results, db)
    finally:
        if response is not None:
            response.source_fidelity_status = status
    if not flags:
        return results
    return [r for r in results
            if getattr(getattr(r, "fact", None), "fact_id", "") not in flags]


__all__ = ["flag_unverified"]
