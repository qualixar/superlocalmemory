# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Fact consolidation — writes a DISPLAY summary, never a memory.

Groups warm/cold facts that share an entity and writes one summary per cluster
into ``consolidated_summaries``, a display-only table. Nothing in the retrieval
corpus is created, modified or hidden by this module.

WHAT CHANGED AND WHY IT HAD TO
------------------------------
From v3.6.4 to 4.0.9 this module ended each cluster with a raw
``INSERT INTO atomic_facts`` carrying ``memory_id=''``, ``importance=0.8`` and
``entities_json`` holding *every* entity in the cluster. Three consequences, all
measured on the author's 5,089-fact store:

  * 1,195 model-written rows entered the retrieval corpus, and because each
    carried its whole cluster's entity list they had more entity links than any
    real memory, so the entity channel ranked them first. Asked "what am I
    working on", the store answered "Unfortunately, there is no information
    available about 'Gateway', 'State', 'Bounded', or 'Claude' in the provided
    text." at ranks 1, 2 and 3.
  * The rows had no ``temporal_events`` at all (0 of 1,195), so they won the
    temporal channel through its ``created_at`` recency fallback as well.
  * Their entity clusters made them eligible for consolidation in turn: 353 of
    them are summaries of summaries. A store summarising its own summaries
    drifts away from what the user actually said, one pass at a time.

Bypassing ``DatabaseManager`` also bypassed the constraint that would have
refused the row outright — ``atomic_facts`` declares
``FOREIGN KEY (memory_id) REFERENCES memories(memory_id)`` and ``memories`` has
no ``''`` row, but ``storage/memory_write.py`` sets only ``busy_timeout``, not
``PRAGMA foreign_keys=ON``.

So the boundary this module now keeps is the one v3.6 intended: a summary is a
*view* of memory, shown on the dashboard and to Mode B/C readers, and it is not
a thing recall can return. ``community_summaries`` is the precedent.

CONTRACT
--------
  1. NEVER writes to ``atomic_facts`` — not the summary, not the sources.
  2. NEVER archives the source facts. Archiving them made sense only while a
     retrievable summary stood in for them; a display-only summary does not, so
     archiving would replace ten reachable memories with nothing.
  3. Model output is cleaned (``clean_llm_summary``) and then rejected if it is
     a non-answer (``is_non_answer``) — checked before any write, with the
     reason logged.
  4. Only clusters warm/cold facts; never touches 'active' or 'pinned'.
  5. All writes per cluster wrapped in SAVEPOINT for atomicity.
  6. Entity ID LIKE patterns use JSON-boundary quoting to prevent substring
     false positives.

Modes: A extractive, B Ollama, C cloud LLM with fallbacks down to extractive.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from superlocalmemory.storage.database import DatabaseManager

from superlocalmemory.summaries.base import clean_llm_summary
from superlocalmemory.summaries.non_answer import MIN_USEFUL_CHARS, is_non_answer

logger = logging.getLogger("superlocalmemory.fact_consolidator")

_MAX_CLUSTER_SIZE = 10   # Max facts to merge into one
_MIN_CLUSTER_SIZE = 3    # Need at least 3 related facts to consolidate
_MAX_CONSOLIDATED_CHARS = 2000


def consolidate_facts(
    db_or_path: "Union[DatabaseManager, str, Path]",
    profile_id: str = "default",
    max_clusters: int = 20,
    dry_run: bool = False,
    config: object | None = None,
) -> dict:
    """Find and consolidate clusters of related facts.

    Concurrency fix (v3.8.4 — implements TODO from Fix-A comment):
    When a DatabaseManager is provided the consolidation now uses per-cluster
    short write transactions instead of one long raw_connection() that held
    the write lock across Ollama / Cloud-LLM calls for every cluster after
    the first.  The fix:

      1. Discover clusters with a short memory_read() (no write lock).
      2. For each cluster: fetch fact content (memory_read()), generate the
         summary OUTSIDE any lock (Ollama / Cloud LLM may take 30 s), then
         do the SAVEPOINT write inside a short memory_write() block.

    SAVEPOINT atomicity per cluster is preserved: _consolidate_cluster still
    uses SAVEPOINT / RELEASE / ROLLBACK TO internally.

    Mode behavior:
      - Mode A: Extractive only (no LLM). Always available.
      - Mode B: Ollama LLM summarization. Falls back to extractive if Ollama is
        down OR if the model answers with a non-answer.
      - Mode C: Cloud LLM (user's configured provider), then Ollama, then
        extractive, with the same non-answer rejection at each step.

    Returns stats: consolidated, clusters_found, facts_summarized,
    rejected, errors.
    """
    from superlocalmemory.storage.database import DatabaseManager
    from superlocalmemory.storage.memory_write import memory_read, memory_write

    stats: dict = {
        "clusters_found": 0,
        "consolidated": 0,
        # Renamed from facts_archived. Nothing is archived any more, and a key
        # that keeps reporting a count for an action no longer taken is how a
        # behaviour change hides from whoever reads the numbers.
        "facts_summarized": 0,
        # Clusters whose summary was a non-answer and was refused. Counted
        # separately from `errors`: refusing junk is the guard working, not a
        # failure, but a run where every cluster is refused means the model is
        # misbehaving and that has to be visible.
        "rejected": 0,
        # Clusters whose summary already existed over the same sources, so no
        # model call was made. On a settled store this becomes the whole count.
        "unchanged": 0,
        "facts_archived": 0,  # retained at 0 for callers that still read it
        "errors": 0,
        "error_detail": "",
        "mode": "a",
    }

    if config:
        mode = getattr(config, 'mode', None)
        if mode:
            mode_str = getattr(mode, 'value', str(mode)).lower()
            stats["mode"] = mode_str

    if isinstance(db_or_path, DatabaseManager):
        db_path = db_or_path.db_path
        try:
            # Step 1: discover clusters — short read, no write lock.
            with memory_read(db_path) as rconn:
                rconn.row_factory = sqlite3.Row
                clusters = _find_consolidation_clusters(rconn, profile_id, max_clusters)
            stats["clusters_found"] = len(clusters)

            for entity_id, entity_name, fact_ids in clusters:
                try:
                    # Step 2: load fact content for this cluster (read, no write lock).
                    placeholders = ",".join("?" * len(fact_ids))
                    with memory_read(db_path) as rconn:
                        rconn.row_factory = sqlite3.Row
                        facts = rconn.execute(
                            f"SELECT fact_id, content, confidence, created_at, "
                            f"canonical_entities_json, scope, shared_with "
                            f"FROM atomic_facts "
                            f"WHERE fact_id IN ({placeholders}) ORDER BY created_at",
                            fact_ids,
                        ).fetchall()
                        facts = [dict(f) for f in facts]

                    if len(facts) < _MIN_CLUSTER_SIZE:
                        continue

                    # Step 2b: has this exact cluster already been summarised?
                    #
                    # Necessary because 4.0.10 stopped archiving the sources. The
                    # old code made a cluster ineligible by archiving it; now the
                    # same warm facts are eligible on every maintenance pass, so
                    # without this the summarizer is re-run for an unchanged
                    # cluster every 30 minutes forever. Measured over five
                    # consecutive passes on one cluster before this check:
                    # consolidated_summaries converged at 1, but
                    # fact_consolidations grew 1, 2, 3, 4, 5 -- and in Mode B/C
                    # each of those passes paid for a model call to regenerate
                    # text it already had.
                    #
                    # Checked before generation, not after, because the cost
                    # being avoided IS the generation.
                    if _already_summarised(db_path, profile_id, entity_id, fact_ids):
                        stats["unchanged"] += 1
                        continue

                    # Step 3: generate summary OUTSIDE any write lock.
                    # Ollama (Mode B) or Cloud LLM (Mode C) may take 30 s here.
                    summary, generated_by = _generate_summary(
                        entity_name, facts, config,
                    )
                    if not summary:
                        stats["rejected"] += 1
                        continue

                    if dry_run:
                        stats["consolidated"] += 1
                        stats["facts_summarized"] += len(fact_ids)
                        continue

                    # Step 4: short per-cluster write — hold lock only for SQL.
                    with memory_write(db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        result = _consolidate_cluster(
                            conn, profile_id, entity_id, entity_name,
                            fact_ids, dry_run=False, config=None,
                            _presummary=summary, _generated_by=generated_by,
                        )
                    if result:
                        stats["consolidated"] += 1
                        stats["facts_summarized"] += len(fact_ids)
                    else:
                        stats["rejected"] += 1
                except Exception as exc:
                    logger.warning(
                        "Consolidation failed for %s: %s",
                        entity_name, exc, exc_info=True,
                    )
                    stats["errors"] += 1

            if stats["consolidated"] > 0:
                logger.info(
                    "Fact consolidation: %d display summaries written over "
                    "%d facts, %d clusters refused",
                    stats["consolidated"], stats["facts_summarized"],
                    stats["rejected"],
                )
        except Exception as exc:
            logger.error("Fact consolidation failed: %s", exc, exc_info=True)
            stats["errors"] += 1
            stats["error_detail"] = str(exc)
        return stats

    # Backward-compat: str | Path — open our own connection.
    #
    # Type-checked, not assumed. This branch used to run for ANYTHING that was
    # not a DatabaseManager, stringify it, and hand the result to
    # sqlite3.connect — which creates whatever filename it is given. A test
    # passing a MagicMock therefore had a real 4 KB SQLite file named
    # "<MagicMock id='4422448000'>" written into the repository root, one per
    # test. 42 of them had accumulated, and tests/test_ci_guards/
    # test_no_magicmock_artifacts.py failed after any full-suite run.
    #
    # 4.0.6 is where this started firing: it wired consolidate_facts into
    # run_maintenance, so every caller with a mock config reached this line.
    #
    # Refusing an unusable argument is also right beyond the test symptom —
    # silently creating a database at a nonsense path cannot be what any caller
    # wanted, and it hides the real bug (the caller passed the wrong thing).
    if not isinstance(db_or_path, (str, Path)):
        raise TypeError(
            "consolidate_facts() expects a DatabaseManager, or a str/Path to "
            f"memory.db for backward compatibility; got {type(db_or_path).__name__}. "
            "Passing anything else previously created a database file named after "
            "the object's repr."
        )

    logger.warning(
        "consolidate_facts: passing a db_path is deprecated — pass a "
        "DatabaseManager instead (Fix A backward-compat shim active)"
    )
    conn = sqlite3.connect(str(db_or_path))
    wal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
    if wal_mode and wal_mode[0] != "wal":
        logger.warning("WAL mode not active, got: %s", wal_mode[0])
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row

    try:
        _run_consolidation(conn, profile_id, max_clusters, dry_run, config, stats)
        if not dry_run:
            conn.commit()
    except Exception as exc:
        logger.error("Fact consolidation failed: %s", exc, exc_info=True)
        stats["errors"] += 1
        stats["error_detail"] = str(exc)
    finally:
        conn.close()

    return stats


def _run_consolidation(
    conn: sqlite3.Connection,
    profile_id: str,
    max_clusters: int,
    dry_run: bool,
    config: object | None,
    stats: dict,
) -> None:
    """Core consolidation logic — connection-agnostic inner function."""
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
                stats["facts_summarized"] += len(fact_ids)
            else:
                stats["rejected"] += 1
        except Exception as exc:
            logger.warning(
                "Consolidation failed for %s: %s",
                entity_name, exc, exc_info=True,
            )
            stats["errors"] += 1

    if stats["consolidated"] > 0:
        logger.info(
            "Fact consolidation: %d display summaries written over %d facts",
            stats["consolidated"], stats["facts_summarized"],
        )


def _already_summarised(
    db_path: "str | Path",
    profile_id: str,
    entity_id: str,
    fact_ids: list[str],
) -> bool:
    """Whether this exact cluster already has a display summary.

    Compares SORTED source ids, because cluster order comes from an
    ``ORDER BY confidence DESC, created_at DESC`` that two equal-confidence
    facts can swap between passes -- comparing the raw lists would report
    "changed" on a cluster that did not.

    Any error means "not summarised", which costs one redundant generation and
    never skips work that was needed. Read-only.
    """
    from superlocalmemory.storage.memory_write import memory_read

    wanted = sorted(fact_ids)
    try:
        with memory_read(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT source_fact_ids FROM consolidated_summaries "
                " WHERE profile_id = ? AND entity_id = ?",
                (profile_id, entity_id),
            ).fetchall()
    except Exception as exc:  # noqa: BLE001 -- never block consolidation
        logger.debug("cluster-unchanged check skipped: %s", exc)
        return False

    for row in rows:
        try:
            if sorted(json.loads(row["source_fact_ids"] or "[]")) == wanted:
                return True
        except (json.JSONDecodeError, TypeError):
            continue
    return False


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
    *,
    _presummary: str | None = None,
    _generated_by: str = "extractive",
) -> dict | None:
    """Write one display summary for a cluster of facts.

    Touches ``consolidated_summaries`` and ``fact_consolidations`` and nothing
    else. In particular it does not write, update or archive any row in
    ``atomic_facts`` — see the module docstring.

    All writes are wrapped in a SAVEPOINT for atomicity — if any step fails,
    the entire cluster consolidation is rolled back.

    _presummary (v3.8.4): when the caller supplies a pre-computed summary
    (generated OUTSIDE the write lock), skip the _generate_summary() call.
    This is the short-lock path used by the DatabaseManager branch of
    consolidate_facts().  The str | Path backward-compat path still calls
    _generate_summary() inline (legacy behaviour, no regression).

    _generated_by records which mode produced the text ('extractive',
    'ollama', 'cloud'), so a reader can tell a local extractive digest from
    model prose without inspecting the words.
    """
    c = conn.cursor()

    # Load fact contents including canonical_entities_json.
    # Even when _presummary is provided we still re-read from the DB inside
    # the write lock so the SAVEPOINT has a fresh, authoritative facts list.
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

    if _presummary is not None:
        summary = _presummary
    else:
        # Legacy path (str | Path caller) — may call Ollama with write lock held.
        summary, _generated_by = _generate_summary(entity_name, facts, config)
    if not summary:
        return None

    # Last line of defence, on BOTH paths. _generate_summary already cleans and
    # vets its own output, but this function is reachable with a caller-supplied
    # _presummary, and "the caller checked" is exactly the assumption that let
    # 1,195 non-answers into the store.
    #
    # Cleaning runs here too, not just judging. A first draft of this guard
    # only judged, and a test supplying "Sure! Here is a concise summary:
    # <real content>" as a _presummary stored the scaffolding verbatim -- the
    # text passed the non-answer check because it does contain a real summary,
    # and nothing had stripped the two sentences in front of it.
    # clean_llm_summary is idempotent, so running it on both paths is free.
    summary = clean_llm_summary(summary)
    _rejected, _why = is_non_answer(summary, min_chars=MIN_USEFUL_CHARS)
    if _rejected:
        logger.info(
            "Consolidation summary for '%s' rejected before write (%s)",
            entity_name, _why,
        )
        return None

    if dry_run:
        return {"entity": entity_name, "facts": len(facts), "summary_len": len(summary)}

    # Use SAVEPOINT for atomic multi-step write
    savepoint_name = f"consolidate_{uuid.uuid4().hex[:8]}"
    c.execute(f"SAVEPOINT {savepoint_name}")

    try:
        new_fact_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()

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

        # The cluster's pooled entity list is deliberately NOT carried onto the
        # summary. Pooling ten facts' entities gave the old row more entity
        # links than any single real memory, which is precisely how these rows
        # came to out-rank the user's own words in the entity channel. The
        # summary is identified by the one entity that seeded the cluster.

        # The summary goes in the DISPLAY table. Not atomic_facts — see the
        # module docstring for the 1,195 rows that taught us why.
        #
        # UNIQUE (profile_id, entity_id, content) makes a repeated pass
        # idempotent: re-summarising an unchanged cluster refreshes the
        # coverage window instead of accumulating a near-duplicate every time
        # maintenance runs. The old code's reinforce-or-insert dance against
        # atomic_facts existed for the same reason and is no longer needed.
        _dates = [f["created_at"] for f in facts if f["created_at"]]
        # A repeated pass rewrites the coverage of an existing summary: keep the
        # row as it was first (WS-6, versions first), in this savepoint.
        from superlocalmemory.compiler import versions

        versions.keep_previous(
            c, "consolidated_summary",
            {"profile_id": profile_id, "entity_id": entity_id, "content": summary})
        c.execute("""
            INSERT INTO consolidated_summaries
            (summary_id, profile_id, entity_id, entity_name, content,
             source_fact_ids, source_count, char_count, generated_by,
             scope, shared_with, source_earliest, source_latest, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (profile_id, entity_id, content) DO UPDATE SET
                source_fact_ids = excluded.source_fact_ids,
                source_count    = excluded.source_count,
                source_earliest = excluded.source_earliest,
                source_latest   = excluded.source_latest,
                created_at      = excluded.created_at
        """, (
            new_fact_id, profile_id, entity_id, entity_name, summary,
            json.dumps(fact_ids), len(facts), len(summary),
            _generated_by, _sum_scope, _sum_shared,
            min(_dates) if _dates else None,
            max(_dates) if _dates else None,
            now,
        ))

        # Provenance ledger, kept for the repair pass and for auditability.
        # `strategy` distinguishes a display summary from the retrieval-corpus
        # rows the old path wrote, so 'entity_cluster' remains an exact
        # selector for what has to be quarantined.
        # Deterministic id + INSERT OR IGNORE, so re-running over the same
        # cluster does not append a row. fact_consolidations carries no unique
        # constraint, so a random uuid grew the ledger by one row per
        # maintenance pass per cluster with nothing to show for it -- roughly a
        # thousand rows a day on a store with twenty clusters.
        consolidation_id = hashlib.sha256(
            "\0".join((
                profile_id, new_fact_id, "display_summary",
                *sorted(fact_ids),
            )).encode("utf-8")
        ).hexdigest()[:16]
        c.execute("""
            INSERT OR IGNORE INTO fact_consolidations
            (consolidation_id, profile_id, consolidated_fact_id,
             source_fact_ids, strategy, created_at)
            VALUES (?, ?, ?, ?, 'display_summary', ?)
        """, (consolidation_id, profile_id, new_fact_id,
              json.dumps(fact_ids), now))

        # NO archiving, and NO association_edge deletion.
        #
        # Both were correct while a retrievable summary replaced the facts it
        # merged. A display-only summary replaces nothing, so archiving the
        # sources would take ten reachable memories out of recall and put
        # nothing in their place. Measured cost of the old behaviour on the
        # author's store: 528 genuine memories sitting in retention zone
        # 'archive', every one of them put there by this function and by
        # nothing else, all unreachable in normal recall.
        #
        # This module is now purely additive to the store. A test asserts it:
        # tests/test_core/test_consolidation_writes_no_memories.py
        c.execute(f"RELEASE SAVEPOINT {savepoint_name}")

    except Exception:
        c.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        raise

    logger.info(
        "Display summary for '%s' from %d facts → %s (%d chars, %s)",
        entity_name, len(facts), new_fact_id[:8], len(summary), _generated_by,
    )

    return {"entity": entity_name, "facts": len(facts), "new_fact_id": new_fact_id}


def _generate_summary(
    entity_name: str,
    facts: list,
    config: object | None = None,
) -> tuple[str | None, str]:
    """Generate a display summary for the user's configured mode.

    Returns ``(summary_or_None, generated_by)`` where ``generated_by`` is
    'extractive', 'ollama' or 'cloud'. It used to return the text alone, which
    left no way to tell a local digest from model prose after the fact — and
    since only the model paths can produce a non-answer, that distinction is
    what makes a bad batch traceable to its source.

    Model output goes through two stages before it is offered to the caller:

      1. ``clean_llm_summary`` strips chat scaffolding *around* the answer.
         This stripper has existed in ``summaries/base.py`` since 3.6 and this
         module referenced it zero times, which is why 20 of the author's
         stored summaries begin "Here is a concise summary paragraph".
      2. ``is_non_answer`` rejects text that is scaffolding *all the way
         through* — a refusal, a request for more input, a report that the
         input was empty. Order matters: step 1 rescues "Here is a summary:
         <real content>", and running step 2 first would discard it.

    A rejected model summary falls back to extractive, which is derived
    mechanically from the facts and so cannot refuse. Falling back beats
    writing nothing: the user still gets a digest.

    All modes cap output at _MAX_CONSOLIDATED_CHARS.
    """
    mode = "a"
    if config:
        m = getattr(config, 'mode', None)
        if m:
            mode = getattr(m, 'value', str(m)).lower()

    def _vet(
        text: str | None, source: str, *, model_written: bool = True,
    ) -> str | None:
        """Clean, then reject a non-answer. Returns None if unusable."""
        if not text:
            return None
        cleaned = clean_llm_summary(text)
        rejected, why = is_non_answer(
            cleaned, min_chars=MIN_USEFUL_CHARS, model_written=model_written,
        )
        if rejected:
            logger.info(
                "%s summary for '%s' discarded (%s); falling back",
                source, entity_name, why,
            )
            return None
        return cleaned

    result: str | None = None
    generated_by = "extractive"

    if mode == "b":
        result = _vet(_summarize_with_ollama(entity_name, facts, config), "Ollama")
        if result:
            generated_by = "ollama"
    elif mode == "c":
        # The configured provider only: no Ollama-format fallback to the cloud host.
        result = _vet(_summarize_with_cloud_llm(entity_name, facts, config), "Cloud LLM")
        if result:
            generated_by = "cloud"

    if not result:
        # Extractive is assembled from the facts' own sentences, so it has no
        # opinion to refuse with — and applying the refusal rules to it would
        # reject a summary because the owner's own memory happened to contain a
        # phrase like "the facts provided". Still checked for the two things
        # that do apply: length, and tool-call markup, which a cluster of
        # markup-carrying facts would otherwise reassemble verbatim.
        result = _vet(
            _summarize_extractive(entity_name, facts), "Extractive",
            model_written=False,
        )
        generated_by = "extractive"

    # Uniform cap across all modes
    if result and len(result) > _MAX_CONSOLIDATED_CHARS:
        result = result[:_MAX_CONSOLIDATED_CHARS - 3] + "..."

    return result, generated_by


def _summarize_with_ollama(
    entity_name: str,
    facts: list,
    config: object | None = None,
) -> str | None:
    """Mode B: Summarize using the configured Ollama (through the outbound gate)."""
    try:
        import urllib.request

        from superlocalmemory.core import outbound_http

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
        resp = outbound_http.urlopen(req, timeout=timeout)
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
