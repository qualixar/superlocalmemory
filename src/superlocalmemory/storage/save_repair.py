# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Re-file saves whose searchable copy was folded onto another memory (4.1.25 L-06).

THE WRITER THAT CAUSED IT (fixed on release by 15094b73, 2026-10-11)

Before that change ``DatabaseManager.store_fact`` deduplicated a new fact
against ANY live fact in the profile with the same text, ignoring its memory
and its scope. A save's queryable fact is created inside the save's own new
memory; when the same words already existed as a fact of another memory, the
new fact was folded onto the old one and the operation recorded the OLD fact.
Background enrichment then derived the operation's memory from that fact and
found someone else's memory:

* ``... memory scope mismatch``: the other memory had another scope (the 31
  Codex saves of 2026-10-10: personal saves folded onto global memories);
* ``... memory content mismatch``: the other memory's text differs (its fact
  was an extracted sentence equal to the new note);
* ``... fact profile mismatch or missing fact``: the folded-onto fact was
  later deleted.

Each retried ten times and was dead-lettered. The save itself was not lost:
its own memory row, with the right text and scope, exists with no fact, and
the operation (immutable: what was asked) is the authority for it.

THE REPAIR

:func:`plan` reads only. For each such operation it finds the operation's own
memory (``metadata.ingestion_operation_id``) and decides:

* ``add_fact``: own memory matches the operation; give it its searchable fact;
* ``repoint``: own memory already has facts; point the operation at them;
* ``add_memory_and_fact``: no own memory (never written); write both;
* ``restore_then_add_fact`` / ``restore_then_repoint``: own memory is a blanked
  copy of the operation's text (a pre-4.1.19 build replaced spans with
  ``[REDACTED:...]``, #308). Its text, and the text of its facts, is restored
  with the guarded statements of ``store_repair`` (shared, not copied), then
  the fact is added or the operation pointed at the restored facts. The
  one-time store repair only ever looked at saves already failed when it ran,
  so these saves were stuck for good. An original that holds a secret, or that
  the ingest gate rejects, is not restored: it stays in review.
* review: own memory no longer matches the operation (changed after saving):
  left alone, listed.

:func:`apply` does all of it in one transaction, each write guarded by the
state it expects, then resets the operation to ``queryable`` (attempts 0) and
removes its dead-letter rows, so the materializer enriches it as new. The
other memory's fact is not touched. It also clears dead-letter rows whose operation
has since completed, which only record an old failure. The same command backfills a missing agent
name on a memory from its own operation, where the operation recorded one.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field

REASONS = (
    "queryable ingestion memory scope mismatch",
    "queryable ingestion memory content mismatch",
    "queryable ingestion fact profile mismatch or missing fact",
    # The spelling the enrichment step raises today (core/engine_ingestion).
    "queryable fact profile mismatch or missing fact",
    "queryable ingestion shared scope mismatch",
    "queryable ingestion fact belongs to another memory",
)
#: The memory a save wrote: it names the operation, directly or (after an equal
#: save was folded into it, storage/source_identity) in its metadata history.
_OWN_MEMORY = (
    "SELECT memory_id, content, scope, shared_with, metadata_json FROM memories "
    "WHERE profile_id=:p AND (json_extract(metadata_json, '$.ingestion_operation_id')=:o "
    "OR EXISTS (SELECT 1 FROM json_each(metadata_json, "
    "'$.metadata_history.ingestion_operation_id') WHERE value=:o))"
)
#: Failing that, a live memory that is the same source by the identity rule
#: storage/source_identity applies: same profile, text and visibility.
_SAME_SOURCE = (
    "SELECT memory_id, content, scope, shared_with, metadata_json FROM memories "
    "WHERE profile_id=? AND content=? AND scope=?"
)


@dataclass(frozen=True, slots=True)
class RefileItem:
    operation_id: str
    profile_id: str
    #: add_fact | repoint | add_memory_and_fact | restore_then_add_fact | restore_then_repoint
    action: str
    memory_id: str
    reason: str
    fact_ids: tuple[str, ...] = ()
    operation_fingerprint: str = ""
    fact_fingerprint: str = ""
    #: sha256 of the blanked memory text a restore replaces (empty for other actions).
    memory_fingerprint: str = ""


RESTORE_ACTIONS = frozenset({"restore_then_add_fact", "restore_then_repoint"})
REPOINT_ACTIONS = frozenset({"repoint", "restore_then_repoint"})
NOT_RESTORED_SECRET = "original holds what looks like a secret; review it"
NOT_RESTORED_REJECTED = ("original is rejected by the ingest gate "
                         "(prompt template or over 1 MB); review it")
NOT_RESTORED_FACTS = "its searchable facts were changed after saving"
#: 4.2.13 said "its text was changed after saving", but no SLM path rewrites a
#: memory's text after a save (an edit adds a correction fact; #308 review), so
#: that blamed the user for something else. Say what is known and what to do.
NOT_RESTORED_DIFFERENT = ("its stored text differs from the save by more than redaction "
                          "markers; compare the two, and if the save is not wanted run "
                          "`slm ops resolve <operation id> --action cancel`")


@dataclass(frozen=True, slots=True)
class SavePlan:
    refile: tuple[RefileItem, ...]
    review: tuple[tuple[str, str], ...]          # (operation_id, why)
    attribution: tuple[tuple[str, str], ...]     # (memory_id, agent_id from its operation)
    unattributed_by_source: dict = field(default_factory=dict)
    #: Dead-letter rows whose operation has since completed (old failures, not failed saves).
    stale_dead_letters: int = 0

    def summary(self) -> dict:
        actions: dict[str, int] = {}
        for item in self.refile:
            actions[item.action] = actions.get(item.action, 0) + 1
        reasons: dict[str, int] = {}
        for _, why in self.review:
            reasons[why] = reasons.get(why, 0) + 1
        return {"saves_to_refile": len(self.refile), "by_action": actions,
                "left_for_review": len(self.review), "left_for_review_by_reason": reasons,
                "agent_names_to_backfill": len(self.attribution),
                "memories_still_without_agent": dict(self.unattributed_by_source),
                "stale_dead_letters": self.stale_dead_letters}


def _candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    marks = ",".join("?" * len(REASONS))
    return conn.execute(
        "SELECT * FROM ingestion_operations WHERE state='failed' AND ("
        f"last_error IN ({marks}) OR operation_id IN (SELECT original_op_id FROM "
        f"dead_letter_operations WHERE error IN ({marks}))) ORDER BY created_at",
        REASONS + REASONS).fetchall()


STALE_DEAD_LETTERS = (
    "dead_letter_operations WHERE original_op_id IN "
    "(SELECT operation_id FROM ingestion_operations WHERE state='complete')"
)


def count_stale_dead_letters(conn: sqlite3.Connection) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {STALE_DEAD_LETTERS}").fetchone()[0])


def _same_audience(stored: str | None, op_json: str | None) -> bool:
    return set(json.loads(stored or "null") or []) == set(json.loads(op_json or "[]") or [])


def _own_memory(conn: sqlite3.Connection, op: sqlite3.Row) -> list[sqlite3.Row]:
    own = conn.execute(_OWN_MEMORY, {"p": op["profile_id"], "o": op["operation_id"]}).fetchall()
    if own:
        return own
    return [row for row in conn.execute(
        _SAME_SOURCE, (op["profile_id"], op["raw_content"], op["scope"]))
        if _same_audience(row["shared_with"], op["shared_with_json"])][:1]


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _fact_evidence(conn: sqlite3.Connection, op: sqlite3.Row, memory_id: str):
    rows = conn.execute(
        "SELECT fact_id, content, scope, shared_with, lifecycle FROM atomic_facts "
        "WHERE memory_id=? AND profile_id=? AND lifecycle IN ('active','warm','cold') "
        "ORDER BY created_at, fact_id", (memory_id, op["profile_id"])).fetchall()
    if any((row["scope"] or "personal") != op["scope"] or not _same_audience(
            row["shared_with"], op["shared_with_json"]) for row in rows):
        return None
    return tuple(str(row["fact_id"]) for row in rows), _fingerprint([dict(row) for row in rows])


def _classify(conn: sqlite3.Connection, op: sqlite3.Row) -> RefileItem | tuple[str, str]:
    fingerprint = _fingerprint(dict(op))
    own = _own_memory(conn, op)
    if len(own) > 1:
        return op["operation_id"], "more than one memory claims this save"
    if not own:
        return RefileItem(op["operation_id"], op["profile_id"], "add_memory_and_fact",
                          uuid.uuid4().hex, op["last_error"], operation_fingerprint=fingerprint)
    mem = own[0]
    restore = mem["content"] != op["raw_content"]
    if restore:
        refusal = _restore_refusal(mem, op)
        if refusal:
            return op["operation_id"], refusal
    if (mem["scope"] or "personal") != op["scope"] or not _same_audience(
            mem["shared_with"], op["shared_with_json"]):
        return op["operation_id"], "its visibility was changed after saving"
    evidence = _fact_evidence(conn, op, mem["memory_id"])
    if evidence is None:
        return op["operation_id"], "its searchable facts have different visibility"
    facts, fact_fingerprint = evidence
    action = ("restore_then_" if restore else "") + ("repoint" if facts else "add_fact")
    if restore and not _facts_restorable(conn, facts, mem["content"], op["raw_content"]):
        return op["operation_id"], NOT_RESTORED_FACTS
    return RefileItem(op["operation_id"], op["profile_id"], action, mem["memory_id"],
                      op["last_error"], facts, fingerprint, fact_fingerprint,
                      _fingerprint(mem["content"]) if restore else "")


def _restore_refusal(mem: sqlite3.Row, op: sqlite3.Row) -> str:
    """Why a differing memory text is not restored from the operation; empty if it can be."""
    from superlocalmemory.core.ingest_gate import apply_ingest_gate
    from superlocalmemory.core.security_primitives import detect_secrets
    from superlocalmemory.storage.store_repair import is_blanked_copy

    if not is_blanked_copy(mem["content"], op["raw_content"]):
        return NOT_RESTORED_DIFFERENT
    gate = apply_ingest_gate(op["raw_content"])
    if gate.rejected:
        return NOT_RESTORED_REJECTED
    # The gate clamps and rejects but never redacts: restoring a span a 4.1.18
    # build blanked could put a real credential back. Hit => a person decides.
    if detect_secrets(op["raw_content"]):
        return NOT_RESTORED_SECRET
    return ""


def _facts_restorable(conn: sqlite3.Connection, fact_ids: tuple[str, ...], blanked: str,
                      original: str) -> bool:
    """Every live fact of the memory is the gate's form of the blanked or restored text."""
    from superlocalmemory.core.ingest_gate import apply_ingest_gate

    if not fact_ids:
        return True
    allowed = {apply_ingest_gate(blanked).fact_content, apply_ingest_gate(original).fact_content}
    marks = ",".join("?" * len(fact_ids))
    rows = conn.execute(f"SELECT content FROM atomic_facts WHERE fact_id IN ({marks})", fact_ids)
    return all(row[0] in allowed for row in rows)


def _attribution(conn: sqlite3.Connection) -> tuple[list[tuple[str, str]], dict]:
    rows = conn.execute(
        "SELECT m.memory_id, json_extract(o.raw_metadata_json, '$.agent_id') AS agent, "
        "o.source_type FROM memories m LEFT JOIN ingestion_operations o ON "
        "o.operation_id = json_extract(m.metadata_json, '$.ingestion_operation_id') "
        "WHERE COALESCE(json_extract(m.metadata_json, '$.agent_id'), '') = ''").fetchall()
    fill = [(r["memory_id"], str(r["agent"])) for r in rows if r["agent"]]
    unknown: dict[str, int] = {}
    for r in rows:
        if not r["agent"]:
            key = r["source_type"] or "no save record"
            unknown[key] = unknown.get(key, 0) + 1
    return fill, unknown


def plan(conn: sqlite3.Connection) -> SavePlan:
    """What :func:`apply` would change. Reads only."""
    conn.row_factory = sqlite3.Row
    refile: list[RefileItem] = []
    review: list[tuple[str, str]] = []
    for op in _candidates(conn):
        decided = _classify(conn, op)
        (refile if isinstance(decided, RefileItem) else review).append(decided)
    fill, unknown = _attribution(conn)
    return SavePlan(tuple(refile), tuple(review), tuple(fill), unknown,
                    count_stale_dead_letters(conn))


__all__ = ["REASONS", "REPOINT_ACTIONS", "RESTORE_ACTIONS", "RefileItem",
           "STALE_DEAD_LETTERS", "SavePlan", "count_stale_dead_letters", "plan"]
