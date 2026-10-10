# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""``remember(..., replaces=<id>)``: the caller says which earlier memory a new one replaces.

"Latest wins" is then exact when the caller knows it, instead of guessed.

WHICH ID
--------
Callers hold two kinds of id. ``remember`` returns ``fact_ids`` - one fact per
new memory, the searchable copy of exactly what was saved. ``recall`` returns a
``fact_id`` and a ``memory_id`` for every result. All of them are accepted:

* a ``memory_id`` retires every current fact of that memory;
* the fact id ``remember`` returned for a memory stands for that whole memory,
  so it also retires every current fact of it - including the facts that
  background enrichment extracted from it after it was saved. Without this the
  most common call (save, keep the id you were given, later save the update
  with ``replaces=<that id>``) would leave the old memory's extracted facts in
  every recall, which is exactly what this exists to stop;
* any other fact id - one fact extracted from a longer memory, as ``recall``
  may return - retires just that fact.

"Current" means not already retired. A fact that an earlier correction or
replacement retired keeps that record untouched.

WHO MAY REPLACE WHAT
--------------------
The rule every memory mutation follows (``_authorize_memory_mutation`` in
``server/routes/memories.py`` and the canonical mutation writer): an
authenticated writer with WRITE permission -- and, for a replacement, the
CORRECT policy -- on the profile being written, acting only on facts that
profile owns. That profile is the one the request names with ``profile_id``,
or the active one when it names none; the writer accepts a replacement routed
to another profile (``core/mutation_routing.py``), and its undo through
``review_correction`` is routed the same way. A global or shared memory of
another profile is visible to this one but is refused with a plain message; a
private memory of another profile is reported as not found, exactly like an id
that does not exist, so ``replaces`` cannot be used to probe other profiles.
The new memory must be saved to the same profile and in the same scope as what
it replaces, because a correction case never crosses a profile or a scope.

HOW IT IS RECORDED
------------------
Through the review-gated correction ledger (``storage/correction_cases.py``),
not a mechanism of its own. Saving with ``replaces`` is the reviewer's decision
made explicitly, so each retired fact gets one case that is proposed and
applied inside a single canonical-writer transaction: every fact or none.
Applying snapshots the old fact's temporal row and marks it replaced by the new
fact, reason ``replaced_by_caller``; nothing is deleted. Undo is the ledger's
own rollback (``review_correction`` with ``action="rollback"``), which restores
that snapshot exactly and leaves the new memory in place. Store repair undoes
only marks written by automatic checks, matched by reason, never this one.

A whole memory may still be enriching when it is replaced. Facts enrichment
derives from it afterwards are retired as they are written, in the write's own
transaction, and come back when the replacement is undone - see
``storage/replaced_memory.py``. Its cases are marked as a whole-memory group
for that reason; a single-fact replacement is not, and leaves later facts of
the memory alone.

ORDER
-----
The id is checked before anything is saved (``check_replaceable``). The old
facts are marked only after the new memory is durably saved
(``replace_after_save``), keyed on the save's operation id, so a replayed
request is answered from the first receipt instead of marking twice. If the
mark fails, the memory stays saved and the caller is told - in ``replaced`` -
that nothing was replaced and why. A replacement is never reported that is not
in force.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core.remember_runtime import (
    CanonicalMutationConflict,
    UnknownMutationProfile,
)
from superlocalmemory.core.replaces_input import (
    NOT_ALLOWED,
    NOT_FOUND,
    ReplacesRejected,
    normalize_replaces,
)
from superlocalmemory.storage.correction_cases import (
    CALLER_REPLACEMENT_REASON,
    CorrectionActor,
    CorrectionCase,
    CorrectionCaseError,
    propose_on_connection,
    transition_on_connection,
)
from superlocalmemory.storage.replaced_memory import (
    SINGLE_FACT,
    WHOLE_MEMORY,
    case_id_for,
    group_key,
)

logger = logging.getLogger("superlocalmemory.audit")

#: Most facts one replacement may retire. A memory is a few facts; a memory
#: with hundreds is an imported document, and retiring it wholesale in one
#: writer transaction is not something to do on a remember call.
MAX_FACTS = 200
UNDO_HINT = ("To undo, call review_correction with each case_id, action='rollback' "
             "and that case's version.")
#: The same, for a replacement saved to a profile named by ``profile_id``: the
#: undo must name it too, or it would look in the active profile.
ROUTED_UNDO_HINT = ("To undo, call review_correction with profile_id={profile!r}, each "
                    "case_id, action='rollback' and that case's version.")

Query = Callable[[str, tuple[Any, ...]], Sequence[Any]]
_COLUMNS = "fact_id, memory_id, profile_id, scope, shared_with"


class ReplacementRefused(CanonicalMutationConflict):
    """The writer declined to mark anything; the message says why, plainly.

    A ``CanonicalMutationConflict`` so the writer reports it as a decision, not
    as an outage, and its transaction rolls back with nothing changed.
    """


@dataclass(frozen=True, slots=True)
class _Fact:
    fact_id: str
    memory_id: str
    profile_id: str
    scope: str
    shared_with: str | None


def _facts(query: Query, where: str, params: tuple[Any, ...]) -> list[_Fact]:
    rows = query(f"SELECT {_COLUMNS} FROM atomic_facts WHERE {where}", params)
    return [_Fact(str(r["fact_id"]), str(r["memory_id"] or ""), str(r["profile_id"]),
                  str(r["scope"] or "personal"), r["shared_with"]) for r in rows]


def _visible_to(fact: _Fact, profile_id: str) -> bool:
    if fact.scope == "global":
        return True
    if fact.scope != "shared" or not fact.shared_with:
        return False
    try:
        names = json.loads(fact.shared_with)
    except (TypeError, ValueError):
        return False
    return isinstance(names, list) and profile_id in names


def _not_found(replaces: str) -> ReplacesRejected:
    return ReplacesRejected(
        NOT_FOUND, f"No memory with id {replaces} in this profile. Use a fact_id or "
                   "memory_id returned by remember or recall.")


def not_found(replaces: str) -> ReplacesRejected:
    """The refusal for an id that names no memory; also given for one a remote app may not see."""
    return _not_found(replaces)


def _foreign(facts: list[_Fact], profile_id: str, replaces: str) -> ReplacesRejected:
    if any(_visible_to(f, profile_id) for f in facts):
        return ReplacesRejected(
            NOT_ALLOWED, f"Memory {replaces} belongs to another profile. Only memories "
                         "saved in this profile can be replaced.")
    return _not_found(replaces)


def _stands_for_its_memory(query: Query, fact: _Fact) -> bool:
    """True when ``fact`` is the copy ``remember`` made of its whole memory."""
    try:
        memory = query("SELECT metadata_json FROM memories WHERE memory_id=? AND profile_id=?",
                       (fact.memory_id, fact.profile_id))
        if not memory:
            return False
        operation_id = json.loads(memory[0]["metadata_json"] or "{}").get(
            "ingestion_operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            return False
        operation = query("SELECT queryable_fact_ids_json FROM ingestion_operations "
                          "WHERE operation_id=? AND profile_id=?",
                          (operation_id, fact.profile_id))
        return bool(operation) and fact.fact_id in json.loads(
            operation[0]["queryable_fact_ids_json"] or "[]")
    except (sqlite3.OperationalError, TypeError, ValueError, AttributeError) as exc:
        # A store without the ingestion ledger, or unreadable metadata: the id
        # is then treated as naming one fact, the narrower reading.
        logger.debug("replaces: %s read as a single fact (%s)", fact.fact_id,
                     type(exc).__name__)
        return False


def _resolve(query: Query, replaces: str, profile_id: str) -> tuple[list[_Fact], str | None]:
    """The facts ``replaces`` names, and the memory id when it names a whole memory."""
    by_fact = _facts(query, "fact_id = ?", (replaces,))
    if by_fact:
        fact = by_fact[0]
        if fact.profile_id != profile_id:
            raise _foreign(by_fact, profile_id, replaces)
        if not fact.memory_id or not _stands_for_its_memory(query, fact):
            return [fact], None
        return _facts(query, f"memory_id = ? AND profile_id = ? ORDER BY fact_id "
                             f"LIMIT {MAX_FACTS + 1}", (fact.memory_id, profile_id)), fact.memory_id
    by_memory = _facts(query, f"memory_id = ? LIMIT {MAX_FACTS + 1}", (replaces,))
    own = [f for f in by_memory if f.profile_id == profile_id]
    if own:
        return own, replaces
    if by_memory:
        raise _foreign(by_memory, profile_id, replaces)
    raise _not_found(replaces)


def named_facts(query: Query, replaces: str, profile_id: str) -> list[_Fact]:
    """Every fact of ``profile_id`` that ``replaces`` names (module docstring)."""
    return _resolve(query, replaces, profile_id)[0]


def _require_scope(facts: list[_Fact], scope: str, replaces: str) -> None:
    other = sorted({f.scope for f in facts} - {scope})
    if other:
        raise ReplacesRejected(
            NOT_ALLOWED, f"A {scope} memory can only replace another {scope} memory, and "
                         f"{replaces} is {', '.join(other)}. Save this one with that "
                         "scope, or leave replaces out.")


def check_replaceable(db: Any, *, replaces: object, profile_id: str, scope: str) -> str:
    """Refuse a ``replaces`` value before anything is saved; return the clean id.

    ``profile_id`` is the profile the new memory is being saved to, routed or
    active: what ``replaces`` names must belong to it. Read-only. Whether the
    named facts are still current is decided at mark time instead: a replayed
    request must not be refused because its own first attempt already retired
    them.
    """
    named = normalize_replaces(replaces)
    facts = named_facts(db.execute, named, profile_id)
    _require_scope(facts, scope, named)
    return named


def ledger_actor_id(actor_id: str) -> str:
    """The trusted actor as the ledger can hold it: a safe id of at most 128 bytes.

    Install-token writers carry ids longer than that; they are recorded by a
    stable digest, so the same writer always maps to the same ledger id.
    """
    raw = actor_id.encode("utf-8")
    if 0 < len(raw) <= 128 and not any(ch.isspace() for ch in actor_id) and "\x00" not in actor_id:
        return actor_id
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _payload_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"replacement command is missing {key}")
    return value


def _current(query: Query, facts: list[_Fact], profile_id: str) -> list[_Fact]:
    if not facts:
        return []
    marks = ",".join("?" for _ in facts)
    retired = {str(r["fact_id"]) for r in query(
        "SELECT fact_id FROM fact_temporal_validity WHERE profile_id = ? "
        f"AND system_expired_at IS NOT NULL AND fact_id IN ({marks})",
        (profile_id, *(f.fact_id for f in facts)))}
    return [f for f in facts if f.fact_id not in retired]


def _retire(conn: sqlite3.Connection, fact: _Fact, successor: str,
            actor: CorrectionActor, profile_id: str, group: str | None) -> CorrectionCase:
    case_id = case_id_for(profile_id, fact.fact_id, successor)

    def is_profile(candidate: str) -> bool:
        return candidate == profile_id

    def is_actor(candidate: CorrectionActor) -> bool:
        return candidate == actor

    # A whole-memory replacement carries its group, so facts the memory's
    # enrichment writes later are retired with it (storage/replaced_memory.py).
    key = f"{WHOLE_MEMORY}{group}:{case_id}" if group else f"{SINGLE_FACT}{case_id}"
    propose_on_connection(
        conn, case_id=case_id, profile_id=profile_id, scope=fact.scope,
        predecessor_fact_id=fact.fact_id, successor_fact_id=successor,
        reason_code=CALLER_REPLACEMENT_REASON, actor=actor,
        idempotency_key=key,
        is_profile_active=is_profile, is_actor_trusted=is_actor,
    )
    return transition_on_connection(
        conn, case_id=case_id, expected_version=0, actor=actor,
        operation_id=f"replaces:apply:{case_id}", from_status="proposed",
        to_status="applied", mutate_temporal=True,
        is_profile_active=is_profile, is_actor_trusted=is_actor,
    )


def apply_replacement(conn: sqlite3.Connection, profile_id: str,
                      payload: Mapping[str, Any]) -> dict[str, Any]:
    """Writer-transaction body for ``CommandKind.REPLACE_BY_CALLER``.

    Re-resolves what ``replaces`` names inside the transaction, so nothing that
    changed since the pre-save check is acted on blindly. Any refusal or error
    leaves the transaction to roll back: all facts are retired, or none.
    """
    replaces = _payload_text(payload, "replaces")
    successor = _payload_text(payload, "successor_fact_id")
    actor_id = _payload_text(payload, "trusted_actor_id")

    def query(sql: str, params: tuple[Any, ...]) -> list[Any]:
        return conn.execute(sql, params).fetchall()

    new = _facts(query, "fact_id = ? AND profile_id = ?", (successor, profile_id))
    if not new:
        raise ReplacementRefused("The new memory is not in this profile, so nothing was replaced.")
    try:
        facts, whole_memory = _resolve(query, replaces, profile_id)
        _require_scope(facts, new[0].scope, replaces)
    except ReplacesRejected as exc:
        raise ReplacementRefused(exc.message) from exc
    group = group_key(profile_id, whole_memory, successor) if whole_memory else None
    facts = [f for f in facts if f.fact_id != successor and f.memory_id != new[0].memory_id]
    if len(facts) > MAX_FACTS:
        raise ReplacementRefused(
            f"{replaces} has more than {MAX_FACTS} facts; replace them by fact_id instead.")
    current = _current(query, facts, profile_id)
    if not current:
        raise ReplacementRefused(f"Nothing current to replace: {replaces} was already replaced.")
    # A fact may hold one open case at a time. One a PERSON proposed, waiting
    # for review, blocks this one; say so, rather than a retry that can never
    # succeed. One SLM proposed by itself is overtaken by this explicit request
    # (core/overtaken_cases.py, Varun 2026-10-06), in this same transaction.
    from superlocalmemory.core import overtaken_cases as _ot

    open_cases = [c for c in _ot.cases_naming(conn, [f.fact_id for f in current],
                                              predecessor_only=True)
                  if c["profile_id"] == profile_id and c["status"] == "proposed"]
    waiting = _ot.blocking(open_cases)
    if waiting:
        raise ReplacementRefused(
            f"{waiting[0]['predecessor_fact_id']} has a correction waiting for review, so "
            f"nothing was replaced. Review it (list_corrections, review_correction) in profile "
            f"{profile_id!r}, then repeat this request.")
    _ot.overtake(conn, open_cases, user_action="replace", actor_id=actor_id,
                 operation_id=f"replaces:{replaces}:{successor}")
    actor = CorrectionActor(actor_id=ledger_actor_id(actor_id),
                            actor_kind="host_authenticated", trust_tier="trusted")
    try:
        cases = [_retire(conn, fact, successor, actor, profile_id, group) for fact in current]
    except (CorrectionCaseError, ValueError) as exc:
        raise ReplacementRefused(
            f"The correction ledger refused the replacement ({exc}).") from exc
    logger.info("caller replacement: %d fact(s) named by %s replaced by %s",
                len(cases), replaces[:32], successor[:32])
    return {
        "ok": True,
        "replaces": replaces,
        "successor_fact_id": successor,
        "fact_ids": [c.predecessor_fact_id for c in cases],
        "cases": [{"case_id": c.case_id, "version": c.version} for c in cases],
    }


def _not_replaced(replaces: str, reason: str) -> dict[str, Any]:
    return {"ok": False, "replaces": replaces, "reason": reason}


def _undone_since(engine: Any, cases: list[dict[str, Any]]) -> bool:
    """A replayed receipt describes the first attempt; say so if it was undone."""
    ids = [str(c.get("case_id")) for c in cases]
    if not ids:
        return False
    try:
        rows = engine._db.execute(
            "SELECT 1 FROM correction_cases WHERE status != 'applied' AND case_id IN ("
            + ",".join("?" for _ in ids) + ") LIMIT 1", tuple(ids))
    except Exception as exc:  # noqa: BLE001 - the committed receipt stays authoritative
        logger.warning("replaces: could not re-read case status (%s)", type(exc).__name__)
        return False
    return bool(rows)


def replace_after_save(runtime: Any, engine: Any, *, replaces: str, profile_id: str,
                       successor_fact_ids: Sequence[str], operation_id: str,
                       trusted_actor_id: str, routed: bool = False) -> dict[str, Any]:
    """Mark what ``replaces`` names, after the new memory is durably saved.

    Never raises: the save has already happened and must be reported as such.
    The result is the ``replaced`` field of the remember response. ``routed``
    says the request named ``profile_id``, so the undo hint names it as well.
    """
    if not successor_fact_ids:
        return _not_replaced(replaces, "The new memory produced nothing searchable, "
                                       "so nothing was replaced.")
    key = "replaces:" + hashlib.sha256(
        f"{profile_id}\0{operation_id}".encode("utf-8")).hexdigest()[:48]
    try:
        receipt = runtime.replace_by_caller(
            profile_id, replaces, str(successor_fact_ids[0]),
            trusted_actor_id=trusted_actor_id, idempotency_key=key)
    except ReplacementRefused as exc:
        return _not_replaced(replaces, str(exc))
    except UnknownMutationProfile:
        # No retry hint: a deleted profile does not come back.
        return _not_replaced(replaces, f"Profile {profile_id!r} no longer exists, so "
                                       "nothing was replaced.")
    except CanonicalMutationConflict:
        return _not_replaced(replaces, "This request was already saved replacing something "
                                       "else; nothing more was replaced.")
    except Exception as exc:  # noqa: BLE001 - the save stands; report, never raise
        logger.warning("replaces: %s not marked (%s)", replaces[:32], type(exc).__name__)
        return _not_replaced(
            replaces, f"The new memory is saved, but nothing was replaced: the writer could "
                      f"not record it ({type(exc).__name__}). Repeat the same request with "
                      "the same idempotency key to try again.")
    cases = [dict(c) for c in receipt.get("cases") or ()]
    if not receipt.get("ok") or not cases:
        return _not_replaced(replaces, "The writer did not confirm the replacement.")
    if _undone_since(engine, cases):
        return _not_replaced(replaces, "This replacement was recorded earlier and has "
                                       "since been undone.")
    from superlocalmemory.core.mutations import purge_profile_context_cache

    purge_profile_context_cache(engine, profile_id)
    undo = ROUTED_UNDO_HINT.format(profile=profile_id) if routed else UNDO_HINT
    return {"ok": True, "replaces": replaces, "fact_ids": list(receipt.get("fact_ids") or ()),
            "cases": cases, "undo": undo}


__all__ = ["MAX_FACTS", "ROUTED_UNDO_HINT", "ReplacementRefused", "UNDO_HINT", "not_found",
           "apply_replacement", "check_replaceable",
           "ledger_actor_id", "named_facts", "replace_after_save"]
