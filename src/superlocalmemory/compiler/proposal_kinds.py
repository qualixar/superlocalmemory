# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What each kind of proposal does when applied, and exactly how it is undone (WS-6, I-10).

A handler has four parts: ``validate`` (at proposal time: the target exists and
the payload is well formed; returns the fact ids the proposal names, which erase
it with them, I-11), ``describe`` (one plain sentence for the review list),
``apply`` (does the change inside the caller's savepoint and returns its inverse)
and ``rollback`` (restores exactly what ``apply`` changed, from that inverse).

``retire`` and ``merge`` go through the correction ledger, which stays the one
place a memory is retired: applying creates a ledger case and applies it,
rolling back rolls the ledger case back (the ledger restores the predecessor's
validity row from its own snapshot). ``edit_derived`` rewrites the text of one
derived artifact after keeping its previous row as a version, and records the
whole previous row as its inverse, so rollback does not depend on how many
versions are retained. ``create_page`` is registered by the pages module
(``compiler.pages``, D-18) and ``wiki_edit`` by ``compiler.wiki_edit`` (D-17) with
:func:`register_handler`.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from superlocalmemory.compiler import versions
from superlocalmemory.compiler.versions import select_rows as _select

#: Modules that register further kinds when present (pages: package E).
OPTIONAL_HANDLER_MODULES = ("superlocalmemory.compiler.pages",
                            "superlocalmemory.compiler.wiki_edit")
_TEXT_LIMIT = 20_000
_GLANCE = 90


class ProposalError(ValueError):
    """The proposal or the change is not valid; nothing was changed."""


@dataclass(frozen=True)
class ApplyContext:
    profile_id: str
    set_id: str
    proposal_id: str
    actor_id: str
    actor_kind: str
    trust_tier: str


@dataclass(frozen=True)
class Handler:
    validate: Callable[[Any, str, dict], list[str]]
    describe: Callable[[Any, str, dict], str]
    apply: Callable[[Any, ApplyContext, dict], dict]
    rollback: Callable[[Any, ApplyContext, dict, dict], None]
    #: Optional: called when a person rejects the set (a wrapped ledger case is rejected too).
    on_reject: Callable[[Any, ApplyContext, dict], None] | None = None


HANDLERS: dict[str, Handler] = {}


def register_handler(kind: str, handler: Handler) -> None:
    HANDLERS[kind] = handler


def handler_for(kind: str) -> Handler:
    try:
        return HANDLERS[kind]
    except KeyError:
        raise ProposalError(f"No handler for proposals of kind {kind!r}.") from None


def load_optional_handlers() -> None:
    for name in OPTIONAL_HANDLER_MODULES:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise


def _glance(text: Any) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= _GLANCE else flat[:_GLANCE - 1] + "…"


def _fact(conn: Any, profile_id: str, fact_id: Any) -> dict:
    rows = _select(conn, "SELECT fact_id, scope, content FROM atomic_facts "
                   "WHERE fact_id = ? AND profile_id = ?", (str(fact_id), profile_id))
    if not rows:
        raise ProposalError("A memory named by the proposal is not in this profile.")
    return rows[0]


# -- the correction ledger ------------------------------------------------------------------------

def _ledger(ctx: ApplyContext):
    from superlocalmemory.storage.correction_cases import CorrectionActor

    actor = CorrectionActor(ctx.actor_id, ctx.actor_kind, ctx.trust_tier)
    return actor, (lambda p: p == ctx.profile_id), (lambda a: a == actor)


def _op(ctx: ApplyContext, case_id: str, step: str) -> str:
    return "prop-" + hashlib.sha256(f"{ctx.proposal_id}\0{case_id}\0{step}".encode()).hexdigest()[:40]


def _case(conn: Any, profile_id: str, case_id: str) -> dict:
    rows = _select(conn, "SELECT case_id, status, version, predecessor_fact_id, successor_fact_id "
                   "FROM correction_cases WHERE case_id = ? AND profile_id = ?",
                   (case_id, profile_id))
    if not rows:
        raise ProposalError("The correction case is no longer in this profile.")
    return rows[0]


def _transition(conn: Any, ctx: ApplyContext, case_id: str, frm: str, to: str, *,
                temporal: bool) -> None:
    from superlocalmemory.storage.correction_cases import transition_on_connection

    case = _case(conn, ctx.profile_id, case_id)
    if case["status"] != frm:
        raise ProposalError(f"The correction case is {case['status']}, not {frm}; "
                            "it changed since this was proposed.")
    actor, active, trusted = _ledger(ctx)
    transition_on_connection(
        conn, case_id=case_id, expected_version=int(case["version"]), actor=actor,
        operation_id=_op(ctx, case_id, to), from_status=frm, to_status=to,
        mutate_temporal=temporal, is_profile_active=active, is_actor_trusted=trusted)


def _open_and_apply(conn: Any, ctx: ApplyContext, predecessor: str, successor: str,
                    reason: str) -> str:
    from superlocalmemory.storage.correction_cases import propose_on_connection

    actor, active, trusted = _ledger(ctx)
    case_id = "prop-" + hashlib.sha256(
        f"{ctx.proposal_id}\0{predecessor}\0{successor}".encode()).hexdigest()[:40]
    scope = str(_fact(conn, ctx.profile_id, predecessor)["scope"] or "personal")
    propose_on_connection(
        conn, case_id=case_id, profile_id=ctx.profile_id, scope=scope,
        predecessor_fact_id=predecessor, successor_fact_id=successor, reason_code=reason,
        actor=actor, idempotency_key=_op(ctx, case_id, "propose"),
        is_profile_active=active, is_actor_trusted=trusted)
    _transition(conn, ctx, case_id, "proposed", "applied", temporal=True)
    return case_id


def _check_open(conn: Any, profile_id: str, predecessor: str) -> None:
    from superlocalmemory.storage.correction_cases import open_case_id

    if open_case_id(conn, profile_id, predecessor) is not None:
        raise ProposalError("One of the memories already has an open correction; "
                            "settle that first.")


# -- retire ----------------------------------------------------------------------------------------

def _retire_validate(conn: Any, profile_id: str, payload: dict) -> list[str]:
    if payload.get("case_id"):
        case = _case(conn, profile_id, str(payload["case_id"]))
        if case["status"] != "proposed":
            raise ProposalError("Only a correction case still waiting for review can be proposed.")
        return [case["predecessor_fact_id"], case["successor_fact_id"]]
    old, new = payload.get("predecessor"), payload.get("successor")
    if not old or not new or old == new:
        raise ProposalError("A retire proposal names the memory to retire and the one that "
                            "replaces it.")
    _fact(conn, profile_id, old)
    _fact(conn, profile_id, new)
    _check_open(conn, profile_id, str(old))
    return [str(old), str(new)]


def _retire_describe(conn: Any, profile_id: str, payload: dict) -> str:
    if payload.get("case_id"):
        case = _case(conn, profile_id, str(payload["case_id"]))
        old, new = case["predecessor_fact_id"], case["successor_fact_id"]
    else:
        old, new = payload["predecessor"], payload["successor"]
    return (f"Retire “{_glance(_fact(conn, profile_id, old)['content'])}” because "
            f"“{_glance(_fact(conn, profile_id, new)['content'])}” replaces it")


def _retire_apply(conn: Any, ctx: ApplyContext, payload: dict) -> dict:
    if payload.get("case_id"):
        case_id = str(payload["case_id"])
        _transition(conn, ctx, case_id, "proposed", "applied", temporal=True)
        return {"cases": [case_id], "adopted": True}
    case_id = _open_and_apply(conn, ctx, str(payload["predecessor"]), str(payload["successor"]),
                              str(payload.get("reason") or "compiler_retire"))
    return {"cases": [case_id], "adopted": False}


def _rollback_cases(conn: Any, ctx: ApplyContext, inverse: dict) -> None:
    for case_id in reversed(inverse.get("cases") or []):
        _transition(conn, ctx, str(case_id), "applied", "rolled_back", temporal=True)


def _retire_rollback(conn: Any, ctx: ApplyContext, payload: dict, inverse: dict) -> None:
    _rollback_cases(conn, ctx, inverse)


def _retire_reject(conn: Any, ctx: ApplyContext, payload: dict) -> None:
    if payload.get("case_id"):
        _transition(conn, ctx, str(payload["case_id"]), "proposed", "rejected", temporal=False)


# -- merge -----------------------------------------------------------------------------------------

def _merge_validate(conn: Any, profile_id: str, payload: dict) -> list[str]:
    keep, drop = payload.get("keep"), payload.get("retire")
    if not keep or not isinstance(drop, list) or not drop or keep in drop or len(set(drop)) != len(drop):
        raise ProposalError("A merge names the memory to keep and the copies to retire.")
    for fact_id in [keep, *drop]:
        _fact(conn, profile_id, fact_id)
    for fact_id in drop:
        _check_open(conn, profile_id, str(fact_id))
    return [str(keep), *[str(d) for d in drop]]


def _merge_describe(conn: Any, profile_id: str, payload: dict) -> str:
    kept = _glance(_fact(conn, profile_id, payload["keep"])["content"])
    n = len(payload["retire"])
    return f"Keep “{kept}” and retire {n} {'copy' if n == 1 else 'copies'} of it"


def _merge_apply(conn: Any, ctx: ApplyContext, payload: dict) -> dict:
    cases = [_open_and_apply(conn, ctx, str(old), str(payload["keep"]), "compiler_merge")
             for old in payload["retire"]]
    return {"cases": cases, "adopted": False}


# -- edit_derived ----------------------------------------------------------------------------------

_LABEL = {"entity_profile": "an entity profile", "community_summary": "a topic summary",
          "persona": "the persona summary", "consolidated_summary": "a consolidated summary",
          "scene": "a scene title", "core_block": "a core memory block"}


def _target(conn: Any, spec: versions.ArtifactSpec, key: Mapping[str, Any]) -> dict:
    if set(key) != set(spec.key_columns):
        raise ProposalError(f"The target of {spec.name} is named by {', '.join(spec.key_columns)}.")
    where = " AND ".join(f"{c} = ?" for c in spec.key_columns)
    rows = _select(conn, f"SELECT * FROM {spec.table} WHERE {where}",  # noqa: S608
                   tuple(key[c] for c in spec.key_columns))
    if len(rows) != 1:
        raise ProposalError("The text to edit is not there (it may have been rebuilt).")
    return rows[0]


def _edit_validate(conn: Any, profile_id: str, payload: dict) -> list[str]:
    spec = versions.spec_for(str(payload.get("artifact") or ""))
    key, new = payload.get("key"), payload.get("set")
    if not isinstance(key, dict) or not isinstance(new, dict) or not new:
        raise ProposalError("An edit names the text (artifact and key) and what to set.")
    if set(new) - set(spec.editable):
        raise ProposalError(f"Only {', '.join(spec.editable)} can be edited on {spec.name}.")
    if any(not isinstance(v, str) or len(v) > _TEXT_LIMIT for v in new.values()):
        raise ProposalError("Edited text must be text of a sensible length.")
    row = _target(conn, spec, key)
    if str(row.get("profile_id")) != profile_id:
        raise ProposalError("That text belongs to another profile.")
    return spec.facts(conn, row)


def _edit_describe(conn: Any, profile_id: str, payload: dict) -> str:
    text = next(iter(payload["set"].values()))
    return f"Edit {_LABEL.get(payload['artifact'], 'derived text')}: “{_glance(text)}”"


def _edit_apply(conn: Any, ctx: ApplyContext, payload: dict) -> dict:
    spec = versions.spec_for(payload["artifact"])
    key = dict(payload["key"])
    before = versions._record(_target(conn, spec, key))
    versions.keep_previous(conn, spec.name, key, reason=versions.EDIT, proposal_id=ctx.proposal_id)
    changes = dict(payload["set"])
    if spec.name == "core_block" and "content" in changes:
        changes["char_count"] = len(changes["content"])
    where = " AND ".join(f"{c} = ?" for c in spec.key_columns)
    conn.execute(f"UPDATE {spec.table} SET " + ", ".join(f"{c} = ?" for c in changes)  # noqa: S608
                 + f" WHERE {where}", (*changes.values(), *[key[c] for c in spec.key_columns]))
    after = _target(conn, spec, key)
    return {"artifact": spec.name, "key": key, "before": before,
            "after_hash": versions.content_hash(spec, after)}


def _edit_rollback(conn: Any, ctx: ApplyContext, payload: dict, inverse: dict) -> None:
    spec = versions.spec_for(inverse["artifact"])
    current = _target(conn, spec, inverse["key"])
    if versions.content_hash(spec, current) != inverse["after_hash"]:
        raise ProposalError("This text was changed again after the edit, so it was not rolled "
                            "back. The earlier text is kept as a version.")
    versions.keep_previous(conn, spec.name, inverse["key"], reason=versions.RESTORE,
                           proposal_id=ctx.proposal_id)
    versions.write_row(conn, spec, inverse["before"])


register_handler("retire", Handler(_retire_validate, _retire_describe, _retire_apply,
                                   _retire_rollback, _retire_reject))
register_handler("merge", Handler(_merge_validate, _merge_describe, _merge_apply,
                                  _retire_rollback))
register_handler("edit_derived", Handler(_edit_validate, _edit_describe, _edit_apply,
                                         _edit_rollback))

__all__ = ["ApplyContext", "HANDLERS", "Handler", "ProposalError", "handler_for",
           "load_optional_handlers", "register_handler"]
