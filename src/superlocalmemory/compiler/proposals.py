# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""ProposalSets: a unit of proposed change with its basis and its exact rollback (WS-6, D-09, I-10).

A set holds proposals of four kinds (``retire``, ``merge``, ``edit_derived``,
``create_page``). Nothing in it touches a memory until the whole set is applied,
and applying needs a basis (I-10):

* a **person's approval** (``Basis.person``: who and through which door), or
* a **certificate id** / **rule-certificate id** of an open ``judge_certificates``
  row that covers the kind of change (retire: SLOT_RELATION ``replaces``; merge:
  DUPLICATE ``duplicate``; edit and page: CITATION_SUPPORT ``supported``),
  which acts only while ``compiler.auto_apply`` is on and the kill switch is off.

Applying runs every change inside one savepoint and records each change's
inverse on the proposal, so rolling the set back restores exactly what the
apply changed (``rollback_set``). A rejected set changes nothing. The correction
ledger (``correction_cases``) stays the single store of retirements: a retire
or merge proposal creates its ledger cases when it is applied, and a case that
already exists can be adopted as a retire proposal (``adopt_case``) or shown as
one (``case_as_proposal``).

Whether a caller is a person is decided by the door (``slm proposals`` at an
interactive terminal, the dashboard on this computer), never here: this module
only refuses to act without a basis.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.compiler.proposal_kinds import (
    HANDLERS,
    ApplyContext,
    ProposalError,
    handler_for,
    load_optional_handlers,
)
from superlocalmemory.compiler.versions import select_rows as _select

PROPOSED, APPLIED, ROLLED_BACK, REJECTED = "proposed", "applied", "rolled_back", "rejected"
PERSON, CERTIFICATE, RULE_CERTIFICATE = "person", "certificate", "rule_certificate"
#: Doors where a person can sit: the terminal and the dashboard.
PERSON_CHANNELS = frozenset({"cli", "dashboard"})
#: Which certified verdict covers each kind of change (question id, verdict).
COVERS: dict[str, tuple[str, str]] = {
    "retire": ("SLOT_RELATION", "replaces"),
    "merge": ("DUPLICATE", "duplicate"),
    "edit_derived": ("CITATION_SUPPORT", "supported"),
    "create_page": ("CITATION_SUPPORT", "supported"),
}
RULE_TIER = "rules"
_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


class BasisRefused(ProposalError):
    """The change has no lawful basis (I-10); nothing was changed."""


class StateRefused(ProposalError):
    """The set is not in a state that allows this; nothing was changed."""


@dataclass(frozen=True)
class Basis:
    """Why a change may happen: a person's approval, a certificate or a rule certificate."""

    kind: str
    ref: str
    actor: str = ""
    channel: str = ""

    @classmethod
    def person(cls, actor: str, channel: str) -> Basis:
        return cls(PERSON, f"{actor}@{channel}", actor, channel)

    @classmethod
    def certificate(cls, cert_id: str) -> Basis:
        return cls(CERTIFICATE, cert_id)

    @classmethod
    def rule_certificate(cls, cert_id: str) -> Basis:
        return cls(RULE_CERTIFICATE, cert_id)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _flags(flags: Any) -> Any:
    if flags is not None:
        return flags
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    return FeatureFlags(canonical_data_root())


def check_basis(conn: Any, profile_id: str, basis: Basis | None, kinds: Sequence[str],
                flags: Any = None) -> None:
    """Raise :class:`BasisRefused` unless ``basis`` lawfully covers every kind (I-10)."""
    if basis is None:
        raise BasisRefused("A change needs a certificate id, a rule-certificate id or a "
                           "person's approval. Nothing was changed.")
    if basis.kind == PERSON:
        if not basis.actor or not _ID.match(basis.actor) or basis.channel not in PERSON_CHANNELS:
            raise BasisRefused("A person's approval names who approved and the terminal or "
                               "dashboard they used. Nothing was changed.")
        return
    if basis.kind not in (CERTIFICATE, RULE_CERTIFICATE) or not basis.ref:
        raise BasisRefused("Unknown basis. Nothing was changed.")
    if not _flags(flags).enabled("compiler.auto_apply"):
        raise BasisRefused("Applying on a certificate needs compiler.auto_apply to be on "
                           "(it is off by default). Nothing was changed.")
    rows = _select(conn, "SELECT tier, question_id, verdict FROM judge_certificates "
                   "WHERE cert_id = ? AND profile_id = ? AND voided_at IS NULL",
                   (basis.ref, profile_id))
    if not rows:
        raise BasisRefused("That certificate does not exist, is voided or belongs to another "
                           "profile. Nothing was changed.")
    cert = rows[0]
    if (cert["tier"] == RULE_TIER) != (basis.kind == RULE_CERTIFICATE):
        raise BasisRefused("The basis kind does not match the certificate (rules vs models). "
                           "Nothing was changed.")
    for kind in kinds:
        wanted = COVERS.get(kind)
        if wanted is None or (cert["question_id"], cert["verdict"]) != wanted:
            raise BasisRefused(f"That certificate does not cover a {kind} change. "
                               "Nothing was changed.")


def create_set(conn: Any, *, profile_id: str, origin: str, title: str = "",
               proposals: Sequence[Mapping[str, Any]]) -> str:
    """Record a new set of proposals (state ``proposed``); nothing changes. Returns its id."""
    load_optional_handlers()
    if not proposals:
        raise ProposalError("A set needs at least one proposal.")
    if not origin or not _ID.match(origin):
        raise ProposalError("A set needs an origin (who proposed it).")
    prepared = []
    for item in proposals:
        kind = str(item.get("kind") or "")
        handler = handler_for(kind)
        payload = dict(item.get("payload") or {})
        links = handler.validate(conn, profile_id, payload)
        prepared.append((kind, payload, handler.describe(conn, profile_id, payload), links))
    set_id, now = uuid.uuid4().hex, _now()
    conn.execute("INSERT INTO proposal_sets (set_id, profile_id, title, origin, state, created_at) "
                 "VALUES (?, ?, ?, ?, 'proposed', ?)", (set_id, profile_id, title[:200], origin, now))
    for seq, (kind, payload, summary, links) in enumerate(prepared):
        pid = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO proposals (proposal_id, set_id, profile_id, seq, kind, summary, "
            "payload_json, state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?)",
            (pid, set_id, profile_id, seq, kind, summary, json.dumps(payload, sort_keys=True), now))
        for fact_id in sorted(set(links)):
            conn.execute("INSERT OR IGNORE INTO proposal_facts (proposal_id, fact_id) VALUES (?, ?)",
                         (pid, fact_id))
    return set_id


def _set_row(conn: Any, profile_id: str, set_id: str) -> dict:
    rows = _select(conn, "SELECT * FROM proposal_sets WHERE set_id = ? AND profile_id = ?",
                   (set_id, profile_id))
    if not rows:
        raise LookupError("no such proposal set in this profile")
    return rows[0]


def _members(conn: Any, set_id: str) -> list[dict]:
    return _select(conn, "SELECT * FROM proposals WHERE set_id = ? ORDER BY seq", (set_id,))


def _view(row: dict, members: list[dict]) -> dict:
    basis = ({"kind": row["basis_kind"], "ref": row["basis_ref"]} if row["basis_kind"] else None)
    return {
        "set_id": row["set_id"], "profile_id": row["profile_id"], "title": row["title"],
        "origin": row["origin"], "state": row["state"], "basis": basis,
        "decided_by": row["decided_by"], "decided_channel": row["decided_channel"],
        "created_at": row["created_at"], "decided_at": row["decided_at"],
        "applied_at": row["applied_at"], "rolled_back_at": row["rolled_back_at"],
        "proposals": [{"proposal_id": m["proposal_id"], "kind": m["kind"],
                       "summary": m["summary"], "state": m["state"]} for m in members],
    }


def get_set(conn: Any, profile_id: str, set_id: str) -> dict:
    row = _set_row(conn, profile_id, set_id)
    return _view(row, _members(conn, set_id))


def list_sets(conn: Any, profile_id: str, *, states: Sequence[str] | None = None,
              limit: int = 50) -> list[dict]:
    """Sets of a profile, newest first; a store without the tables lists none."""
    sql, args = "SELECT * FROM proposal_sets WHERE profile_id = ?", [profile_id]
    if states:
        sql += " AND state IN (" + ",".join("?" for _ in states) + ")"
        args.extend(states)
    try:
        rows = _select(conn, sql + " ORDER BY created_at DESC LIMIT ?",
                       (*args, max(1, min(int(limit), 200))))
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return []
        raise
    return [_view(r, _members(conn, r["set_id"])) for r in rows]


def count_pending(conn: Any, profile_id: str) -> int:
    try:
        return int(_select(conn, "SELECT COUNT(*) AS n FROM proposal_sets WHERE profile_id = ? "
                           "AND state = 'proposed'", (profile_id,))[0]["n"])
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return 0
        raise


def _actor_for(basis: Basis) -> tuple[str, str, str]:
    if basis.kind == PERSON:
        return basis.actor, "host_authenticated", "trusted"
    return "slm-compiler", "system", "canonical_writer"


def _run_atomically(conn: Any, work) -> Any:
    name = "sp_" + uuid.uuid4().hex[:12]
    conn.execute(f"SAVEPOINT {name}")
    try:
        result = work()
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    conn.execute(f"RELEASE {name}")
    return result


def apply_set(conn: Any, *, profile_id: str, set_id: str, basis: Basis | None,
              flags: Any = None) -> dict:
    """Apply every proposal of a ``proposed`` set under ``basis`` (I-10), all or nothing."""
    load_optional_handlers()
    row = _set_row(conn, profile_id, set_id)
    members = _members(conn, set_id)
    check_basis(conn, profile_id, basis, [m["kind"] for m in members], flags)
    if row["state"] != PROPOSED:
        raise StateRefused(f"This set is {row['state']}, not waiting for a decision. "
                           "Nothing was changed.")
    actor_id, actor_kind, tier = _actor_for(basis)

    def work() -> None:
        now = _now()
        for m in members:
            ctx = ApplyContext(profile_id, set_id, m["proposal_id"], actor_id, actor_kind, tier)
            inverse = handler_for(m["kind"]).apply(conn, ctx, json.loads(m["payload_json"]))
            conn.execute("UPDATE proposals SET state = 'applied', inverse_json = ?, applied_at = ? "
                         "WHERE proposal_id = ?", (json.dumps(inverse, sort_keys=True), now,
                                                  m["proposal_id"]))
        conn.execute(
            "UPDATE proposal_sets SET state = 'applied', basis_kind = ?, basis_ref = ?, "
            "decided_by = ?, decided_channel = ?, decided_at = ?, applied_at = ? WHERE set_id = ?",
            (basis.kind, basis.ref, basis.actor or None, basis.channel or None, now, now, set_id))

    _run_atomically(conn, work)
    return get_set(conn, profile_id, set_id)


def reject_set(conn: Any, *, profile_id: str, set_id: str, actor: str, channel: str) -> dict:
    """Decline a ``proposed`` set. Changes no memory; needs a named person at a door."""
    row = _set_row(conn, profile_id, set_id)
    check_basis(conn, profile_id, Basis.person(actor, channel), [])
    if row["state"] != PROPOSED:
        raise StateRefused(f"This set is {row['state']}, not waiting for a decision.")
    load_optional_handlers()

    def work() -> None:
        now = _now()
        for m in _members(conn, set_id):
            hook = handler_for(m["kind"]).on_reject
            if hook is not None:  # a wrapped ledger case is declined in the ledger too
                hook(conn, ApplyContext(profile_id, set_id, m["proposal_id"], actor,
                                        "host_authenticated", "trusted"),
                     json.loads(m["payload_json"]))
        conn.execute("UPDATE proposals SET state = 'rejected' WHERE set_id = ?", (set_id,))
        conn.execute("UPDATE proposal_sets SET state = 'rejected', decided_by = ?, "
                     "decided_channel = ?, decided_at = ? WHERE set_id = ?",
                     (actor, channel, now, set_id))

    _run_atomically(conn, work)
    return get_set(conn, profile_id, set_id)


def rollback_set(conn: Any, *, profile_id: str, set_id: str, basis: Basis | None,
                 flags: Any = None) -> dict:
    """Undo an ``applied`` set exactly, restoring what each change replaced (newest first)."""
    load_optional_handlers()
    row = _set_row(conn, profile_id, set_id)
    members = _members(conn, set_id)
    check_basis(conn, profile_id, basis, [m["kind"] for m in members], flags)
    if row["state"] != APPLIED:
        raise StateRefused(f"This set is {row['state']}, not applied. Nothing was changed.")
    actor_id, actor_kind, tier = _actor_for(basis)

    def work() -> None:
        now = _now()
        for m in reversed(members):
            ctx = ApplyContext(profile_id, set_id, m["proposal_id"], actor_id, actor_kind, tier)
            handler_for(m["kind"]).rollback(conn, ctx, json.loads(m["payload_json"]),
                                            json.loads(m["inverse_json"] or "{}"))
            conn.execute("UPDATE proposals SET state = 'rolled_back', rolled_back_at = ? "
                         "WHERE proposal_id = ?", (now, m["proposal_id"]))
        conn.execute("UPDATE proposal_sets SET state = 'rolled_back', rolled_back_at = ? "
                     "WHERE set_id = ?", (now, set_id))

    _run_atomically(conn, work)
    return get_set(conn, profile_id, set_id)


def adopt_case(conn: Any, *, profile_id: str, case_id: str, origin: str = "ledger") -> str:
    """Wrap a correction case that still awaits review as a one-proposal set (kind ``retire``)."""
    rows = _select(conn, "SELECT status FROM correction_cases WHERE case_id = ? AND profile_id = ?",
                   (case_id, profile_id))
    if not rows:
        raise LookupError("no such correction case in this profile")
    if rows[0]["status"] != PROPOSED:
        raise StateRefused("Only a case still waiting for review can be adopted.")
    return create_set(conn, profile_id=profile_id, origin=origin, title="Correction awaiting review",
                      proposals=[{"kind": "retire", "payload": {"case_id": case_id}}])


def case_as_proposal(conn: Any, profile_id: str, case_id: str) -> dict:
    """A correction case in the shape of a ``retire`` proposal, with the basis it was decided on.

    Read-only. The ledger records who reviewed a case; it holds no certificate
    id, so a machine-decided case shows its actor and says so.
    """
    rows = _select(conn, "SELECT * FROM correction_cases WHERE case_id = ? AND profile_id = ?",
                   (case_id, profile_id))
    if not rows:
        raise LookupError("no such correction case in this profile")
    c = rows[0]
    reviewer = c["reviewed_by_actor_id"]
    basis = None
    if c["status"] in (APPLIED, ROLLED_BACK) and reviewer:
        machine = c["proposed_by_actor_kind"] in ("system", "host_attested")
        basis = {"kind": "recorded_before_certificates" if machine and reviewer.startswith("slm-")
                 else PERSON, "ref": reviewer}
    return {"kind": "retire", "state": c["status"], "case_id": case_id, "basis": basis,
            "payload": {"predecessor": c["predecessor_fact_id"],
                        "successor": c["successor_fact_id"], "reason": c["reason_code"]}}


__all__ = ["APPLIED", "Basis", "BasisRefused", "CERTIFICATE", "COVERS", "HANDLERS", "PERSON",
           "PERSON_CHANNELS", "PROPOSED", "ProposalError", "REJECTED", "ROLLED_BACK",
           "RULE_CERTIFICATE", "StateRefused", "adopt_case", "apply_set", "case_as_proposal",
           "check_basis", "count_pending", "create_set", "get_set", "list_sets", "reject_set",
           "rollback_set"]
