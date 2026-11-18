# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm why <fact_id>``: everything that explains one memory (WS-5).

One read-only answer for a person or an agent that asks "where did this come from, can I
trust it, and why was it shown to me":

* **origin**: the class and taint, the raw signals they were computed from (channel,
  transform, who saved it, which door), the mapping row that decided the class and why that
  row exists, and the parents that lowered the taint (``trust/fact_trust``, ``trust/origin_class``);
* **firewall**: whether the instruction firewall would hold it back, and the approval
  records (who asked for what, who decided, through which door; ``trust_approvals``);
* **signature**: signed and unchanged, modified, signed under another key, or unsigned
  (``trust/signatures``); checked only if the caller holds the key;
* **truth**: its slots and their values over time, open disputes and the state of the file
  it came from (``truth/fact_truth``, reused as it is);
* **compiler**: versions of derived artifacts built from it, proposals that touch it and
  compiled pages that cite it (``compiler``: ids, kinds, states, a page's id, kind and title;
  never the text of a proposal or a sentence);
* **deliveries**: the last N times it was pushed or pulled into an agent's context, with each
  delivery's reason in words (path, role, rank, whether withholding or an exclusion left it
  out). Ids, numbers and fixed words from the receipts, never query text.

Reads never write and never touch the keychain: ``signer`` is whatever the caller already
holds. A store that lacks a table (before the migration that adds it) leaves that section
empty. The only memory text returned is a short ``preview`` of the memory asked about (the
caller has the right to read it), and what ``truth`` already shows with its visibility filter.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from typing import Any

from superlocalmemory.trust import fact_trust, signatures
from superlocalmemory.trust import origin_class as oc

PREVIEW_CHARS = 240
#: How many recent deliveries are listed (routing-only: a page size).
DELIVERIES = 10
LISTED = 10
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")

_PATH_WORDS = {
    "session_init": "at session start", "session_context": "at session start",
    "hook_recall": "with a prompt (automatic recall)",
    "mcp_recall": "for an agent's recall", "cli_recall": "for a recall at the command line",
    "http_recall": "for a recall over HTTP", "view_recall": "for a recall from the dashboard",
    "cue": "on a cue (a file, command, error, repository or date)",
}
_ROLE_WORDS = {"pinned": "as a pinned memory", "standing": "as a standing rule",
               "core": "as a core block", "recall": "as a recall result",
               "recent": "as a recent memory", "session_summary": "as a session summary"}
_RANKED = frozenset({"recall", "recent", "session_summary"})


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    """Rows, or none when the store predates the table (a section is then empty)."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) or "no such column" in str(exc):
            return []
        raise


def _preview(text: str) -> str:
    return " ".join(_CONTROL.sub(" ", text or "").split())[:PREVIEW_CHARS]


def _rule(basis: str) -> dict[str, str] | None:
    """The mapping row behind a class: its id and why it exists. ``None`` for a person's
    approval or a fact with no origin row."""
    head = basis.split(":", 1)[-1] if basis.startswith(oc.VERSION + ":") else ""
    row_id = head.split("+", 1)[0]
    for row in oc.ROWS:
        if row.row_id == row_id:
            return {"row": row.row_id, "why": row.why, "mapping": oc.VERSION,
                    "capped_by_transform": head.split("+", 1)[1] if "+" in head else ""}
    return None


def _origin(conn: sqlite3.Connection, fact_id: str) -> dict[str, Any]:
    # A store from before origins were kept (or without the table) holds no class: unknown.
    found = fact_trust.origin_of(conn, fact_id) or fact_trust.Trust(
        oc.UNKNOWN, oc.UNKNOWN, f"{oc.VERSION}:no_origin_row").as_dict()
    raw_rows = _rows(
        conn, "SELECT channel, transform, source_type, actor_id, agent_id, agent_id_source, "
        "speaker, role, parent_memory_id, parent_fact_id, recorded_at FROM fact_origin "
        "WHERE fact_id = ?", (fact_id,))
    names = ("channel", "transform", "source_type", "actor_id", "agent_id", "agent_id_source",
             "speaker", "role", "parent_memory_id", "parent_fact_id", "recorded_at")
    raw = dict(zip(names, raw_rows[0], strict=True)) if raw_rows else None
    node = _safe_parents(conn, fact_id)
    parents = []
    for parent in node:
        trust = fact_trust.origin_of(conn, parent)
        if trust:
            parents.append({"fact_id": parent, "class": trust["class"], "taint": trust["taint"],
                            "label": trust["label"]})
    taint, own = found.get("taint"), found.get("class")
    if taint == own:
        words = f"This memory is {found['label']}."
    else:
        words = (f"This memory is {found['label']}, but it was derived from something that is "
                 f"{found['taint_label']}, so it counts as {found['taint_label']}.")
    return {**{k: v for k, v in found.items() if k != "basis"}, "basis": found.get("basis", ""),
            "rule": _rule(found.get("basis", "")), "signals": raw, "parents": parents,
            "words": words}


def _safe_parents(conn: sqlite3.Connection, fact_id: str) -> list[str]:
    try:
        node = fact_trust._node(fact_trust.on_conn(conn), fact_id)
    except sqlite3.Error:
        return []
    return list(dict.fromkeys(node[1]))[:LISTED] if node else []


def _firewall(conn: sqlite3.Connection, fact_id: str, taint: str | None) -> dict[str, Any]:
    from superlocalmemory.trust import firewall

    approvals = [
        {"wanted": r[0], "state": r[1], "requested_by": r[2], "requested_channel": r[3],
         "requested_at": r[4], "decided_by": r[5], "decided_channel": r[6], "decided_at": r[7]}
        for r in _rows(conn, "SELECT wanted, state, requested_by, requested_channel, requested_at, "
                       "decided_by, decided_channel, decided_at FROM trust_approvals "
                       "WHERE fact_id = ? ORDER BY requested_at", (fact_id,))]
    acting = firewall.active()
    below = not firewall.is_trusted_taint(taint)
    if below and acting:
        words = ("The instruction firewall holds this memory back from being a pin, a standing "
                 "rule, core memory or part of the session start until you approve it. It can "
                 "still be recalled as evidence.")
    elif below:
        words = ("It would be held back from being a pin, a rule or core memory, but the "
                 "instruction firewall is not acting (trust.taint is not on).")
    else:
        words = "Trusted enough to be an instruction."
    return {"acting": acting, "below_system": below, "approvals": approvals, "words": words}


def _compiler(conn: sqlite3.Connection, fact_id: str) -> dict[str, list[dict]]:
    versions = [{"version_id": r[0], "artifact": r[1], "seq": r[2], "reason": r[3],
                 "proposal_id": r[4], "created_at": r[5]}
                for r in _rows(
                    conn, "SELECT v.version_id, v.artifact, v.seq, v.reason, v.proposal_id, "
                    "v.created_at FROM artifact_version_facts l JOIN artifact_versions v "
                    "ON v.version_id = l.version_id WHERE l.fact_id = ? "
                    "ORDER BY v.created_at DESC LIMIT ?", (fact_id, LISTED))]
    proposals = [{"proposal_id": r[0], "set_id": r[1], "kind": r[2], "state": r[3],
                  "set_state": r[4], "created_at": r[5]}
                 for r in _rows(
                     conn, "SELECT p.proposal_id, p.set_id, p.kind, p.state, s.state, p.created_at "
                     "FROM proposal_facts l JOIN proposals p ON p.proposal_id = l.proposal_id "
                     "JOIN proposal_sets s ON s.set_id = p.set_id WHERE l.fact_id = ? "
                     "ORDER BY p.created_at DESC LIMIT ?", (fact_id, LISTED))]
    pages = [{"page_id": r[0], "kind": r[1], "title": r[2]}
             for r in _rows(
                 conn, "SELECT DISTINCT p.page_id, p.kind, p.title FROM page_citations c JOIN "
                 "compiled_pages p ON p.page_id = c.page_id WHERE c.fact_id = ? "
                 "ORDER BY p.kind, p.title LIMIT ?", (fact_id, LISTED))]
    return {"versions": versions, "proposals": proposals, "pages": pages}


def _delivery_words(path: str, role: str, position: int, delivered: int, included: int,
                    exclusion: str | None) -> str:
    where = _PATH_WORDS.get(path, f"by {path}" if path else "to an agent")
    how = _ROLE_WORDS.get(role, f"as {role}" if role else "as a result")
    if role in _RANKED or role not in _ROLE_WORDS:
        how += f" (rank {int(position) + 1})"
    if delivered:
        return f"Shown {where}, {how}."
    if not included:
        return (f"Chosen {where}, {how}, but left out on purpose: a randomised check of whether "
                "memory helps (it can be switched off in Memory Health).")
    return f"Considered {where}, {how}, but it did not reach the agent's text."


def _deliveries(conn: sqlite3.Connection, fact_id: str, limit: int) -> dict[str, Any]:
    rows = _rows(
        conn, "SELECT d.delivery_id, d.path, d.client, d.session_id, d.delivered_at, "
        "d.holdout_state, d.arm, r.role, r.position, r.score, r.eligible, r.exclusion, "
        "r.propensity, r.included, r.delivered FROM injection_receipts r "
        "JOIN injection_deliveries d ON d.delivery_id = r.delivery_id "
        "WHERE r.fact_id = ? ORDER BY d.delivered_at DESC LIMIT ?", (fact_id, int(limit)))
    items = []
    for (did, path, client, session, at, holdout, arm, role, pos, score, eligible, exclusion,
         propensity, included, delivered) in rows:
        items.append({
            "delivery_id": did, "at": at, "path": path, "client": client, "session_id": session,
            "holdout_state": holdout, "arm": arm,
            "why": {"role": role, "rank": int(pos) + 1,
                    "score": None if score is None else round(float(score), 4),
                    "eligible_for_withholding": bool(eligible), "exclusion": exclusion,
                    "propensity": propensity, "included": bool(included),
                    "delivered": bool(delivered),
                    "words": _delivery_words(path, role, pos, delivered, included, exclusion)}})
    totals = _rows(conn, "SELECT COUNT(*), COALESCE(SUM(delivered), 0) FROM injection_receipts "
                   "WHERE fact_id = ?", (fact_id,))
    seen, shown = (int(totals[0][0]), int(totals[0][1])) if totals else (0, 0)
    return {"shown": shown, "considered": seen, "recent": items}


def explain(conn: sqlite3.Connection, fact_id: str, *, profile_id: str | None = None,
            signer: Any = None, deliveries: int = DELIVERIES,
            hidden: Callable[[list[str]], set[str]] = lambda ids: set()) -> dict[str, Any] | None:
    """Everything that explains ``fact_id``; ``None`` when there is no such memory (in ``profile_id``
    when given). Read-only."""
    rows = _rows(conn, "SELECT profile_id, content, created_at, memory_id FROM atomic_facts "
                 "WHERE fact_id = ?", (fact_id,))
    if not rows or (profile_id is not None and rows[0][0] != profile_id):
        return None
    profile, content, created_at, memory_id = rows[0]
    origin = _origin(conn, fact_id)
    from superlocalmemory.truth import fact_truth

    truth = fact_truth.for_fact(conn, profile, fact_id, hidden=hidden)
    return {
        "fact_id": fact_id, "profile_id": profile, "memory_id": memory_id,
        "created_at": created_at, "preview": _preview(content),
        "origin": origin,
        "firewall": _firewall(conn, fact_id, origin.get("taint")),
        "signature": signatures.for_fact(conn, fact_id, signer),
        "truth": {k: truth[k] for k in ("slots", "disputes", "source")},
        "compiler": _compiler(conn, fact_id),
        "deliveries": _deliveries(conn, fact_id, deliveries),
    }


#: Most memories one ``fetch(why=true)`` explains (routing-only: keeps one call small).
MAX_ATTACHED = 10


def attach(db_path: Any, profile_id: str, items: list[dict[str, Any]], *,
           signer: Any = None) -> list[dict[str, Any]]:
    """``items`` (each with a ``fact_id``) with a ``why`` report added, for ``fetch(why=true)``.

    New dicts; the first :data:`MAX_ATTACHED` are explained and the rest carry
    ``why: {"skipped": "limit"}``. One read-only connection; any failure leaves that item with
    ``why: {"error": ...}`` and never fails the fetch.
    """
    from superlocalmemory.truth.invariants import open_read_only

    out: list[dict[str, Any]] = []
    try:
        conn = open_read_only(str(db_path))
    except sqlite3.Error as exc:
        return [{**i, "why": {"error": f"store not readable ({type(exc).__name__})"}} for i in items]
    try:
        for position, item in enumerate(items):
            if position >= MAX_ATTACHED:
                out.append({**item, "why": {"skipped": "limit"}})
                continue
            try:
                report = explain(conn, str(item.get("fact_id", "")), profile_id=profile_id,
                                 signer=signer, deliveries=3)
                out.append({**item, "why": report if report is not None else {"error": "not found"}})
            except Exception as exc:  # noqa: BLE001 - an explanation never fails a fetch
                out.append({**item, "why": {"error": type(exc).__name__}})
    finally:
        conn.close()
    return out


def lines(report: dict[str, Any]) -> list[str]:
    """The text ``slm why`` prints."""
    out = [f"{report['fact_id']}  (saved {report['created_at']})", f"  {report['preview']}", "",
           "Where it came from", f"  {report['origin']['words']}"]
    rule = report["origin"].get("rule")
    if rule:
        out.append(f"  Rule {rule['mapping']}:{rule['row']}: {rule['why']}")
    for p in report["origin"]["parents"]:
        out.append(f"  Derived from {p['fact_id']}: {p['label']}")
    out += ["", "Instruction firewall", f"  {report['firewall']['words']}"]
    for a in report["firewall"]["approvals"]:
        how = (f"{a['state']} by {a['decided_by']} via {a['decided_channel']}"
               if a["decided_by"] else a["state"])
        out.append(f"  It asked to be: {a['wanted']} ({how})")
    out += ["", "Signature", f"  {report['signature']['words']}"]
    truth = report["truth"]
    out += ["", "Truth",
            f"  {len(truth['slots'])} slot(s), {len(truth['disputes'])} open dispute(s)."]
    if truth.get("source"):
        out.append(f"  Source file: {truth['source'].get('state', 'unknown')}")
    comp = report["compiler"]
    if comp["versions"] or comp["proposals"] or comp["pages"]:
        out += ["", "Derived work",
                f"  {len(comp['versions'])} version(s) of derived artifacts, "
                f"{len(comp['proposals'])} proposal(s) and {len(comp['pages'])} page(s) mention it."]
    d = report["deliveries"]
    out += ["", f"Shown to your agents: {d['shown']} time(s) of {d['considered']} considered"]
    out += [f"  {i['at']}  {i['why']['words']}" for i in d["recent"]]
    return out


__all__ = ["DELIVERIES", "MAX_ATTACHED", "attach", "explain", "lines"]
