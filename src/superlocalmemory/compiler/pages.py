# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Compiled pages: the ``create_page`` proposal kind, and reading a page back (WS-6, D-18).

A page enters the store only as a ProposalSet of kind ``create_page`` (I-10): applying
it needs a person, or a CITATION_SUPPORT certificate, and none exists in 4.2.30, so
in practice a person. Applying re-checks every sentence against the store as it is
*now* (a fact may have changed or gone since the page was drafted); a sentence that
no longer passes is dropped and counted as withheld, and a page with nothing left is
refused. The page's previous row, if any, is kept as a version first (versions
first), and the inverse of the write is recorded, so rolling the set back restores
the earlier page exactly, or removes a page that was new.

Reading a page checks its sentences again (:func:`show_page`): the page a reader
sees never contains a sentence that is not verified at that moment, and it says how
many were withheld, now or when it was built. Pages carry origin ``derived`` (a CHECK
in M083) and are never written to the fact tables.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.compiler import page_templates as tpl
from superlocalmemory.compiler import versions
from superlocalmemory.compiler.page_text import (
    KINDS,
    MAX_SENTENCES,
    REASON_WORDS,
    UNVERIFIABLE,
    content_hash,
    page_id,
    renumber,
)
from superlocalmemory.compiler.proposal_kinds import (
    ApplyContext,
    Handler,
    ProposalError,
    register_handler,
)
from superlocalmemory.compiler.versions import select_rows as _select

ARTIFACT = "compiled_page"
_KEYS = {"seq", "type", "text", "facts", "template", "params"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


# -- the create_page proposal kind ---------------------------------------------------------------

def _page(payload: Mapping[str, Any]) -> dict:
    page = payload.get("page")
    if not isinstance(page, dict):
        raise ProposalError("A page proposal carries the page.")
    return page


def _validate(conn: Any, profile_id: str, payload: dict) -> list[str]:
    page = _page(payload)
    kind, subject = page.get("kind"), str(page.get("subject_key") or "")
    sentences = page.get("sentences")
    if kind not in KINDS or not subject or not str(page.get("title") or "").strip():
        raise ProposalError("A page has a kind, a subject and a title.")
    if page.get("page_id") != page_id(profile_id, kind, subject):
        raise ProposalError("The page id does not match its profile, kind and subject.")
    if (not isinstance(sentences, list) or not sentences or len(sentences) > MAX_SENTENCES
            or any(not isinstance(s, dict) or not _KEYS <= set(s) for s in sentences)):
        raise ProposalError("A page has between one and %d well-formed sentences." % MAX_SENTENCES)
    cited = sorted({str(f) for s in sentences for f in s["facts"]})
    found = {r["fact_id"] for r in _select(
        conn, "SELECT fact_id FROM atomic_facts WHERE profile_id = ? AND fact_id IN ("
        + ",".join("?" for _ in cited) + ")", (profile_id, *cited))} if cited else set()
    if not cited or set(cited) - found:
        raise ProposalError("Every sentence cites memories of this profile.")
    return cited


def _existing(conn: Any, profile_id: str, pid: str) -> dict | None:
    rows = _select(conn, "SELECT * FROM compiled_pages WHERE page_id = ? AND profile_id = ?",
                   (pid, profile_id))
    return rows[0] if rows else None


def _describe(conn: Any, profile_id: str, payload: dict) -> str:
    page = _page(payload)
    withheld = sum((page.get("withheld") or {}).values())
    verb = "Update" if _existing(conn, profile_id, page["page_id"]) else "Add"
    return (f"{verb} the {page['kind'].replace('_', ' ')} page “{page['title']}”: "
            f"{len(page['sentences'])} sentences, {withheld} withheld")


def _write_citations(conn: Any, pid: str, sentences: list[dict]) -> None:
    conn.execute("DELETE FROM page_citations WHERE page_id = ?", (pid,))
    for s in sentences:
        for fact_id in s["facts"]:
            conn.execute("INSERT OR IGNORE INTO page_citations (page_id, seq, fact_id) "
                         "VALUES (?, ?, ?)", (pid, s["seq"], fact_id))


def _apply(conn: Any, ctx: ApplyContext, payload: dict) -> dict:
    page, profile = _page(payload), ctx.profile_id
    kept, dropped = tpl.verify_all(conn, profile, page["sentences"])
    if not kept:
        raise ProposalError("None of the page's sentences can be verified against the store "
                            "now; it changed since the page was drafted. Nothing was changed.")
    sentences = renumber(kept)
    withheld = dict(page.get("withheld") or {})
    if dropped:
        withheld[UNVERIFIABLE] = withheld.get(UNVERIFIABLE, 0) + dropped
    digest = content_hash(page["title"], sentences)
    pid, before = page["page_id"], _existing(conn, profile, page["page_id"])
    withheld_json = json.dumps(withheld, sort_keys=True)
    if before and before["content_hash"] == digest and before["withheld_json"] == withheld_json:
        return {"mode": "unchanged", "page_id": pid}
    fields = (page["title"], json.dumps(sentences, sort_keys=True), digest, len(sentences),
              sum(withheld.values()), withheld_json, _now(), ctx.set_id)
    if before is None:
        conn.execute(
            "INSERT INTO compiled_pages (page_id, profile_id, kind, subject_key, title, "
            "body_json, content_hash, shown_count, withheld_count, withheld_json, built_at, "
            "set_id, version, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'derived')",
            (pid, profile, page["kind"], page["subject_key"], *fields))
        inverse: dict[str, Any] = {"mode": "create", "page_id": pid, "citations_before": []}
    else:
        key = {"profile_id": profile, "page_id": pid}
        versions.keep_previous(conn, ARTIFACT, key, reason=versions.EDIT, proposal_id=ctx.proposal_id)
        conn.execute(
            "UPDATE compiled_pages SET title = ?, body_json = ?, content_hash = ?, "
            "shown_count = ?, withheld_count = ?, withheld_json = ?, built_at = ?, set_id = ?, "
            "version = version + 1 WHERE page_id = ? AND profile_id = ?", (*fields, pid, profile))
        inverse = {"mode": "replace", "page_id": pid, "before": versions._record(before),
                   "citations_before": [[r["seq"], r["fact_id"]] for r in _select(
                       conn, "SELECT seq, fact_id FROM page_citations WHERE page_id = ? "
                       "ORDER BY seq, fact_id", (pid,))]}
    _write_citations(conn, pid, sentences)
    inverse["after_hash"] = digest
    return inverse


def _rollback(conn: Any, ctx: ApplyContext, payload: dict, inverse: dict) -> None:
    mode, pid = inverse.get("mode"), inverse.get("page_id")
    if mode == "unchanged":
        return
    now = _existing(conn, ctx.profile_id, str(pid))
    if now is None or now["content_hash"] != inverse.get("after_hash"):
        raise ProposalError("This page was rebuilt after the proposal was applied, so it was "
                            "not rolled back. The earlier page is kept as a version.")
    if mode == "create":
        conn.execute("DELETE FROM compiled_pages WHERE page_id = ? AND profile_id = ?",
                     (pid, ctx.profile_id))
        conn.execute("DELETE FROM page_citations WHERE page_id = ?", (pid,))
        return
    versions.write_row(conn, versions.spec_for(ARTIFACT), inverse["before"])
    conn.execute("DELETE FROM page_citations WHERE page_id = ?", (pid,))
    for seq, fact_id in inverse.get("citations_before") or []:
        conn.execute("INSERT INTO page_citations (page_id, seq, fact_id) VALUES (?, ?, ?)",
                     (pid, seq, fact_id))


register_handler("create_page", Handler(_validate, _describe, _apply, _rollback))


# -- reading -------------------------------------------------------------------------------------

def list_pages(conn: Any, profile_id: str, *, kind: str = "") -> list[dict]:
    """Pages of a profile (no sentence text): what the dashboard's list shows."""
    sql, args = ("SELECT page_id, kind, subject_key, title, version, shown_count, withheld_count, "
                 "built_at FROM compiled_pages WHERE profile_id = ?"), [profile_id]
    if kind:
        sql, args = sql + " AND kind = ?", [*args, kind]
    try:
        return _select(conn, sql + " ORDER BY kind, title, page_id", args)
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return []
        raise


def _origin_labels(conn: Any, fact_ids: list[str]) -> dict[str, str]:
    if not fact_ids:
        return {}
    try:
        from superlocalmemory.trust import origin_class as oc

        rows = _select(conn, "SELECT fact_id, origin_class FROM fact_trust WHERE fact_id IN ("
                       + ",".join("?" for _ in fact_ids) + ")", tuple(fact_ids))
        return {r["fact_id"]: oc.label(r["origin_class"]) for r in rows}
    except Exception:  # noqa: BLE001 - a store before M079 shows no origin
        return {}


def show_page(conn: Any, profile_id: str, pid: str) -> dict | None:
    """One page as a reader sees it: only sentences that pass the span check *now*.

    ``withheld`` adds the sentences that failed just now to those withheld when the
    page was built; ``stale`` is true when any did, meaning the next dream cycle
    proposes the page again. Each sentence names the memories it cites with their
    ``memory_id`` so a reader can follow it. Read-only. ``None`` for no such page.
    """
    row = _existing(conn, profile_id, pid)
    if row is None:
        return None
    stored = json.loads(row["body_json"])
    cited = sorted({f for s in stored for f in s["facts"]})
    live = tpl.live_facts(conn, profile_id, cited)
    origins = _origin_labels(conn, cited)
    shown, failed = [], 0
    for s in stored:
        if tpl.verify(conn, profile_id, s, live)[0]:
            shown.append({**s, "citations": [
                {"fact_id": f, "memory_id": live[f]["memory_id"], "origin": origins.get(f)}
                for f in s["facts"] if f in live]})
        else:
            failed += 1
    reasons = json.loads(row["withheld_json"] or "{}")
    if failed:
        reasons[UNVERIFIABLE] = reasons.get(UNVERIFIABLE, 0) + failed
    return {
        "page_id": row["page_id"], "kind": row["kind"], "subject_key": row["subject_key"],
        "title": row["title"], "origin": row["origin"], "version": row["version"],
        "built_at": row["built_at"], "sentences": shown, "stale": bool(failed),
        "withheld": {"count": sum(reasons.values()),
                     "reasons": {k: {"count": v, "why": REASON_WORDS.get(k, k)}
                                 for k, v in sorted(reasons.items()) if v}},
        "notice": ("Every sentence is a word-for-word span of a memory it cites, or a fixed "
                   "template over its values. No model wrote any of it."),
    }


def status(conn: Any, profile_id: str) -> dict:
    """Counts for ``slm wiki status`` and the dashboard card. Read-only."""
    try:
        by_kind = {r["kind"]: r["n"] for r in _select(
            conn, "SELECT kind, COUNT(*) AS n FROM compiled_pages WHERE profile_id = ? "
            "GROUP BY kind", (profile_id,))}
        pending = _select(conn, "SELECT COUNT(DISTINCT s.set_id) AS n FROM proposal_sets s JOIN "
                          "proposals p ON p.set_id = s.set_id WHERE s.profile_id = ? AND "
                          "s.state = 'proposed' AND p.kind = 'create_page'", (profile_id,))[0]["n"]
    except Exception as exc:  # noqa: BLE001
        if "no such table" in str(exc).lower():
            return {"pages": 0, "by_kind": {}, "pending_sets": 0}
        raise
    return {"pages": sum(by_kind.values()), "by_kind": by_kind, "pending_sets": int(pending)}


__all__ = ["ARTIFACT", "list_pages", "show_page", "status"]
