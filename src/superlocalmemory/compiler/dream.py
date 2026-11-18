# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The dream cycle: lint, propose, verify, queue for review (WS-6, D-09, D-18, D-27).

Once in a while, in the background, SLM looks over a profile and prepares what it
would change, and a person decides:

1. **lint**: exact duplicates (the same text after case, spacing and end punctuation
   are folded: rule ``exact-text-v1``), memories whose source file changed or went
   missing, and open disputes. Rules only: in 4.2.30 shadow means no hosted judge
   (D-27), and the DUPLICATE question has no certificate, so no model is asked.
2. **propose**: a page for every subject the store supports (``page_sources``) whose
   page is new or changed, and a ``merge`` for each group of exact duplicates, as
   *one* ProposalSet (a unit that applies, and rolls back, as a whole).
3. **verify**: every page sentence is checked against the store (the span check,
   ``page_templates.verify``) before it is queued; a sentence that fails is left
   out and counted in the page's withheld total. CITATION_SUPPORT by a model acts
   only with its own certificate, and none exists, so no sentence is model-written.
4. **review**: the set waits in the review queue (``slm proposals``, the dashboard).
   Applying it takes a person, or a certificate that covers it under
   ``compiler.auto_apply`` (off); nothing here applies anything.

It writes only proposal tables (D-27) and calls no model, so there is no model call
to hold a lock across; the one write is a short transaction at the end. A profile
with a dream set still waiting is not given another, one rejected is not proposed
again unchanged, and a new set is not made sooner than ``MIN_INTERVAL``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from superlocalmemory.compiler import page_sources, proposals
from superlocalmemory.compiler import page_templates as tpl
from superlocalmemory.compiler import pages as _pages  # noqa: F401  (registers create_page)
from superlocalmemory.compiler.page_text import UNVERIFIABLE
from superlocalmemory.compiler.proposal_kinds import ProposalError, handler_for
from superlocalmemory.compiler.versions import select_rows as _select

logger = logging.getLogger(__name__)

ORIGIN = "dream"
RULE = "exact-text-v1"
#: Routing only: how often a profile may get a new dream set, and how big one set is.
MIN_INTERVAL = timedelta(hours=6)
MAX_ITEMS_PER_SET = 50
MAX_DUPLICATE_GROUPS = 20
#: Shorter text is too thin to call a duplicate ("ok", "done").
MIN_DUPLICATE_CHARS = 20
_END = re.compile(r"[\s.!?;,:]+$")


def normal_text(text: str) -> str:
    """Text as the duplicate rule compares it: case, spacing and end punctuation folded."""
    return _END.sub("", " ".join((text or "").lower().split()))


# -- lint ----------------------------------------------------------------------------------------

def duplicate_groups(conn: Any, profile_id: str) -> list[tuple[str, list[str]]]:
    """``(keep, [retire, ...])`` for facts with the same normalised text, oldest kept.

    Same scope only, never a pinned memory or a fact whose source file changed.
    """
    rows = _select(
        conn, f"SELECT {tpl.FACT_COLUMNS}, f.scope AS scope FROM atomic_facts f "  # noqa: S608
        f"LEFT JOIN fact_temporal_validity v ON v.fact_id = f.fact_id WHERE {tpl.LIVE} "
        f"AND COALESCE(f.pinned, 0) = 0{tpl.hidden_clause(conn)} "
        "ORDER BY f.created_at, f.fact_id LIMIT ?", (profile_id, page_sources.SCAN_LIMIT))
    stale = page_sources.stale_fact_ids(conn, profile_id)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        text = normal_text(row["content"])
        if len(text) >= MIN_DUPLICATE_CHARS and row["fact_id"] not in stale:
            groups[(row["scope"] or "", text)].append(row["fact_id"])
    found = [(ids[0], ids[1:]) for ids in groups.values() if len(ids) > 1]
    return sorted(found)[:MAX_DUPLICATE_GROUPS]


def lint(conn: Any, profile_id: str, groups: list[tuple[str, list[str]]] | None = None) -> dict:
    """What the store needs, counted; no text."""
    groups = duplicate_groups(conn, profile_id) if groups is None else groups
    try:
        disputes = _select(conn, "SELECT COUNT(*) AS n FROM slot_disputes WHERE profile_id = ? "
                           "AND state = 'open'", (profile_id,))[0]["n"]
    except Exception as exc:  # noqa: BLE001 - a store before the slot tables
        if "no such table" not in str(exc).lower():
            raise
        disputes = 0
    return {"rule": RULE, "duplicate_groups": len(groups),
            "stale_sources": len(page_sources.stale_fact_ids(conn, profile_id)),
            "open_disputes": int(disputes)}


# -- what was already decided --------------------------------------------------------------------

def _declined(conn: Any, profile_id: str) -> tuple[set[tuple[str, str]], set[tuple[str, ...]]]:
    """Pages and merges a person already rejected, so they are not proposed again unchanged."""
    pages: set[tuple[str, str]] = set()
    merges: set[tuple[str, ...]] = set()
    rows = _select(conn, "SELECT p.kind, p.payload_json FROM proposals p JOIN proposal_sets s "
                   "ON s.set_id = p.set_id WHERE s.profile_id = ? AND s.origin = ? "
                   "AND p.state = 'rejected' ORDER BY p.created_at DESC LIMIT 500",
                   (profile_id, ORIGIN))
    for row in rows:
        payload = json.loads(row["payload_json"])
        if row["kind"] == "create_page":
            page = payload.get("page") or {}
            pages.add((page.get("page_id", ""), _declined_hash(page)))
        elif row["kind"] == "merge":
            merges.add((payload.get("keep", ""), *sorted(payload.get("retire") or [])))
    return pages, merges


def _declined_hash(page: dict) -> str:
    from superlocalmemory.compiler.page_text import content_hash

    return content_hash(str(page.get("title") or ""), page.get("sentences") or [])


def _waiting_or_recent(conn: Any, profile_id: str, now: datetime) -> str | None:
    waiting = _select(conn, "SELECT COUNT(*) AS n FROM proposal_sets WHERE profile_id = ? "
                      "AND origin = ? AND state = 'proposed'", (profile_id, ORIGIN))[0]["n"]
    if waiting:
        return "a dream set is still waiting for review"
    last = _select(conn, "SELECT MAX(created_at) AS at FROM proposal_sets WHERE profile_id = ? "
                   "AND origin = ?", (profile_id, ORIGIN))[0]["at"]
    if last and now - datetime.fromisoformat(last) < MIN_INTERVAL:
        return "the last dream set is recent"
    return None


# -- propose and verify --------------------------------------------------------------------------

def _verified(conn: Any, profile_id: str, draft: dict) -> dict | None:
    """The draft with only the sentences that pass the span check, or ``None`` if none do."""
    kept, dropped = tpl.verify_all(conn, profile_id, draft["sentences"])
    if not kept:
        return None
    withheld = dict(draft["withheld"])
    if dropped:
        withheld[UNVERIFIABLE] = withheld.get(UNVERIFIABLE, 0) + dropped
    from superlocalmemory.compiler.page_text import content_hash, renumber

    kept = renumber(kept)
    return {**draft, "sentences": kept, "withheld": withheld,
            "content_hash": content_hash(draft["title"], kept)}


def _stored_hashes(conn: Any, profile_id: str) -> dict[str, tuple[str, str]]:
    return {r["page_id"]: (r["content_hash"], r["withheld_json"]) for r in _select(
        conn, "SELECT page_id, content_hash, withheld_json FROM compiled_pages "
        "WHERE profile_id = ?", (profile_id,))}


def plan(conn: Any, profile_id: str, *, merges: bool = True) -> dict:
    """What a dream set would hold now: ``{"items": [...], "report": {...}}``. Reads only.

    ``merges=False`` plans pages only (a person's rebuild of the wiki)."""
    groups = duplicate_groups(conn, profile_id) if merges else []
    report = lint(conn, profile_id, groups)
    declined_pages, declined_merges = _declined(conn, profile_id)
    stored = _stored_hashes(conn, profile_id)
    items: list[dict] = []
    withheld_total = 0
    for draft in page_sources.build_drafts(conn, profile_id):
        checked = _verified(conn, profile_id, draft)
        if checked is None:
            continue
        key = (checked["page_id"], checked["content_hash"])
        same = stored.get(checked["page_id"], ("", ""))[0] == checked["content_hash"]
        if same or key in declined_pages:
            continue
        withheld_total += sum(checked["withheld"].values())
        items.append({"kind": "create_page", "payload": {"page": checked}})
    for keep, retire in groups:
        if (keep, *sorted(retire)) not in declined_merges:
            items.append({"kind": "merge", "payload": {"keep": keep, "retire": retire,
                                                       "rule": RULE}})
    lawful = []
    for item in items:
        try:
            handler_for(item["kind"]).validate(conn, profile_id, item["payload"])
        except ProposalError as exc:
            logger.debug("dream: %s skipped (%s)", item["kind"], exc)
            continue
        lawful.append(item)
    report.update(pages=sum(1 for i in lawful if i["kind"] == "create_page"),
                  merges=sum(1 for i in lawful if i["kind"] == "merge"),
                  withheld_sentences=withheld_total)
    return {"items": lawful[:MAX_ITEMS_PER_SET], "report": report}


def run_profile(db: Any, profile_id: str, *, now: datetime | None = None) -> dict:
    """One dream cycle for one profile: plan on a read-only connection, queue in one short write.

    Returns the report, with ``set_id`` when a set was queued and ``skipped`` (why) when not.
    """
    from superlocalmemory.truth.invariants import open_read_only

    now = now or datetime.now(UTC)
    reader = open_read_only(str(db.db_path))
    try:
        why = _waiting_or_recent(reader, profile_id, now)
        if why:
            return {"profile_id": profile_id, "skipped": why}
        planned = plan(reader, profile_id)
    finally:
        reader.close()
    report = {"profile_id": profile_id, **planned["report"]}
    if not planned["items"]:
        return {**report, "skipped": "nothing to propose"}
    with db.raw_connection() as conn:
        shown = planned["report"]
        set_id = proposals.create_set(
            conn, profile_id=profile_id, origin=ORIGIN, proposals=planned["items"],
            title=f"Dream cycle: {shown['pages']} page(s), {shown['merges']} duplicate group(s)")
    logger.info("dream cycle: queued %d proposals for review", len(planned["items"]))
    return {**report, "set_id": set_id, "proposals": len(planned["items"])}


def maintenance_step(db: Any, profile_ids: tuple[str, ...]) -> list[dict]:
    """The scheduler's share: one cycle per profile, only while ``compiler.dream`` is on."""
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    if not FeatureFlags(canonical_data_root()).enabled("compiler.dream"):
        return []
    return [run_profile(db, pid) for pid in profile_ids]


__all__ = ["MAX_ITEMS_PER_SET", "MIN_INTERVAL", "ORIGIN", "RULE", "duplicate_groups", "lint",
           "maintenance_step", "normal_text", "plan", "run_profile"]
