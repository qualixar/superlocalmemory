# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The Trust section of Memory Health (WS-5, D-49): origins, what is held back until promoted,
the decisions callers made, secrets held back.

Read-only and fail-open: a store missing a table reports that part as empty. Counts only:
no memory text and never the value of a credential. The count of credentials is taken from
the typed placeholders in what is stored (``[REDACTED:TYPE:LAST4]``), so it says what the
store holds *now*, by type; it does not remember what was screened and then erased.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path

from superlocalmemory.core import decisions
from superlocalmemory.trust import firewall, secret_screen
from superlocalmemory.trust import origin_class as oc

logger = logging.getLogger(__name__)

#: Rows of stored text inspected for placeholders per table (routing only: it bounds the read).
SCAN_ROWS = 50_000
#: Waiting memories listed in the report itself (the full list is ``slm trust pending``).
WAITING_SHOWN = 10


def _states(slm_home: Path | None) -> dict[str, str]:
    from superlocalmemory.core.feature_flags import FeatureFlags

    flags = FeatureFlags(slm_home)
    return {"taint": flags.state("trust.taint").value,
            "secret_screen": flags.state("trust.secret_screen").value}


def placeholders(conn: sqlite3.Connection, profile_id: str) -> dict:
    """Credentials held as typed placeholders, by type: ``{"by_type", "memories", "capped"}``."""
    kinds: Counter = Counter()
    holding = 0
    capped = False
    for table in ("memories", "atomic_facts"):
        try:
            rows = conn.execute(
                f"SELECT content FROM {table} WHERE profile_id = ? "  # noqa: S608 - fixed names
                "AND content LIKE '%[REDACTED:%' LIMIT ?", (profile_id, SCAN_ROWS + 1)).fetchall()
        except sqlite3.OperationalError:
            continue
        capped = capped or len(rows) > SCAN_ROWS
        found = [secret_screen.types_in(r[0]) for r in rows[:SCAN_ROWS]]
        if table == "memories":  # a memory is the saved text; its facts repeat it
            holding = sum(1 for f in found if f)
            for f in found:
                kinds += f
    return {"by_type": dict(sorted(kinds.items())), "memories": holding, "capped": capped}


def origins(conn: sqlite3.Connection, profile_id: str) -> dict[str, int]:
    """How many facts have each taint class (``fact_trust``); empty before M079."""
    try:
        rows = conn.execute("SELECT taint_class, COUNT(*) FROM fact_trust WHERE profile_id = ? "
                            "GROUP BY taint_class", (profile_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    # rank 0 (user_direct) is falsy: compare with None, or the most trusted class sorts last
    def position(row: tuple) -> int:
        rank = oc.rank_of(row[0])
        return 99 if rank is None else rank

    return {name: int(n) for name, n in sorted(rows, key=position)}


def build(conn: sqlite3.Connection, profile_id: str, *, slm_home: Path | None = None) -> dict:
    """The Trust section (JSON-ready); ``{}`` only if nothing at all could be read."""
    out: dict = {}
    try:
        out.update(_states(slm_home))
        counts = firewall.counts(conn, profile_id)
        out["waiting"] = counts["waiting"]
        out["approved"] = counts["approved"]
        queue = firewall.pending(conn, profile_id, WAITING_SHOWN)
        out["unclassified"] = queue["unclassified"]
        # Which memories are held back and why; any caller may promote one on purpose.
        out["waiting_items"] = [
            {"fact_id": i["fact_id"], "origin": i["origin_class"], "wanted": i["wanted"],
             "preview": i["preview"][:120]} for i in queue["items"]]
        out["origins"] = origins(conn, profile_id)
        out["secrets_held_back"] = placeholders(conn, profile_id)
        out["decisions"] = decisions.counts(conn, profile_id)
    except Exception:  # noqa: BLE001 - this section never breaks the report
        logger.debug("trust section unreadable", exc_info=True)
    return out


def describe(section: dict) -> list[str]:
    """The section as short lines for a terminal (no values, no memory text)."""
    if not section:
        return []
    lines = [f"Trust: instruction firewall {section.get('taint', 'unknown')}, secret screen "
             f"{section.get('secret_screen', 'unknown')}"]
    if section.get("taint") == "off" and section.get("waiting"):
        # Nothing is held while the firewall is off: say what it would hold.
        lines.append(f"  Would be held back if the firewall were on: {section['waiting']}; "
                     f"promoted: {section.get('approved', 0)}")
    elif section.get("waiting") or section.get("approved"):
        lines.append(f"  Held back until promoted: {section.get('waiting', 0)}; promoted: "
                     f"{section.get('approved', 0)}")
    if section.get("waiting"):
        lines.append("  See them with: slm trust pending. Promote one on purpose with: "
                     "slm trust promote FACT_ID (recorded, and it can be withdrawn)")
    if section.get("unclassified"):
        lines.append(f"  Not yet classified by origin: {section['unclassified']}")
    made = section.get("decisions") or {}
    if made.get("total"):
        lines.append(f"  Decisions by your agents: {made['total']} recorded, {made.get('undone', 0)} "
                     "undone (the dashboard lists them with an Undo)")
    held = section.get("secrets_held_back") or {}
    if held.get("by_type"):
        kinds = ", ".join(f"{k} {n}" for k, n in held["by_type"].items())
        lines.append(f"  Credentials stored as placeholders: {kinds}"
                     + (" (more than counted)" if held.get("capped") else ""))
    return lines


__all__ = ["SCAN_ROWS", "WAITING_SHOWN", "build", "describe", "origins", "placeholders"]
