# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""What the Governance audit pane shows, and how sure it may say it is.

4.1.25 audit findings L-16 / G-L2 / G-H2. The pane read ``event_type``,
``actor``, ``target`` and ``result`` while the API returned ``operation`` and
``agent_id``, so rows were blank; every actor was a capability digest; the
content hash was empty; and a green "Chain verified" stood over all of it,
which a manager reads as "the contents are verified".

* Each event keeps its stored fields and gains the presentation fields, built
  from what the entry actually records (who, on what, with what outcome).
* The badge says what the chain check covered: every field (hash version 2),
  only order and the six core fields (entries written before 4.1.25), or that
  the check failed or could not run. It never defaults to "verified".
"""

from __future__ import annotations

import json
from typing import Any

from superlocalmemory.compliance.audit import FULL_VERSIONS


def _metadata(event: dict) -> dict:
    try:
        parsed = json.loads(event.get("metadata") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _actor(event: dict, meta: dict) -> str:
    agent = str(event.get("agent_id") or "")
    connection = str(meta.get("connection_agent") or "")
    user = str(meta.get("user_id") or "")
    parts = [p for p in (f"user:{user}" if user else "", connection, agent) if p]
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return " · ".join(seen) or "unknown"


def _target(event: dict, meta: dict) -> str:
    bits = []
    if meta.get("target_user"):
        bits.append(f"user {meta['target_user']}")
    fact_ids = meta.get("fact_ids") or []
    if isinstance(fact_ids, list) and fact_ids:
        bits.append(f"{len(fact_ids)} fact(s)")
    if meta.get("operation_id"):
        bits.append(f"operation {meta['operation_id']}")
    profile = event.get("profile_id") or ""
    if profile:
        bits.append(f"workspace {profile}")
    else:
        bits.append("installation-wide")
    return ", ".join(bits)


def present_event(event: dict) -> dict:
    """The stored entry plus the fields the dashboard renders."""
    meta = _metadata(event)
    outcome = meta.get("outcome")
    if outcome is None and "success" in meta:
        outcome = "success" if meta.get("success") else "failed"
    return {
        **event,
        "event_type": event.get("operation", ""),
        "actor": _actor(event, meta),
        "action": str(event.get("operation", "")).replace("access.", ""),
        "target": _target(event, meta),
        "result": str(outcome or "recorded"),
        "content_bound": bool(event.get("content_hash")),
        "fully_covered": event.get("hash_version") in FULL_VERSIONS,
    }


def integrity_summary(chain: Any, *, cached: bool = False) -> dict[str, Any]:
    """``{verified, label, report}`` for the badge; never optimistic on error.

    ``cached`` answers a screen that asks again and again from the last complete
    walk plus the entries added since (see ``compliance.audit_verify_cache``);
    the default is the full walk.
    """
    try:
        if cached:
            from superlocalmemory.compliance.audit_verify_cache import cached_report

            report = cached_report(chain)
        else:
            report = chain.verify_report()
    except Exception:  # noqa: BLE001 - an unrun check is not a passed one
        return {"verified": False, "report": None,
                "label": "Integrity not checked: the audit trail could not be read"}
    if not report["ok"]:
        return {"verified": False, "report": report,
                "label": f"Integrity issue: {report['reason']}"}
    if report["legacy"]:
        label = (f"Order verified for all {report['entries']} entries; full details "
                 f"verified for {report['fully_covered']} (the {report['legacy']} written "
                 "before 4.1.25 record less)")
    else:
        label = f"Verified: all {report['entries']} entries, every field"
    return {"verified": True, "label": label, "report": report}


__all__ = ["integrity_summary", "present_event"]
