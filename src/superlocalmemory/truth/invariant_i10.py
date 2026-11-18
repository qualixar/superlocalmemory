# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""I-10 restated (D-49): no memory change without a certificate id, a rule-certificate id, or a
recorded decision, that is an ``actor@channel`` pair (``core/decisions``).

``slm doctor --truth`` runs these read-only queries, one per place a change is recorded: judge
receipts that acted, settled disputes, applied (or rolled back) proposal sets, promotions
(``fact_trust`` rows a caller confirmed) and approved firewall requests. Each returns the ids
that break the rule, prefixed by the table's short name so a sample says where to look. A
store without a table skips that part. Nothing here asks whether a person decided: a decision
by an agent over MCP is lawful when it names who and the channel, because it is recorded and
can be undone (``core/decision_undo``).
"""

from __future__ import annotations

import sqlite3

#: The channels a decision can name (``core/decisions.CHANNELS``), as a SQL ``GLOB`` suffix list.
_CHANNELS = ("mcp", "cli", "http", "dashboard")
_IN = "(" + ", ".join(f"'{c}'" for c in _CHANNELS) + ")"


def _decided(column: str) -> str:
    """SQL: ``column`` holds ``actor@channel`` (a name, then ``@``, then one of the channels)."""
    return "(" + " OR ".join(f"COALESCE({column}, '') GLOB '?*@{c}'" for c in _CHANNELS) + ")"


_RECEIPTS = f"""
    SELECT 'receipt:' || r.receipt_id FROM judge_receipts r
     WHERE r.action_taken IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM judge_certificates c
                        WHERE c.cert_id = r.action_ref AND c.profile_id = r.profile_id)
       AND NOT {_decided('r.action_ref')}"""

# 'judge_ladder' settles with a certificate (its receipts are checked above); 'dismissed' rows
# changed no memory (a dismissal only closes a question).
_DISPUTES = f"""
    SELECT 'dispute:' || d.dispute_id FROM slot_disputes d
     WHERE d.state = 'resolved' AND COALESCE(d.resolved_by, '') <> 'judge_ladder'
       AND NOT {_decided('d.resolved_by')}"""

_PROPOSAL_SETS = f"""
    SELECT 'proposal_set:' || p.set_id FROM proposal_sets p
     WHERE p.state IN ('applied', 'rolled_back')
       AND NOT (
         (p.basis_kind IN ('certificate', 'rule_certificate') AND EXISTS (
              SELECT 1 FROM judge_certificates c
               WHERE c.cert_id = p.basis_ref AND c.profile_id = p.profile_id))
         OR (p.basis_kind IN ('decision', 'person') AND COALESCE(p.decided_by, '') <> ''
             AND COALESCE(p.decided_channel, '') IN {_IN}))"""

# A promotion is a fact_trust row whose basis is ``confirmed:<channel>:<actor>``; the one other
# lawful channel is the upgrade's own record that it was promoted before 4.2.30 (M084).
_PROMOTIONS = f"""
    SELECT 'promotion:' || t.fact_id FROM fact_trust t
     WHERE t.basis LIKE 'confirmed:%'
       AND t.basis NOT LIKE 'confirmed:grandfathered:%'
       AND NOT ({" OR ".join(f"t.basis GLOB 'confirmed:{c}:?*'" for c in _CHANNELS)})"""

_APPROVALS = f"""
    SELECT 'approval:' || a.approval_id FROM trust_approvals a
     WHERE a.state = 'approved'
       AND (COALESCE(a.decided_by, '') = '' OR COALESCE(a.decided_channel, '') NOT IN {_IN})"""

#: (tables the query needs, query), in the order they are reported.
_CHECKS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("judge_receipts", "judge_certificates"), _RECEIPTS),
    (("slot_disputes",), _DISPUTES),
    (("proposal_sets", "judge_certificates"), _PROPOSAL_SETS),
    (("fact_trust",), _PROMOTIONS),
    (("trust_approvals",), _APPROVALS),
)


def violations(conn: sqlite3.Connection, present: set[str]) -> list[str]:
    """Every change recorded without a certificate or a decision, ``table:id`` each."""
    found: list[str] = []
    for needs, sql in _CHECKS:
        if set(needs) <= present:
            found.extend(str(r[0]) for r in conn.execute(sql))
    return found


__all__ = ["violations"]
