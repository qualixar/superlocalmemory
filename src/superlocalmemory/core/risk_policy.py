# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The error levels certificates are held to (blueprint D-25).

alpha is the error rate a certificate bounds, delta the chance the bound is
wrong. They are policy, set by the owner, not tuned: changing them changes what
every certificate means. Decided by the owner on 15 Nov 2026 (blueprint D-25).
"""

from __future__ import annotations

#: False retirement: acting on "replaces" when the older memory is still true.
ALPHA_RETIRE = 0.05
#: False keep: closing a dispute that was a real replacement. Must sit below the
#: replacement base rate of the inbox, or any judge passes (Fable, blueprint M-02).
ALPHA_CLOSE = 0.01
DELTA = 0.05


#: The chance a judged receipt is drawn for a blind audit label (M-04): one
#: constant across every score within a certificate cell, so audited counts are
#: binomial there. D-35 (Varun, 16 Nov) sets it per cell: 1.0 for any verdict
#: that retires (rare, costly, reviewed anyway); this default elsewhere until
#: each cell's rate is derived from its measured volume (F-19 re-measure).
AUDIT_PROBABILITY = 0.10
#: The same draw for the inbox sample (one per pair, W0-05).
INBOX_PROBABILITY = AUDIT_PROBABILITY


def alpha_for(verdict: str) -> float:
    return ALPHA_RETIRE if verdict == "replaces" else ALPHA_CLOSE


def audit_probability_for(verdict: str, base: float | None) -> float | None:
    """The cell's audit probability (D-35): 1.0 for a retiring verdict."""
    if base is None:
        return None
    return 1.0 if verdict == "replaces" else base


__all__ = ["ALPHA_CLOSE", "ALPHA_RETIRE", "AUDIT_PROBABILITY", "DELTA", "INBOX_PROBABILITY",
           "alpha_for", "audit_probability_for"]
