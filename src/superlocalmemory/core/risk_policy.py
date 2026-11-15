# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The error levels certificates are held to (blueprint D-25).

alpha is the error rate a certificate bounds, delta the chance the bound is
wrong. They are policy, set by the owner, not tuned: changing them changes what
every certificate means. Decided by the owner on 15 Nov 2026 (blueprint D-25).
"""

from __future__ import annotations

DECIDED = True
#: False retirement: acting on "replaces" when the older memory is still true.
ALPHA_RETIRE = 0.05
#: False keep: closing a dispute that was a real replacement. Must sit below the
#: replacement base rate of the inbox, or any judge passes (Fable, blueprint M-02).
ALPHA_CLOSE = 0.01
DELTA = 0.05


def alpha_for(verdict: str) -> float:
    return ALPHA_RETIRE if verdict == "replaces" else ALPHA_CLOSE


__all__ = ["ALPHA_CLOSE", "ALPHA_RETIRE", "DECIDED", "DELTA", "alpha_for"]
