# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Per-delivery randomised withholding with logged propensities (D-14, D-28).

To learn whether a delivered memory helps, SLM must sometimes not deliver it,
at random and on record. Each *eligible* candidate of a delivery is included
iff a keyed-hash draw falls below its inclusion probability ``p = 1 - EPSILON``;
the probability and the draw are logged with it (``injection_receipts``), so
the inverse-propensity estimates of WS-3 (M-12) are exact. The estimand is the
marginal effect of including one candidate.

The design, explicitly:

* Bernoulli, independent per candidate, with one constant probability for the
  whole design (no adaptive or Thompson propensities: D-14);
* ``EPSILON`` never exceeds ``EPSILON_MAX`` (0.10, D-28; the schema checks it);
* never randomised (D-28): pinned facts, standing rules, the core block, and
  every path that is not a push (explicit ``recall``; cues and lease warnings
  when they exist). Those candidates are logged with probability 1;
* the draw is HMAC(key, "holdout:" + delivery id + ":" + fact id): reproducible
  from the key, unpredictable without it, independent of the fact's score.

``EPSILON`` is a design constant of the experiment (how often a candidate is
held out), not a threshold that decides a memory change from data; it sits at
the cap Varun set so the estimates gain power soonest (proposed as D-33, awaits
the W0 phase review).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

DESIGN = "bernoulli-v1"
#: The cap on the share of eligible deliveries withheld (D-28, Varun 15 Nov).
EPSILON_MAX = 0.10
#: This design's withholding probability per eligible candidate (D-33, proposed).
EPSILON = 0.10

#: Delivery paths that write receipts (W0: the two session-start pushes).
PATHS = frozenset({"session_init", "session_context"})
#: Paths whose eligible candidates are randomised: pushes only, never a pull.
RANDOMISED_PATHS = frozenset({"session_init", "session_context"})

#: Candidate roles, and the reason a role is never withheld.
ELIGIBLE_ROLES = frozenset({"recall", "recent", "session_summary"})
NEVER_WITHHELD = {"pinned": "pinned", "standing": "standing_rule", "core": "core_block"}
ROLES = ELIGIBLE_ROLES | frozenset(NEVER_WITHHELD)

#: Why an otherwise eligible candidate was not randomised on this delivery.
NOT_RANDOMISED_PATH = "path_not_randomised"
HOLDOUT_OFF = "holdout_off"


@dataclass(frozen=True)
class Candidate:
    fact_id: str
    role: str
    score: float | None = None


@dataclass(frozen=True)
class Choice:
    fact_id: str
    role: str
    position: int
    score: float | None
    eligible: bool
    exclusion: str | None
    propensity: float
    draw: float | None
    included: bool


def draw(key: bytes, delivery_id: str, fact_id: str) -> float:
    """A uniform number in [0, 1) fixed by the key, the delivery and the fact."""
    msg = f"holdout:{delivery_id}:{fact_id}".encode()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def _exclusion(role: str, path: str, holdout_on: bool) -> str | None:
    if role in NEVER_WITHHELD:
        return NEVER_WITHHELD[role]
    if path not in RANDOMISED_PATHS:
        return NOT_RANDOMISED_PATH
    if not holdout_on:
        return HOLDOUT_OFF
    return None


def choose(key: bytes, delivery_id: str, path: str, candidates: Iterable[Candidate], *,
           holdout_on: bool, epsilon: float = EPSILON) -> tuple[Choice, ...]:
    """One choice per distinct candidate fact, in order (a fact's first role wins)."""
    if path not in PATHS:
        raise ValueError(f"unknown delivery path: {path!r}")
    if not 0.0 <= epsilon <= EPSILON_MAX:
        raise ValueError(f"withholding probability {epsilon} outside [0, {EPSILON_MAX}]")
    choices: list[Choice] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.fact_id or candidate.fact_id in seen:
            continue
        if candidate.role not in ROLES:
            raise ValueError(f"unknown candidate role: {candidate.role!r}")
        seen.add(candidate.fact_id)
        exclusion = _exclusion(candidate.role, path, holdout_on)
        if exclusion is None:
            u = draw(key, delivery_id, candidate.fact_id)
            p = 1.0 - epsilon
            choices.append(Choice(candidate.fact_id, candidate.role, len(choices) + 1,
                                  candidate.score, True, None, p, u, u < p))
        else:
            choices.append(Choice(candidate.fact_id, candidate.role, len(choices) + 1,
                                  candidate.score, False, exclusion, 1.0, None, True))
    return tuple(choices)


__all__ = ["DESIGN", "ELIGIBLE_ROLES", "EPSILON", "EPSILON_MAX", "HOLDOUT_OFF",
           "NEVER_WITHHELD", "NOT_RANDOMISED_PATH", "PATHS", "RANDOMISED_PATHS", "ROLES",
           "Candidate", "Choice", "choose", "draw"]
