# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Randomised withholding with logged propensities: design ``one-out-v1`` (D-14, D-33).

To learn whether a delivered memory helps, SLM must sometimes not deliver it,
at random and on record. In a share ``EPSILON`` of eligible deliveries exactly
one eligible candidate, drawn uniformly, is left out; every other delivery is
untouched (D-28: at most 0.10 of eligible deliveries). With ``k`` eligible
candidates each one's inclusion probability is ``1 - EPSILON / k``, exact and
logged with it, so the inverse-propensity estimates of WS-3 (M-12) are exact.
The contrast is on *included* (assignment), not on *delivered*, which the token
budget decides after assignment.

The design, explicitly:

* two keyed draws per delivery, HMAC(key, "holdout:" + delivery id + ":withhold")
  and (":pick"): reproducible from the key, unpredictable without it, and
  independent of every candidate's score;
* never randomised (D-28): pinned facts, standing rules, the core block, and
  every path that is not a push (explicit ``recall``; cues and lease warnings
  when they exist). Those candidates are logged with probability 1. Which facts
  are pinned or standing is read from the store by ``receipts.plan`` (W0-01),
  never trusted to the role a surface assigns;
* ``EPSILON`` never exceeds ``EPSILON_MAX`` (the schema checks it too).

``EPSILON`` is a design constant of the experiment, not a threshold deciding a
memory change from data (D-33, Varun 16 Nov). Withholding is off by default
until D-28's shipping conditions exist (D-34).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

DESIGN = "one-out-v1"
#: The cap on the share of eligible deliveries withheld (D-28, Varun 15 Nov).
EPSILON_MAX = 0.10
#: This design's share of eligible deliveries that lose one memory (D-33).
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
    included: bool


@dataclass(frozen=True)
class Assignment:
    choices: tuple[Choice, ...]
    eligible_count: int
    #: The two keyed draws; None when nothing on this delivery was randomised.
    withhold_draw: float | None
    pick_draw: float | None


def draw(key: bytes, delivery_id: str, label: str) -> float:
    """A uniform number in [0, 1) fixed by the key, the delivery and the label."""
    msg = f"holdout:{delivery_id}:{label}".encode()
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


def _distinct(candidates: Iterable[Candidate]) -> list[Candidate]:
    """One candidate per fact, in order: a fact's first role wins."""
    out: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.fact_id or candidate.fact_id in seen:
            continue
        if candidate.role not in ROLES:
            raise ValueError(f"unknown candidate role: {candidate.role!r}")
        seen.add(candidate.fact_id)
        out.append(candidate)
    return out


def choose(key: bytes, delivery_id: str, path: str, candidates: Iterable[Candidate], *,
           holdout_on: bool, epsilon: float = EPSILON) -> Assignment:
    if path not in PATHS:
        raise ValueError(f"unknown delivery path: {path!r}")
    if not 0.0 <= epsilon <= EPSILON_MAX:
        raise ValueError(f"withholding share {epsilon} outside [0, {EPSILON_MAX}]")
    distinct = _distinct(candidates)
    exclusions = [_exclusion(c.role, path, holdout_on) for c in distinct]
    eligible = [i for i, e in enumerate(exclusions) if e is None]
    k = len(eligible)
    withhold_draw = pick_draw = None
    withheld: int | None = None
    if k:
        withhold_draw = draw(key, delivery_id, "withhold")
        pick_draw = draw(key, delivery_id, "pick")
        if withhold_draw < epsilon:
            withheld = eligible[int(pick_draw * k)]
    p_eligible = 1.0 - epsilon / k if k else 1.0
    choices = tuple(
        Choice(c.fact_id, c.role, i + 1, c.score, exclusions[i] is None, exclusions[i],
               p_eligible if exclusions[i] is None else 1.0, i != withheld)
        for i, c in enumerate(distinct))
    return Assignment(choices, k, withhold_draw, pick_draw)


__all__ = ["DESIGN", "ELIGIBLE_ROLES", "EPSILON", "EPSILON_MAX", "HOLDOUT_OFF",
           "NEVER_WITHHELD", "NOT_RANDOMISED_PATH", "PATHS", "RANDOMISED_PATHS", "ROLES",
           "Assignment", "Candidate", "Choice", "choose", "draw"]
