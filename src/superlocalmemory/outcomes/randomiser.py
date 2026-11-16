# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Randomised withholding with logged propensities: design ``one-out-v2`` (D-14, D-33, D-44).

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

**Two arms share the one budget (D-44, Varun 16 Nov; ``one-out-v2``).** A
treated delivery (withhold draw < EPSILON) gets exactly one change: either one
eligible memory left out (the *memory* arm, as above) or one retrieval channel
switched off for the fetch behind it (the *channel* arm, W3-3), never both, so
at most EPSILON of eligible deliveries are touched (D-28). The arm comes from a
third keyed draw (":arm"), uniform over the arms the path has: only a delivery
that runs retrieval (``session_init``) has the channel arm; the hook's session
start reads SQL, so its treated deliveries all take the memory arm. The channel
is drawn uniformly from ``DROPPABLE_CHANNELS`` with the pick draw. Every
probability is exact and logged: a memory's inclusion probability is
``1 - EPSILON * memory_share / k`` and a channel's drop probability is
``EPSILON * (1 - memory_share) / len(DROPPABLE_CHANNELS)``.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass

DESIGN = "one-out-v2"

MEMORY_ARM = "memory"
CHANNEL_ARM = "channel"
#: The arms each randomised path offers (a path without retrieval has no channel arm).
ARMS = {"session_init": (MEMORY_ARM, CHANNEL_ARM), "session_context": (MEMORY_ARM,)}
#: Channels the channel arm may switch off: the text retrieval channels. ``media``
#: answers picture questions only, so dropping it at a session start would spend
#: the budget on a change that almost never shows.
DROPPABLE_CHANNELS = ("semantic", "bm25", "temporal", "hopfield", "spreading_activation",
                      "entity_graph", "profile")
#: The cap on the share of eligible deliveries withheld (D-28, Varun 15 Nov).
EPSILON_MAX = 0.10
#: This design's share of eligible deliveries that lose one memory (D-33).
EPSILON = 0.10

#: Paths whose eligible candidates are randomised: pushes only, never a pull.
RANDOMISED_PATHS = frozenset({"session_init", "session_context"})
#: Pull paths (W3-1): an agent asked, so nothing is withheld (each candidate is
#: logged at propensity 1, ``path_not_randomised``), but every answer is on record.
PULL_PATHS = frozenset({"mcp_recall", "cli_recall", "http_recall", "hook_recall",
                        "view_recall"})
#: Delivery paths that write receipts: every surface that puts memories in an
#: agent's context (D-22). A person reading the dashboard is not a delivery.
PATHS = RANDOMISED_PATHS | PULL_PATHS

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
class Treatment:
    """What one delivery's draws decided, before any candidate is known."""

    #: The keyed draws; None when nothing on this delivery is randomised.
    withhold_draw: float | None
    arm_draw: float | None
    pick_draw: float | None
    #: ``memory`` or ``channel`` when treated, else None.
    arm: str | None
    #: The channel switched off (channel arm only).
    channel: str | None
    #: P(memory arm | treated) on this path; None when not randomised.
    memory_share: float | None

    @property
    def randomised(self) -> bool:
        return self.withhold_draw is not None


UNTREATED = Treatment(None, None, None, None, None, None)


def treat(key: bytes, delivery_id: str, path: str, *, holdout_on: bool,
          epsilon: float = EPSILON, channel_arm: bool = True) -> Treatment:
    """The delivery's arm (if any), from its keyed draws; nothing on a path that
    is not randomised or with the holdout off. ``channel_arm=False`` when the
    fetch has already run: a channel can no longer be dropped, so only the
    memory arm is offered (and logged as such)."""
    if path not in PATHS:
        raise ValueError(f"unknown delivery path: {path!r}")
    if not 0.0 <= epsilon <= EPSILON_MAX:
        raise ValueError(f"withholding share {epsilon} outside [0, {EPSILON_MAX}]")
    if path not in RANDOMISED_PATHS or not holdout_on:
        return UNTREATED
    arms = ARMS[path] if channel_arm else (MEMORY_ARM,)
    u = draw(key, delivery_id, "withhold")
    a = draw(key, delivery_id, "arm")
    pick = draw(key, delivery_id, "pick")
    arm = arms[int(a * len(arms))] if u < epsilon else None
    channel = (DROPPABLE_CHANNELS[int(pick * len(DROPPABLE_CHANNELS))]
               if arm == CHANNEL_ARM else None)
    return Treatment(u, a, pick, arm, channel, arms.count(MEMORY_ARM) / len(arms))


def channel_drop_probability(treatment: Treatment, epsilon: float = EPSILON) -> float:
    """P(this delivery's fetch lost one given channel), exact."""
    if treatment.memory_share is None:
        return 0.0
    return epsilon * (1.0 - treatment.memory_share) / len(DROPPABLE_CHANNELS)


@dataclass(frozen=True)
class Assignment:
    choices: tuple[Choice, ...]
    eligible_count: int
    #: The two keyed draws; None when nothing on this delivery was randomised.
    withhold_draw: float | None
    pick_draw: float | None
    #: The delivery's treatment (arm, channel, shares), D-44.
    treatment: Treatment = UNTREATED


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
           holdout_on: bool, epsilon: float = EPSILON,
           treatment: Treatment | None = None) -> Assignment:
    """The delivery's candidates with their choices. ``treatment`` is the one
    decided before the fetch (``session_init``); otherwise it is drawn here."""
    if treatment is None:
        treatment = treat(key, delivery_id, path, holdout_on=holdout_on, epsilon=epsilon)
    elif path not in PATHS:
        raise ValueError(f"unknown delivery path: {path!r}")
    distinct = _distinct(candidates)
    exclusions = [_exclusion(c.role, path, holdout_on) for c in distinct]
    eligible = [i for i, e in enumerate(exclusions) if e is None]
    k = len(eligible)
    withheld: int | None = None
    if k and treatment.arm == MEMORY_ARM:
        withheld = eligible[int(treatment.pick_draw * k)]
    share = treatment.memory_share if treatment.memory_share is not None else 1.0
    p_eligible = 1.0 - epsilon * share / k if k else 1.0
    choices = tuple(
        Choice(c.fact_id, c.role, i + 1, c.score, exclusions[i] is None, exclusions[i],
               p_eligible if exclusions[i] is None else 1.0, i != withheld)
        for i, c in enumerate(distinct))
    return Assignment(choices, k, treatment.withhold_draw, treatment.pick_draw, treatment)


__all__ = ["ARMS", "CHANNEL_ARM", "DROPPABLE_CHANNELS", "MEMORY_ARM", "UNTREATED", "Treatment",
           "channel_drop_probability", "treat",
           "DESIGN", "ELIGIBLE_ROLES", "EPSILON", "EPSILON_MAX", "HOLDOUT_OFF",
           "NEVER_WITHHELD", "NOT_RANDOMISED_PATH", "PATHS", "PULL_PATHS", "RANDOMISED_PATHS",
           "ROLES", "Assignment", "Candidate", "Choice", "choose", "draw"]
