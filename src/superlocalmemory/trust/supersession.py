# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""May one memory retire another without a person? The origin lattice decides (M-15, R-18).

With ``trust.taint`` on, nothing retires a slot's head automatically unless the
class of the winner is at least the class of the one it retires
(``origin_class.at_least``): a web page cannot overwrite what the user said, a
tool's output cannot overwrite an agent's own words. ``unknown`` on either side
never passes (it is outside the order). A blocked decision is not dropped: it
stays an open dispute, relabelled ``origin_below_head``, and only a person
settles it (``slm disputes settle``, the dashboard): a person's answer is a
lawful basis whatever the classes (I-10).

The class compared is the *taint* (the lowest class among a fact and what it
was derived from), because a fact built from a web page is no more trustworthy
than the page. With the flag off nothing here is called and behaviour is
exactly as before.

Read-only: classes come from ``fact_trust`` when a row exists and from the raw
signals otherwise. Ids and class names only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from superlocalmemory.trust import fact_trust
from superlocalmemory.trust import origin_class as oc

#: The dispute reason of a supersession the lattice refused.
ORIGIN_BELOW_HEAD = "origin_below_head"


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    winner_class: str
    loser_class: str

    @property
    def reason(self) -> str | None:
        return None if self.allowed else ORIGIN_BELOW_HEAD


def _taint(conn: sqlite3.Connection, fact_id: str) -> str:
    found = fact_trust.origin_of(conn, fact_id)
    return found["taint"] if found else oc.UNKNOWN


def decide(conn: sqlite3.Connection, *, winner: str, loser: str) -> Verdict:
    """Whether ``winner`` may retire ``loser`` on its own."""
    w, lo = _taint(conn, winner), _taint(conn, loser)
    return Verdict(oc.at_least(w, lo), w, lo)


def allows(conn: sqlite3.Connection, winner: str, loser: str) -> bool:
    return decide(conn, winner=winner, loser=loser).allowed


__all__ = ["ORIGIN_BELOW_HEAD", "Verdict", "allows", "decide"]
