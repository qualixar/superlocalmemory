# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Per-fact outcome score: read it, update it, and damp it honestly.

WHAT IT IS
----------
One number per (fact, profile): an exponentially-weighted average of the rewards
of the settlements that fact took part in. Ranking otherwise scores a memory
purely by how much it *resembles* the query — nothing in the pipeline knows
whether a memory has ever actually helped.

WHY IT IS NOT A MODEL FEATURE, WHICH AN EARLIER DESIGN GOT BACKWARDS
----------------------------------------------------------------
An earlier design said to add ``"outcome_score"`` to ``FEATURE_NAMES`` for inference and
exclude it from the training matrix. That cannot work, and the model's own
closing gotcha says why: ``booster.predict(X)`` needs the same columns in the
same order the model was trained on. A feature present at inference and absent
in training is a shape mismatch, not a clever exclusion.

Two more reasons, both checkable: ``features.py`` asserts
``len(FEATURE_NAMES) == FEATURE_DIM`` with ``FEATURE_DIM = 20``, and
``routes/brain.py`` surfaces that constant as ``feature_count_expected``. The
live model is a 20-feature model. Adding a 21st silently invalidates it.

So PCOS is applied AFTER the model score, as a bonus on the ranking score. That
satisfies this true by construction: there is nothing to exclude from
training because it never enters training. Self-reinforcement — "model learns
high PCOS wins, which raises PCOS" — is impossible when the model cannot see it.

THE TWO DAMPING RULES, AND WHAT EACH IS FOR
-------------------------------------------
**Confidence weighting.** A fact settled once at reward 1.0 is not a fact that
works; it is a fact that worked once. The bonus is scaled by
``log1p(play_count) / log1p(20)``, so a single settlement contributes ~1/5 of
what twenty do, and an unsettled fact contributes exactly nothing rather than
being penalised for being new.

**Rich-get-richer.** A fact that ranks well gets shown, gets settled, ranks
better. Measured over 1,000 simulated queries with near-tied retrieval scores,
the single most-favoured fact took first place in 5.30% of them with the bonus
on and 1.30% with it off — so the bonus alone breaks the release limit
of "no fact above 5%". The counter-pressure is ``RecentTopCounter``: once a fact
has won first place ``_CAP_MIN`` times inside a rolling window it stops
receiving the bonus. Not removed and not demoted — a memory that genuinely keeps
being relevant must stay returnable. It only stops compounding.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_scores",
    "update_scores",
    "confidence_weight",
    "bonus_for",
    "RecentTopCounter",
    "RECENT_TOPS",
    "TAU",
    "MAX_BONUS",
]

#: EMA rate. Low on purpose: one outcome should nudge a score, not define it.
TAU = 0.1

#: play_count at which a score is trusted in full.
_FULL_CONFIDENCE_PLAYS = 20

#: Largest absolute change PCOS may make to a ranking score, as a fraction.
#: The point of PCOS is to break ties between similar-looking memories, not to
#: overrule the retrieval that found them — a memory that does not match the
#: query must never be dragged to the top by history.
MAX_BONUS = 0.15

#: First-place finishes inside the rolling window before a fact stops
#: receiving the bonus. Three is low on purpose: the measurement shows
#: concentration comes from a small number of repeat winners, so the cap has to
#: bite early to matter at all.
_CAP_MIN = 3

_NEUTRAL = 0.5


def confidence_weight(play_count: int) -> float:
    """How much of a fact's score to believe, from how often it has settled.

    ``log1p`` rather than linear: the difference between one settlement and two
    is far more informative than between nineteen and twenty.
    """
    if play_count <= 0:
        return 0.0
    return min(1.0, math.log1p(play_count) / math.log1p(_FULL_CONFIDENCE_PLAYS))


def bonus_for(score: float, play_count: int) -> float:
    """Signed ranking bonus in ``[-MAX_BONUS, +MAX_BONUS]``.

    Centred on 0.5, so a fact whose outcomes are neutral gets no bonus at all
    and an unproven fact is never penalised relative to one that has never been
    tried.
    """
    return (
        (float(score) - _NEUTRAL) * 2.0
        * confidence_weight(int(play_count))
        * MAX_BONUS
    )


class RecentTopCounter:
    """How often each fact has recently won first place, per profile.

    WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL. Measured over a
    1,000-query simulation with 200 facts and near-tied retrieval scores
    (spread 0.02, which is what an embedding channel actually returns for
    closely related memories), the single most-favoured fact took first place:

        no outcome bonus          1.30% of queries
        bonus, MAX_BONUS = 0.15   5.30% of queries

    The release limit is "no fact reaches more than 5% of displays over
    a 1,000-query simulation". The bonus alone breaks it. A control run with the
    bonus disabled sits at 1.30% at every spread, so the bonus is the cause and
    not the tie-breaking — that control is the only reason this is known.

    HOW IT DIFFERS FROM THE PLAN. The original countermeasure multiplies a capped
    fact's ``ranking_score`` by 0.1. That is a 10x demotion of a memory whose
    only offence is having been useful three times, and it would visibly damage
    answers — a genuinely relevant memory must stay returnable. This instead
    withholds the BONUS from a fact that has recently been winning. The fact
    keeps every point retrieval gave it and simply stops compounding.

    WHY A ROLLING WINDOW RATHER THAN A SESSION. ``run_recall`` has no
    ``session_id`` parameter, so there is no session identity at this layer to
    key on, and threading one through every caller to bound an in-memory
    counter is not worth it. A rolling window over the last ``_WINDOW`` queries
    per profile gives the same property — recent concentration is what
    compounds — without new plumbing. In-process and ephemeral on purpose: a DB
    write per displayed fact per query is exactly the contention the exposure
    enqueue was switched off to avoid.

    COUNTS DIFFERENT QUESTIONS, NOT ASKS (4.1.20). The risk is one memory
    winning MANY questions. Counting every ask meant one question asked a
    fourth time capped its own top answer and flipped it — the answer changed
    because the person asked again. A win is now recorded against the question
    (``query_key``, a hash — never the text); asking it again refreshes that
    win instead of adding one.
    """

    __slots__ = ("_counts", "_seen", "_anonymous")

    #: Queries per profile before the window resets.
    _WINDOW = 200

    def __init__(self) -> None:
        # profile -> fact -> ordered set of the questions it won (oldest first)
        self._counts: dict[str, dict[str, dict[str, None]]] = defaultdict(dict)
        self._seen: dict[str, int] = defaultdict(int)
        self._anonymous = 0

    def record_top(self, profile_id: str, fact_id: str,
                   query_key: str | None = None) -> None:
        """Note that ``fact_id`` took first place for ``profile_id``.

        ``query_key`` identifies the question; without one, every call counts
        as a different question (the behaviour before 4.1.20).

        The window DECAYS rather than being dropped. Emptying it wholesale every
        ``_WINDOW`` queries handed every previously-capped memory its bonus back
        at the same instant, so concentration spiked immediately after each
        reset — the cap stopped biting exactly when the run-up had made it most
        necessary. Halving instead keeps a repeat winner near its cap and lets a
        memory that has stopped winning recover gradually.
        """
        key = profile_id or ""
        self._seen[key] += 1
        if self._seen[key] > self._WINDOW:
            bucket = self._counts.get(key)
            if bucket:
                # Keep the most recent half of each fact's won questions.
                self._counts[key] = {
                    f: dict.fromkeys(list(won)[len(won) - len(won) // 2:])
                    for f, won in bucket.items() if len(won) > 1
                }
            self._seen[key] = 0
        if query_key is None:
            self._anonymous += 1
            query_key = f"#anon{self._anonymous}"
        won = self._counts[key].setdefault(fact_id, {})
        won.pop(query_key, None)
        won[query_key] = None

    def tops(self, profile_id: str, fact_id: str) -> int:
        """How many different questions this fact has recently won."""
        return len(self._counts.get(profile_id or "", {}).get(fact_id, ()))

    def capped(self, profile_id: str, fact_id: str) -> bool:
        """Whether this fact has won often enough to stop earning a bonus."""
        return self.tops(profile_id, fact_id) >= _CAP_MIN

    def forget(self, profile_id: str) -> None:
        """Drop everything held for a profile. Called on erasure.

        In-process and ephemeral, so it dies with the process anyway — but an
        Article 17 request must not leave a profile's recent winners sitting in
        a live process's memory for the rest of its lifetime.
        """
        key = profile_id or ""
        self._counts.pop(key, None)
        self._seen.pop(key, None)


#: One counter for the process. Ephemeral by design — see the class docstring.
RECENT_TOPS = RecentTopCounter()


def fetch_scores(
    conn: sqlite3.Connection,
    profile_id: str,
    fact_ids: list[str],
) -> dict[str, tuple[float, int]]:
    """``{fact_id: (score, play_count)}`` for the ids that have a row.

    One batched query against the ``(fact_id, profile_id)`` primary key rather
    than a LEFT JOIN into the hydration SQL, which is what an earlier design proposed.
    The JOIN would have to reach ``AtomicFact``, and ``_row_to_fact`` ignores
    columns the dataclass does not declare — so it would mean adding fields to a
    model used across the whole codebase to carry a number only the ranker
    reads. A missing row is a cold start, and the caller treats it as neutral.

    Never raises: on a store where M045 has not run this returns ``{}`` and
    ranking proceeds exactly as it did before PCOS existed.
    """
    ids = [str(f) for f in fact_ids if f]
    if not ids:
        return {}
    placeholders = ", ".join("?" * len(ids))
    try:
        rows = conn.execute(
            "SELECT fact_id, score, play_count FROM fact_outcome_score "
            f"WHERE profile_id = ? AND fact_id IN ({placeholders})",
            (str(profile_id), *ids),
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("pcos.fetch_scores unavailable: %s", exc)
        return {}
    out: dict[str, tuple[float, int]] = {}
    for row in rows:
        try:
            out[str(row[0])] = (float(row[1]), int(row[2]))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def update_scores(
    conn: sqlite3.Connection,
    profile_id: str,
    fact_ids: list[str],
    reward: float,
) -> int:
    """Fold ``reward`` into each fact's score. Returns rows written.

    Shrinkage on the way in: the effective rate is ``TAU * min(1, plays/10)``, so
    the first few settlements move a score gently and a single lucky outcome
    cannot define a fact. A fact with no row starts from the neutral prior and
    takes one ``TAU`` step toward the reward.

    The caller owns the transaction. Never raises — a lost PCOS update costs one
    increment of a score that is advisory by construction, and it must never
    take down the settlement that produced it.
    """
    ids = [str(f) for f in fact_ids if f]
    if not ids:
        return 0
    try:
        reward_f = max(0.0, min(1.0, float(reward)))
    except (TypeError, ValueError):
        return 0

    existing = fetch_scores(conn, profile_id, ids)
    written = 0
    for fid in ids:
        old_score, old_plays = existing.get(fid, (_NEUTRAL, 0))
        if old_plays <= 0:
            new_score = _NEUTRAL + TAU * (reward_f - _NEUTRAL)
        else:
            rate = TAU * min(1.0, old_plays / 10.0)
            new_score = (1.0 - rate) * old_score + rate * reward_f
        new_score = max(0.0, min(1.0, new_score))
        try:
            conn.execute(
                "INSERT INTO fact_outcome_score "
                "(fact_id, profile_id, score, play_count, updated_at) "
                "VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT (fact_id, profile_id) DO UPDATE SET "
                "  score = excluded.score, "
                "  play_count = fact_outcome_score.play_count + 1, "
                "  updated_at = excluded.updated_at",
                (fid, str(profile_id), new_score, old_plays + 1),
            )
            written += 1
        except sqlite3.Error as exc:
            logger.debug("pcos.update_scores %s: %s", fid, exc)
    return written
