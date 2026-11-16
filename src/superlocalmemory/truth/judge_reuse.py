# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Never judge the same pair twice: the reuse rule the background passes share.

Per (profile, keyed pair hash) under one question version, the receipts say
either that a tier was sure (an accepted decision: act on it, never re-ask) or
which tiers gave a real answer and were not sure (an abstention). A pair every
model tier available here has abstained on waits for a person, so a store never
pays twice for an unanswerable pair. A failed ask judged nothing (R-35), so it
is asked again, as is a pair a newly available tier could answer. Never across
profiles (W0-15). Shadow answers (``<population>:shadow``) are never accepted,
so they cannot become a reused decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from superlocalmemory.core.judge_ladder import Decision

_EARLIER = """
SELECT inputs_hash, receipt_id, tier, model, verdict, confidence, accepted, failure_kind,
       batching_regime, threshold
  FROM judge_receipts WHERE question_version = ? AND profile_id = ? AND inputs_hash IN ({marks})
 ORDER BY decided_at
"""

Earlier = dict[tuple[str, str], "Decision | frozenset[str]"]


def earlier(conn: Any, version: str, keys: Sequence[tuple[str, str]]) -> Earlier:
    """Per (profile, pair hash): the accepted decision, or the tiers that abstained."""
    out: Earlier = {}
    by_profile: dict[str, list[str]] = {}
    for profile, h in keys:
        by_profile.setdefault(profile, []).append(h)
    for profile, hashes in by_profile.items():
        for start in range(0, len(hashes), 500):
            part = hashes[start:start + 500]
            sql = _EARLIER.format(marks=",".join("?" * len(part)))
            for h, rid, tier, model, verdict, conf, accepted, failure, regime, routed in \
                    conn.execute(sql, (version, profile, *part)):
                key = (profile, h)
                seen = out.get(key, frozenset())
                if isinstance(seen, Decision):
                    continue
                if accepted:
                    out[key] = Decision(verdict, 1.0 if conf is None else float(conf), tier,
                                        (rid,), model=model, regime=regime,
                                        routing_threshold=routed)
                else:
                    out[key] = seen | ({tier} if failure is None else set())
    return out


def plan(keys: Sequence[tuple[str, str]], seen: Earlier, available: frozenset[str],
         limit: int) -> tuple[dict[int, Decision], list[int]]:
    """(earlier decisions to act on by index, indexes to ask now, at most ``limit``)."""
    reused: dict[int, Decision] = {}
    ask: list[int] = []
    for i, key in enumerate(keys):
        got = seen.get(key, frozenset())
        if isinstance(got, Decision):
            reused[i] = got
        elif key in seen and available <= got:
            continue  # every tier that could answer here has abstained: page past it
        elif len(ask) < limit:  # never judged, a tier failed, or a tier is newly here
            ask.append(i)
    return reused, ask


__all__ = ["earlier", "plan"]
