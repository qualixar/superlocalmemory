# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the truth layer adds to recall results (D-37, D-38, D-39).

Each returned memory may carry:

* ``disputes``: open disputes on its slot, both sides, with ``next: ask_user``;
* ``source``: the state of the file it came from (``fresh``, ``changed``,
  ``missing``), found with a ``stat`` and, when size or time moved, a hash;
* ``origin`` (only while ``trust.taint`` is on or in shadow, WS-5): the class
  and taint of where it came from, in ids and words, from one batched select of
  ``fact_trust`` (a bounded live computation for facts with no row yet).

Recall stays fast and never writes here: one read-only connection (WAL readers
never wait for the writer), no lock, and any failure leaves the results as
they were. Off when the kill switch (D-23) is set or ``truth.slots`` is off.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def enabled(slm_home: Path | None) -> bool:
    from superlocalmemory.core.feature_flags import FeatureFlags

    try:
        return FeatureFlags(slm_home).observing("truth.slots")
    except Exception:  # noqa: BLE001 - a bad flags file must not break recall
        return False


def origin_enabled(slm_home: Path | None) -> bool:
    from superlocalmemory.core.feature_flags import FeatureFlags

    try:
        return FeatureFlags(slm_home).observing("trust.taint")
    except Exception:  # noqa: BLE001 - a bad flags file must not break recall
        return False


def annotate(db_path: Path | str, profile_id: str, results: list[dict], *,
             hidden: Callable[[list[str]], set[str]] = lambda ids: set(),
             slm_home: Path | None = None) -> list[dict]:
    """Return ``results`` with ``disputes``, ``source`` and ``origin`` added where they apply."""
    fact_ids = [str(r.get("fact_id") or "") for r in results]
    truth_on, trust_on = enabled(slm_home), origin_enabled(slm_home)
    if not any(fact_ids) or not (truth_on or trust_on):
        return results
    from superlocalmemory.trust import fact_trust
    from superlocalmemory.truth import disputes, leases
    from superlocalmemory.truth.invariants import open_read_only

    try:
        conn = open_read_only(str(db_path))
    except Exception as exc:  # noqa: BLE001 - presentation only
        logger.debug("truth notes unavailable (%s)", type(exc).__name__)
        return results
    found: dict = {}
    sources: dict = {}
    origins: dict = {}
    try:
        if truth_on:
            try:
                found = disputes.for_facts(conn, profile_id, fact_ids, hidden=hidden)
                sources = leases.check_for_facts(conn, fact_ids)
            except Exception as exc:  # noqa: BLE001 - presentation only, never fail a recall
                logger.debug("truth notes skipped (%s)", type(exc).__name__)
        if trust_on:
            try:
                origins = fact_trust.for_results(conn, fact_ids)
            except Exception as exc:  # noqa: BLE001 - presentation only
                logger.debug("origin notes skipped (%s)", type(exc).__name__)
    finally:
        conn.close()
    out = []
    for item, fact_id in zip(results, fact_ids):
        extra: dict[str, Any] = {}
        if fact_id in found:
            extra["disputed"] = True
            extra["disputes"] = found[fact_id]
        if fact_id in sources:
            extra["source"] = sources[fact_id]
        if fact_id in origins:  # the basis row is for ``slm why``, not for every result
            extra["origin"] = {k: v for k, v in origins[fact_id].items() if k != "basis"}
        out.append({**item, **extra} if extra else item)
    return out


__all__ = ["annotate", "enabled", "origin_enabled"]
