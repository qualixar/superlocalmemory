# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What the truth layer adds to recall results (D-37, D-38, D-39).

Each returned memory may carry:

* ``disputes``: open disputes on its slot, both sides, with ``next: ask_user``;
* ``source``: the state of the file it came from (``fresh``, ``changed``,
  ``missing``), found with a ``stat`` and, when size or time moved, a hash.

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


def annotate(db_path: Path | str, profile_id: str, results: list[dict], *,
             hidden: Callable[[list[str]], set[str]] = lambda ids: set(),
             slm_home: Path | None = None) -> list[dict]:
    """Return ``results`` with ``disputes`` and ``source`` added where they apply."""
    fact_ids = [str(r.get("fact_id") or "") for r in results]
    if not any(fact_ids) or not enabled(slm_home):
        return results
    from superlocalmemory.truth import disputes, leases
    from superlocalmemory.truth.invariants import open_read_only

    try:
        conn = open_read_only(str(db_path))
    except Exception as exc:  # noqa: BLE001 - presentation only
        logger.debug("truth notes unavailable (%s)", type(exc).__name__)
        return results
    try:
        found = disputes.for_facts(conn, profile_id, fact_ids, hidden=hidden)
        sources = leases.check_for_facts(conn, fact_ids)
    except Exception as exc:  # noqa: BLE001 - presentation only, never fail a recall
        logger.debug("truth notes skipped (%s)", type(exc).__name__)
        return results
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
        out.append({**item, **extra} if extra else item)
    return out


__all__ = ["annotate", "enabled"]
