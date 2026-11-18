# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The audit-trail check for a screen that asks again and again.

Walking a trail of a hundred thousand entries hashes every one of them and takes
seconds of CPU. The Compliance and Health panes ask on every visit, so asking
for the full walk each time stalled everything else the daemon was doing.

This keeps what the last complete walk established (how far the chain was
checked, and the hash it ended on) and, on the next ask, checks only the entries
added since, plus the pieces that do not depend on how many there are: that the
entry we stopped at is still the same entry, that the count matches, and the
external anchor. Entries that were already checked are walked again in full in
the background every ``FULL_RECHECK_S`` seconds, so an edit to an old entry is
still found; it is never found by skipping the walk silently, and the command
line check (``verify_report``) is always the complete one.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from superlocalmemory.compliance.audit import _GENESIS_HASH, FULL_VERSIONS, AuditChain
from superlocalmemory.compliance.audit_keyed import Tally

logger = logging.getLogger(__name__)

#: A failed result is kept this long, so a broken trail is not re-walked per request.
FAILURE_TTL_S = 30.0
#: Entries already checked are walked again in full this often (in the background).
FULL_RECHECK_S = 120.0


@dataclass(frozen=True)
class _State:
    report: dict[str, Any]
    through_id: int
    through_hash: str
    seen_v2: bool
    full_at: float
    at: float
    seen_keyed: bool = False
    active_key: str | None = None


_states: dict[str, _State] = {}
_locks: dict[str, threading.Lock] = {}
_refreshing: set[str] = set()
_registry = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _registry:
        return _locks.setdefault(key, threading.Lock())


def forget() -> None:
    """Drop everything remembered (tests, and a store that was replaced)."""
    with _registry:
        _states.clear()
        _refreshing.clear()


def cached_report(chain: AuditChain) -> dict[str, Any]:
    """``chain.verify_report()``, answered from the last walk plus what was added since."""
    if chain._is_memory:
        return chain.verify_report()
    key = chain._db_path
    with _lock_for(key):
        state = _states.get(key)
        if state is None:
            return copy.deepcopy(_full(chain, key).report)
        if not state.report["ok"]:
            if time.monotonic() - state.at < FAILURE_TTL_S:
                return copy.deepcopy(state.report)
            return copy.deepcopy(_full(chain, key).report)
        state = _incremental(chain, key, state)
    if state.report["ok"] and time.monotonic() - state.full_at > FULL_RECHECK_S:
        _refresh_in_background(chain, key)
    return copy.deepcopy(state.report)


def _full(chain: AuditChain, key: str) -> _State:
    rows = chain.load_rows()
    state = _state_from(rows, chain.verify_loaded(rows))
    with _registry:
        _states[key] = state
    return state


def _state_from(rows: list[dict], report: dict[str, Any]) -> _State:
    now = time.monotonic()
    last = rows[-1] if rows else None
    return _State(
        report=report,
        through_id=int(last["id"]) if last else 0,
        through_hash=last["event_hash"] if last else _GENESIS_HASH,
        seen_v2=any(r.get("hash_version", 1) >= 2 for r in rows),
        full_at=now, at=now, seen_keyed=_seen_keyed(rows), active_key=_active_key(rows))


def _seen_keyed(rows: list[dict]) -> bool:
    return any(r.get("hash_version") == 3 for r in rows)


def _active_key(rows: list[dict]) -> str | None:
    """The key id of the latest anchor: where a walk of the entries added later resumes."""
    for r in reversed(rows):
        if r.get("hash_version") == 3 and r.get("operation") == "key_anchor":
            return r.get("key_id") or None
    return None


def _incremental(chain: AuditChain, key: str, state: _State) -> _State:
    conn = chain._get_conn()
    try:
        if state.through_id:
            row = conn.execute("SELECT event_hash FROM audit_chain WHERE id = ?",
                               (state.through_id,)).fetchone()
            if row is None or row[0] != state.through_hash:
                return _full(chain, key)
        added = [dict(r) for r in conn.execute(
            "SELECT id, timestamp, operation, agent_id, profile_id, content_hash, "
            "prev_hash, event_hash, metadata, hash_version, key_id FROM audit_chain "
            "WHERE id > ? ORDER BY id", (state.through_id,)).fetchall()]
        total = int(conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0])
    finally:
        chain._release_conn(conn)
    before = state.report["entries"]
    if total != before + len(added):
        return _full(chain, key)  # something earlier was removed: walk it all
    tally = Tally(seen_keyed=state.seen_keyed, active_key=state.active_key)
    failure, last_hash, seen_v2 = chain._link_walk(
        added, state.through_hash, state.seen_v2, chain._signer(), tally)
    if failure is None:
        failure = chain._anchor_failure_at(total, lambda n: _entry_at(chain, n), total > 0)
    report = dict(state.report)
    report["entries"] = total
    report["legacy"] += sum(1 for r in added if r.get("hash_version", 1) == 1)
    report["fully_covered"] += sum(1 for r in added if r.get("hash_version") in FULL_VERSIONS)
    for name, count in tally.as_report().items():
        report[name] = report.get(name, 0) + count
    if failure is not None:
        report.update(ok=False, failed_at=failure[0], reason=failure[1])
        logger.warning("Audit chain verification failed at %s: %s", *failure)
    now = time.monotonic()
    result = _State(report=report,
                    through_id=int(added[-1]["id"]) if added else state.through_id,
                    through_hash=last_hash, seen_v2=seen_v2, full_at=state.full_at, at=now,
                    seen_keyed=tally.seen_keyed, active_key=tally.active_key)
    with _registry:
        _states[key] = result
    return result


def _entry_at(chain: AuditChain, position: int) -> tuple[int, str]:
    conn = chain._get_conn()
    try:
        row = conn.execute("SELECT id, event_hash FROM audit_chain ORDER BY id LIMIT 1 OFFSET ?",
                           (position,)).fetchone()
    finally:
        chain._release_conn(conn)
    return (int(row[0]), row[1]) if row else (0, "")


def _refresh_in_background(chain: AuditChain, key: str) -> None:
    with _registry:
        if key in _refreshing:
            return
        _refreshing.add(key)

    def work() -> None:
        try:
            rows = chain.load_rows()
            fresh = _state_from(rows, chain.verify_loaded(rows))
            with _registry:
                _states[key] = fresh
        except Exception:  # noqa: BLE001 - the next ask walks again
            logger.debug("audit full re-check failed", exc_info=True)
        finally:
            with _registry:
                _refreshing.discard(key)

    threading.Thread(target=work, name="slm-audit-recheck", daemon=True).start()


def warm(chain: AuditChain) -> None:
    """Do the first complete walk now, so the first visit to the pane is instant."""
    try:
        cached_report(chain)
    except Exception:  # noqa: BLE001 - warming is best effort
        logger.debug("audit warm-up failed", exc_info=True)
