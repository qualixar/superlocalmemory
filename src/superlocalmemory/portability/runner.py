# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The import job: plan, save in batches through the normal writer, record, finish.

What it never does:

* hold a writer lock across a model or network call. It makes none: each item goes through the
  daemon's writer as one short save (``memory_core.submit``, the path pictures, documents and
  folders use), and the writer commits the memory and its first, keyword-searchable fact at
  once. The extraction that turns a conversation turn into facts (embedding, graph, a model in
  modes B and C) is the daemon's ordinary background enrichment of every save; this job only
  waits for that queue to shrink (``MAX_PENDING``) before sending the next batch. Run progress
  is one short statement between batches.
* overwrite or reuse an existing memory. Items already in the profile are planned out first
  (``dedupe``), and an earlier run's items are recognised by fingerprint, so importing the same
  file twice saves nothing the second time.
* starve anyone else. Between batches it lets recalls in flight finish (``recall_gate``) and
  rests briefly, so agents using SLM at the same time see no delay and no "database is locked".

Every save carries the origin channel ``import`` (class ``imported_file``, the bottom of the
lattice), so imported text never moves a slot head and never becomes a rule, a pin or a
session-start instruction without one recorded promotion.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from superlocalmemory.core import recall_gate
from superlocalmemory.memory_core.save_path import ContentOrigin
from superlocalmemory.memory_core.submit import SavePending, SaveRequest, submit_memory_settled
from superlocalmemory.portability import constants as C
from superlocalmemory.portability import dedupe, runs
from superlocalmemory.portability.model import TURN, Item, Parsed

logger = logging.getLogger(__name__)

SOURCE_TYPE = "http-import"  # origin channel "import": imported_file, bottom of the lattice
#: Saves still being enriched that make the job wait before the next batch (routing only).
MAX_PENDING = 300
PENDING_WAIT_S = 30.0
#: How often a save that the writer could not take is tried again before the run stops.
SAVE_ATTEMPTS = 5

#: Skip reasons this module adds to the ones the importers count (words in ``model.SKIP_WORDS``).
ALREADY_THERE = "already_there"
REJECTED = "rejected"


@dataclass(frozen=True)
class Context:
    """What a run needs from the daemon. ``db`` is the ``DatabaseManager``; ``runtime`` the
    daemon's writer. Who started the run is on the run row and in the decision log; the saves
    are written as the run's own actor (``import:<run id>``), so everything the save path keeps
    per actor (its trust score) belongs to the run and goes with it when the run is undone."""

    db: Any
    runtime: Any
    config: Any
    profile_id: str


@dataclass(frozen=True)
class Planned:
    index: int
    item: Item
    fingerprint: str


def plan(conn: Any, ctx_profile: str, config: Any, parsed: Parsed) -> tuple[list[Planned], dict[str, int]]:
    """Which items to save, and counts of the ones left out because they are already there.

    ``conn`` is a read-only connection. Nothing is written.
    """
    skipped: Counter[str] = Counter()
    prints = [it.fingerprint(parsed.format) for it in parsed.items]
    known = runs.known_fingerprints(conn, ctx_profile, prints) if runs.has_tables(conn) else set()
    keys = [dedupe.key_of(dedupe.stored_form(it, config)) for it in parsed.items]
    present = dedupe.present_keys(conn, ctx_profile, keys)
    todo: list[Planned] = []
    seen: set[str] = set()
    for index, (item, fp, key) in enumerate(zip(parsed.items, prints, keys, strict=True)):
        if fp in known or key in present:
            skipped[ALREADY_THERE] += 1
        elif key in seen:
            skipped["duplicate"] += 1
        else:
            seen.add(key)
            todo.append(Planned(index, item, fp))
    return todo, dict(skipped)


def _request(ctx: Context, run_id: str, fmt: str, planned: Planned) -> SaveRequest:
    item = planned.item
    turn = item.kind == TURN
    return SaveRequest(
        segments=((item.text, ContentOrigin.DERIVED_TEXT),), profile_id=ctx.profile_id,
        source_type=SOURCE_TYPE, trusted_actor_id=f"import:{run_id}",
        session_date=(item.session_at or item.created_at) if (turn or item.created_at) else "",
        session_id=f"import:{fmt}:{item.session_key}"[:200] if turn else "",
        speaker=item.speaker if turn else "", role=(item.role or "user") if turn else "user",
        scope="personal", shared_with=(),
        trusted_metadata={"_slm_import": {"run": run_id, "format": fmt}},
        idempotency_key=f"imp:{run_id}:{planned.index}")


class _Refused(Exception):
    """One item the writer refused for good (too little to remember); the run goes on."""


class _Stopped(Exception):
    """The run cannot go on; the message is plain words for a person."""


def _save_one(ctx: Context, request: SaveRequest, stop: threading.Event) -> str:
    """Save one item. The memory id, ``_Refused`` for good, or ``_Stopped`` if the writer is gone."""
    from superlocalmemory.core.ingestion_command import IngestionRejectedError
    from superlocalmemory.core.remember_admission import AdmissionRejected
    from superlocalmemory.core.remember_runtime import CanonicalRememberUnavailable
    from superlocalmemory.storage.admission_journal import TerminalAdmissionError
    from superlocalmemory.storage.write_coordinator import WriteCoordinatorError

    busy = (SavePending, CanonicalRememberUnavailable, WriteCoordinatorError)
    delay = 1.0
    for attempt in range(1, SAVE_ATTEMPTS + 1):
        try:
            receipt = submit_memory_settled(ctx.runtime, request, config=ctx.config)
        except (ValueError, IngestionRejectedError, AdmissionRejected, TerminalAdmissionError):
            raise _Refused from None
        except busy:
            if attempt == SAVE_ATTEMPTS or stop.wait(delay):
                raise _Stopped("SuperLocalMemory was too busy to take more just now. "
                               "What was saved so far is kept; you can undo it or run it "
                               "again to add the rest.") from None
            delay = min(delay * 2, 8.0)
            continue
        except Exception as exc:  # noqa: BLE001 - one plain message; the class goes to the log
            logger.warning("import: a save failed (%s)", type(exc).__name__)
            raise _Stopped("A save failed, so the import stopped. What was saved so far "
                           "is kept; you can undo it.") from None
        if not receipt.memory_id:
            raise _Stopped("A saved memory could not be tracked, so the import stopped to keep "
                           "Undo exact. You can undo what was saved.")
        return receipt.memory_id
    raise _Stopped("The import stopped.")  # pragma: no cover - the loop always returns or raises


def _pending(ctx: Context) -> int:
    try:
        rows = ctx.db.execute(
            "SELECT COUNT(*) AS c FROM ingestion_operations WHERE profile_id = ? "
            "AND state IN ('raw', 'queryable', 'enriching')", (ctx.profile_id,))
        return int(dict(rows[0])["c"]) if rows else 0
    except Exception:  # noqa: BLE001 - not knowing the backlog never stops an import
        return 0


def _let_others_through(ctx: Context, stop: threading.Event) -> None:
    """Between batches: recalls in flight finish, and the enrichment queue may shrink."""
    recall_gate.yield_to_recalls(C.YIELD_MAX_S)
    deadline = time.monotonic() + PENDING_WAIT_S
    while _pending(ctx) > MAX_PENDING and time.monotonic() < deadline and not stop.is_set():
        stop.wait(1.0)
    stop.wait(C.REST_BETWEEN_BATCHES_S)


def _write(ctx: Context, work: Any) -> Any:
    """One short write transaction (never held across anything slow)."""
    with ctx.db.raw_connection() as conn:
        return work(conn)


def _cancelled(ctx: Context, run_id: str, stop: threading.Event) -> bool:
    if stop.is_set():
        return True
    rows = ctx.db.execute("SELECT cancel_requested FROM import_runs WHERE run_id = ?", (run_id,))
    return bool(rows and dict(rows[0])["cancel_requested"])


def execute(ctx: Context, run_id: str, parsed: Parsed, stop: threading.Event) -> str:
    """Run an import to its end. Returns the final state. Never raises."""
    try:
        return _execute(ctx, run_id, parsed, stop)
    except _Stopped as stopped:
        words = str(stopped)
        _write(ctx, lambda c: runs.set_state(c, run_id, runs.FAILED, note=words))
        return runs.FAILED
    except Exception:  # noqa: BLE001 - a bug must end the run cleanly, with words, not a stack
        logger.exception("import run %s failed", run_id)
        _write(ctx, lambda c: runs.set_state(
            c, run_id, runs.FAILED, note="The import stopped on an unexpected problem. What was "
                                         "saved so far is kept; you can undo it."))
        return runs.FAILED


def _execute(ctx: Context, run_id: str, parsed: Parsed, stop: threading.Event) -> str:
    from superlocalmemory.truth.invariants import open_read_only

    _write(ctx, lambda c: runs.set_state(c, run_id, runs.RUNNING, only_from=(runs.QUEUED,)))
    reader = open_read_only(str(ctx.db.db_path))
    try:
        todo, left_out = plan(reader, ctx.profile_id, ctx.config, parsed)
    finally:
        reader.close()
    if left_out:
        _write(ctx, lambda c: runs.record_batch(c, run_id, ctx.profile_id, [], left_out))
    for start in range(0, len(todo), C.BATCH):
        if _cancelled(ctx, run_id, stop):
            _write(ctx, lambda c: runs.set_state(
                c, run_id, runs.CANCELLED, note="Stopped. What was saved so far is kept."))
            return runs.CANCELLED
        saved: list[tuple[str, str]] = []
        extra: Counter[str] = Counter()
        try:
            for planned in todo[start:start + C.BATCH]:
                try:
                    saved.append((_save_one(ctx, _request(ctx, run_id, parsed.format, planned),
                                            stop), planned.fingerprint))
                except _Refused:
                    extra[REJECTED] += 1
        finally:  # whatever was saved is recorded, even when the run stops on the next item
            _write(ctx, lambda c: runs.record_batch(c, run_id, ctx.profile_id, saved, dict(extra)))
        _let_others_through(ctx, stop)
    _write(ctx, lambda c: runs.set_state(c, run_id, runs.DONE, note=""))
    return runs.DONE


__all__ = ["ALREADY_THERE", "Context", "MAX_PENDING", "Planned", "REJECTED", "SOURCE_TYPE",
           "execute", "plan"]
