"""Review-gated correction-case ledger and atomic temporal lifecycle seam.

The ledger stores identifiers and temporal snapshots only, never raw fact
text.  A trusted reviewer may apply or roll back a case inside the same SQLite
transaction that changes the predecessor's temporal lifecycle.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping
from uuid import uuid4

from superlocalmemory.storage.correction_dismissals import dismissed_in_bulk
from superlocalmemory.storage.write_lock import get_write_lock

_SCOPES = frozenset({"personal", "project", "shared", "global"})
_STATUSES = frozenset({"proposed", "applied", "rejected", "rolled_back"})
_EVENTS = frozenset({"proposed", "applied", "rejected", "rolled_back"})
_MAX_FIELD_LENGTH = 128
#: Reason code of a case the caller recorded by saving a memory with
#: ``replaces`` (core/remember_replaces.py). Proposed and applied in one
#: transaction, so it is never pending review; and the caller's new memory
#: stands on its own, so undoing the case restores the old fact without
#: withdrawing the new one (see the successor checks in storage/database.py).
CALLER_REPLACEMENT_REASON = "replaced_by_caller"
#: Reason codes of the cases the SAVE PATH files when it only guesses that a new
#: memory updates an older one (core/store_pipeline._record_correction_candidate):
#: the temporal check, and every consolidation action ("consolidation_<action>").
#: Their successor is the memory the caller saved, so an unreviewed, rejected or
#: rolled-back guess never withholds it (Varun, 2026-10-08: both stay findable
#: until a person reviews; approving retires the older one).
TEMPORAL_CANDIDATE_REASON = "temporal_contradiction"
CONSOLIDATION_CANDIDATE_PREFIX = "consolidation_"
#: SQL over ``alias``: true when the case's successor is NOT a saved memory, i.e.
#: its pending review may withhold it. Takes ``SAVED_SUCCESSOR_PARAMS``.
SAVED_SUCCESSOR_SQL = (
    "{alias}.reason_code NOT IN (?, ?) "
    "AND substr({alias}.reason_code, 1, ?) != ?"
)
SAVED_SUCCESSOR_PARAMS = (CALLER_REPLACEMENT_REASON, TEMPORAL_CANDIDATE_REASON,
                          len(CONSOLIDATION_CANDIDATE_PREFIX),
                          CONSOLIDATION_CANDIDATE_PREFIX)


class CorrectionCaseError(RuntimeError):
    """Base exception for correction-case lifecycle failures."""


class CorrectionAuthorizationError(CorrectionCaseError):
    """The host/server did not attest the acting identity as trusted."""


class CorrectionCompareAndSetError(CorrectionCaseError):
    """The caller attempted a state transition against a stale case version."""


class CorrectionFactErasingError(CorrectionCompareAndSetError):
    """A fact the case names is being deleted (its erasure tombstone exists)."""


class CorrectionIdempotencyError(CorrectionCaseError):
    """A replay key names a different correction proposal."""


class CorrectionDismissedError(CorrectionCaseError):
    """The user dismissed this machine proposal in bulk; it is not filed again."""


class CorrectionNotFoundError(CorrectionCaseError):
    """No correction case exists for the requested identifier."""


@dataclass(frozen=True, slots=True)
class CorrectionActor:
    """Actor provenance supplied by a host-authenticated integration seam."""

    actor_id: str
    actor_kind: str
    trust_tier: str


@dataclass(frozen=True, slots=True)
class CorrectionCase:
    """A fact-identifier-only correction decision and its lifecycle state."""

    case_id: str
    profile_id: str
    scope: str
    predecessor_fact_id: str
    successor_fact_id: str
    reason_code: str
    status: str
    version: int
    idempotency_key: str
    created_at: str
    updated_at: str
    reviewed_by_actor_id: str | None
    reviewed_at: str | None
    applied_at: str | None
    system_effective_at: str | None
    event_valid_from: str | None
    event_valid_until: str | None
    predecessor_temporal_existed: bool | None
    predecessor_valid_from: str | None
    predecessor_valid_until: str | None
    predecessor_system_created_at: str | None
    predecessor_system_expired_at: str | None
    predecessor_invalidated_by: str | None
    predecessor_invalidation_reason: str | None


class CorrectionCaseStore:
    """Own correction lifecycle metadata with explicit profile/trust seams.

    ``is_actor_trusted`` must be implemented by the server/host integration.
    This storage module never invents identity or authorization from an actor
    string, which keeps unverified clients from self-approving corrections.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        is_profile_active: Callable[[str], bool],
        is_actor_trusted: Callable[[CorrectionActor], bool],
    ) -> None:
        self._path = Path(path)
        self._is_profile_active = is_profile_active
        self._is_actor_trusted = is_actor_trusted

    @property
    def path(self) -> str:
        return str(self._path)

    def propose(
        self,
        *,
        case_id: str,
        profile_id: str,
        scope: str,
        predecessor_fact_id: str,
        successor_fact_id: str,
        reason_code: str,
        actor: CorrectionActor,
        idempotency_key: str,
        event_valid_from: str | None = None,
        event_valid_until: str | None = None,
    ) -> CorrectionCase:
        """Record a candidate without changing facts or retrieval behaviour."""
        with self._transaction() as conn:
            return propose_on_connection(
                conn,
                case_id=case_id,
                profile_id=profile_id,
                scope=scope,
                predecessor_fact_id=predecessor_fact_id,
                successor_fact_id=successor_fact_id,
                reason_code=reason_code,
                actor=actor,
                idempotency_key=idempotency_key,
                event_valid_from=event_valid_from,
                event_valid_until=event_valid_until,
                is_profile_active=self._is_profile_active,
                is_actor_trusted=self._is_actor_trusted,
            )

    def apply(
        self,
        case_id: str,
        *,
        expected_version: int,
        actor: CorrectionActor,
        operation_id: str,
        event_valid_until: str | None = None,
    ) -> CorrectionCase:
        """Atomically approve a case and supersede its scoped predecessor."""
        return self._transition(
            case_id, expected_version=expected_version, actor=actor,
            operation_id=operation_id, from_status="proposed", to_status="applied",
            mutate_temporal=True,
            event_valid_until=event_valid_until,
        )

    def reject(
        self, case_id: str, *, expected_version: int, actor: CorrectionActor, operation_id: str
    ) -> CorrectionCase:
        """Record a reviewed rejection atomically, preserving all history."""
        return self._transition(
            case_id, expected_version=expected_version, actor=actor,
            operation_id=operation_id, from_status="proposed", to_status="rejected",
        )

    def rollback(
        self, case_id: str, *, expected_version: int, actor: CorrectionActor, operation_id: str
    ) -> CorrectionCase:
        """Atomically restore the predecessor tuple and append a rollback event."""
        return self._transition(
            case_id, expected_version=expected_version, actor=actor,
            operation_id=operation_id, from_status="applied", to_status="rolled_back",
            mutate_temporal=True,
        )

    def get_case(self, case_id: str) -> CorrectionCase:
        """Read one active-profile case without exposing fact text."""
        _validate_field("case_id", case_id)
        with self._read_connection() as conn:
            case = _get_case(conn, case_id)
        if not self._is_profile_active(case.profile_id):
            raise CorrectionAuthorizationError("profile is inactive or closing")
        return case

    def list_cases(self, profile_id: str, *, limit: int = 100) -> list[CorrectionCase]:
        """List bounded active-profile cases, newest first, identifiers only."""
        _validate_field("profile_id", profile_id)
        if not self._is_profile_active(profile_id):
            raise CorrectionAuthorizationError("profile is inactive or closing")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer from 1 to 500")
        with self._read_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM correction_cases WHERE profile_id=? "
                "ORDER BY updated_at DESC, case_id DESC LIMIT ?",
                (profile_id, limit),
            ).fetchall()
        return [_case_from_row(row) for row in rows]

    def _transition(
        self,
        case_id: str,
        *,
        expected_version: int,
        actor: CorrectionActor,
        operation_id: str,
        from_status: Literal["proposed", "applied"],
        to_status: Literal["applied", "rejected", "rolled_back"],
        mutate_temporal: bool = False,
        event_valid_until: str | None = None,
    ) -> CorrectionCase:
        with self._transaction() as conn:
            return transition_on_connection(
                conn,
                case_id=case_id,
                expected_version=expected_version,
                actor=actor,
                operation_id=operation_id,
                from_status=from_status,
                to_status=to_status,
                mutate_temporal=mutate_temporal,
                event_valid_until=event_valid_until,
                is_profile_active=self._is_profile_active,
                is_actor_trusted=self._is_actor_trusted,
            )

    def _assert_admitted(self, profile_id: str, actor: CorrectionActor) -> None:
        if not self._is_profile_active(profile_id):
            raise CorrectionAuthorizationError("profile is inactive or closing")
        if not self._is_actor_trusted(actor):
            raise CorrectionAuthorizationError("actor is not trusted by the host/server")

    def _transaction(self):
        return _CorrectionTransaction(self._path)

    def _read_connection(self):
        return _CorrectionReadConnection(self._path)


def propose_on_connection(
    conn: sqlite3.Connection,
    *,
    case_id: str,
    profile_id: str,
    scope: str,
    predecessor_fact_id: str,
    successor_fact_id: str,
    reason_code: str,
    actor: CorrectionActor,
    idempotency_key: str,
    event_valid_from: str | None = None,
    event_valid_until: str | None = None,
    is_profile_active: Callable[[str], bool],
    is_actor_trusted: Callable[[CorrectionActor], bool],
) -> CorrectionCase:
    """Write one candidate through the caller-owned SQLite transaction.

    This function never starts, commits, rolls back, or closes ``conn``. It is
    the only storage entry point suitable for a canonical writer command or a
    bound ``DatabaseManager.raw_connection()`` scope.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("correction proposal requires a sqlite3 connection")
    _validate_case_input(
        case_id, profile_id, scope, predecessor_fact_id, successor_fact_id,
        reason_code, actor, idempotency_key, event_valid_from, event_valid_until,
    )
    if not is_profile_active(profile_id):
        raise CorrectionAuthorizationError("profile is inactive or closing")
    if not is_actor_trusted(actor):
        raise CorrectionAuthorizationError("actor is not trusted by the host/server")

    existing = conn.execute(
        "SELECT * FROM correction_cases WHERE profile_id=? AND idempotency_key=?",
        (profile_id, idempotency_key),
    ).fetchone()
    if existing is not None:
        case = _case_from_row(existing)
        if (
            case.case_id != case_id
            or case.scope != scope
            or case.predecessor_fact_id != predecessor_fact_id
            or case.successor_fact_id != successor_fact_id
            or case.reason_code != reason_code
        ):
            raise CorrectionIdempotencyError("proposal replay key names different data")
        return case

    if (
        actor.actor_kind == "host_attested"  # only the save path's own guesses are remembered
        and (reason_code == TEMPORAL_CANDIDATE_REASON
             or reason_code.startswith(CONSOLIDATION_CANDIDATE_PREFIX))
        and dismissed_in_bulk(conn, profile_id, case_id)
    ):
        raise CorrectionDismissedError(
            "this machine proposal was dismissed by the user and is not filed again")
    refuse_if_being_deleted(conn, (predecessor_fact_id, successor_fact_id), "proposed")
    now = _now()
    conn.execute(
        "INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
        "successor_fact_id, reason_code, status, version, idempotency_key, "
        "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, created_at, "
        "updated_at, event_valid_from, event_valid_until) "
        "VALUES (?, ?, ?, ?, ?, ?, 'proposed', 0, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            case_id, profile_id, scope, predecessor_fact_id, successor_fact_id,
            reason_code, idempotency_key, actor.actor_id, actor.actor_kind,
            actor.trust_tier, now, now, event_valid_from, event_valid_until,
        ),
    )
    _append_event(
        conn, case_id=case_id, profile_id=profile_id, scope=scope,
        event_type="proposed", operation_id=idempotency_key, actor=actor,
        expected_version=None, resulting_version=0, occurred_at=now,
        event_valid_from=event_valid_from, event_valid_until=event_valid_until,
    )
    return _get_case(conn, case_id)


def open_case_id(
    conn: sqlite3.Connection, profile_id: str, predecessor_fact_id: str,
) -> str | None:
    """Return the id of the proposed or applied case for a memory, if any.

    The ledger keeps at most one such case per memory and profile (the partial
    unique index ``uq_correction_cases_active_predecessor``), so this is the
    question a writer asks before proposing another.
    """
    row = conn.execute(
        "SELECT case_id FROM correction_cases WHERE profile_id=? "
        "AND predecessor_fact_id=? AND status IN ('proposed', 'applied') LIMIT 1",
        (profile_id, predecessor_fact_id),
    ).fetchone()
    return None if row is None else str(row[0])


def transition_on_connection(
    conn: sqlite3.Connection,
    *,
    case_id: str,
    expected_version: int,
    actor: CorrectionActor,
    operation_id: str,
    from_status: Literal["proposed", "applied"],
    to_status: Literal["applied", "rejected", "rolled_back"],
    mutate_temporal: bool,
    event_valid_until: str | None = None,
    is_profile_active: Callable[[str], bool],
    is_actor_trusted: Callable[[CorrectionActor], bool],
) -> CorrectionCase:
    """Transition one case in a caller-owned transaction, never nesting it."""
    _validate_field("case_id", case_id)
    _validate_field("operation_id", operation_id)
    if expected_version < 0:
        raise ValueError("expected_version must be non-negative")
    _validate_actor(actor)
    if not is_actor_trusted(actor):
        raise CorrectionAuthorizationError("actor is not trusted by the host/server")
    replay = conn.execute(
        "SELECT event_type FROM correction_events WHERE case_id=? AND operation_id=?",
        (case_id, operation_id),
    ).fetchone()
    if replay is not None:
        if replay[0] != to_status:
            raise CorrectionIdempotencyError("operation replay key has another transition")
        return _get_case(conn, case_id)
    case = _get_case(conn, case_id)
    if not is_profile_active(case.profile_id):
        raise CorrectionAuthorizationError("profile is inactive or closing")
    if case.status != from_status or case.version != expected_version:
        raise CorrectionCompareAndSetError("case state changed; reload before review action")
    refuse_if_being_deleted(conn, (case.predecessor_fact_id, case.successor_fact_id),
                            to_status.replace("_", " "))
    if to_status == "rolled_back":
        dependent = conn.execute(
            "SELECT 1 FROM correction_cases WHERE profile_id=? "
            "AND predecessor_fact_id=? AND status IN ('proposed', 'applied') LIMIT 1",
            (case.profile_id, case.successor_fact_id),
        ).fetchone()
        if dependent is not None:
            raise CorrectionCompareAndSetError(
                "dependent correction is active; resolve it before rollback"
            )
    now = _now()
    if event_valid_until is not None:
        if to_status != "applied":
            raise ValueError("event-valid boundary is permitted only when applying a correction")
        _validate_timestamp("event_valid_until", event_valid_until)
        case = replace(case, event_valid_until=event_valid_until)
    if mutate_temporal:
        if to_status == "applied":
            _apply_predecessor_temporal(conn, case, occurred_at=now)
        else:
            _restore_predecessor_temporal(conn, case)
    next_version = case.version + 1
    cursor = conn.execute(
        "UPDATE correction_cases SET status=?, version=?, updated_at=?, event_valid_until=?, "
        "reviewed_by_actor_id=?, reviewed_at=?, applied_at=?, system_effective_at=? "
        "WHERE case_id=? AND status=? AND version=?",
        (
            to_status,
            next_version,
            now,
            case.event_valid_until,
            actor.actor_id,
            now,
            now if to_status == "applied" else case.applied_at,
            now if to_status == "applied" else case.system_effective_at,
            case_id,
            from_status,
            expected_version,
        ),
    )
    if cursor.rowcount != 1:
        raise CorrectionCompareAndSetError("case state changed; reload before review action")
    _append_event(
        conn,
        case_id=case.case_id,
        profile_id=case.profile_id,
        scope=case.scope,
        event_type=to_status,
        operation_id=operation_id,
        actor=actor,
        expected_version=expected_version,
        resulting_version=next_version,
        occurred_at=now,
        event_valid_from=case.event_valid_from,
        event_valid_until=case.event_valid_until,
    )
    if to_status == "rolled_back" and case.reason_code == CALLER_REPLACEMENT_REASON:
        # Undoing a caller's whole-memory replacement also restores the facts
        # its enrichment derived after it (storage/replaced_memory.py).
        from superlocalmemory.storage.replaced_memory import restore_late_facts

        restore_late_facts(
            conn, case, actor=actor, operation_id=operation_id,
            is_profile_active=is_profile_active, is_actor_trusted=is_actor_trusted,
        )
    return _get_case(conn, case_id)


def refuse_if_being_deleted(conn: sqlite3.Connection, fact_ids: tuple[str, ...],
                            action: str) -> None:
    """Refuse while a delete of either fact is in progress (4.1.22).

    A delete writes the fact's erasure tombstone and removes its search entries
    before its last protection check; a case proposed or reviewed in that window
    would turn the delete into a late refusal. Asked inside the case's own write
    transaction, so it cannot interleave with the tombstone write.
    """
    ph = ",".join("?" * len(fact_ids))
    try:
        row = conn.execute(f"SELECT fact_id FROM projection_tombstones "  # noqa: S608
                           f"WHERE fact_id IN ({ph}) LIMIT 1", fact_ids).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return  # an older store without erasure tombstones
        raise
    if row is not None:
        raise CorrectionFactErasingError(
            f"memory {row[0]} is being deleted, so this correction cannot be {action}; "
            "nothing was changed")


def _append_event(
    conn: sqlite3.Connection,
    *,
    case_id: str,
    profile_id: str,
    scope: str,
    event_type: str,
    operation_id: str,
    actor: CorrectionActor,
    expected_version: int | None,
    resulting_version: int,
    occurred_at: str,
    event_valid_from: str | None,
    event_valid_until: str | None,
) -> None:
    conn.execute(
        "INSERT INTO correction_events (event_id, case_id, profile_id, scope, event_type, "
        "operation_id, actor_id, actor_kind, actor_trust_tier, expected_version, "
        "resulting_version, system_occurred_at, event_valid_from, event_valid_until) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            uuid4().hex, case_id, profile_id, scope, event_type, operation_id,
            actor.actor_id, actor.actor_kind, actor.trust_tier, expected_version,
            resulting_version, occurred_at, event_valid_from, event_valid_until,
        ),
    )


def _get_case(conn: sqlite3.Connection, case_id: str) -> CorrectionCase:
    row = conn.execute("SELECT * FROM correction_cases WHERE case_id=?", (case_id,)).fetchone()
    if row is None:
        raise CorrectionNotFoundError("correction case does not exist")
    return _case_from_row(row)


def _apply_predecessor_temporal(
    conn: sqlite3.Connection, case: CorrectionCase, *, occurred_at: str
) -> None:
    """Snapshot and supersede exactly one profile-and-scope-owned predecessor."""
    rows = conn.execute(
        "SELECT fact_id, profile_id, scope FROM atomic_facts "
        "WHERE fact_id IN (?, ?)",
        (case.predecessor_fact_id, case.successor_fact_id),
    ).fetchall()
    facts = {str(row["fact_id"]): row for row in rows}
    if set(facts) != {case.predecessor_fact_id, case.successor_fact_id}:
        raise CorrectionNotFoundError("correction facts are no longer available")
    for row in facts.values():
        if row["profile_id"] != case.profile_id or row["scope"] != case.scope:
            raise CorrectionAuthorizationError("correction facts are outside the approved scope")

    prior = conn.execute(
        "SELECT valid_from, valid_until, system_created_at, system_expired_at, "
        "invalidated_by, invalidation_reason FROM fact_temporal_validity "
        "WHERE fact_id=? AND profile_id=?",
        (case.predecessor_fact_id, case.profile_id),
    ).fetchone()
    if prior is None:
        conn.execute(
            "UPDATE correction_cases SET predecessor_temporal_existed=0, "
            "predecessor_valid_from=NULL, predecessor_valid_until=NULL, "
            "predecessor_system_created_at=NULL, predecessor_system_expired_at=NULL, "
            "predecessor_invalidated_by=NULL, predecessor_invalidation_reason=NULL "
            "WHERE case_id=?",
            (case.case_id,),
        )
        conn.execute(
            "INSERT INTO fact_temporal_validity "
            "(fact_id, profile_id, system_created_at) VALUES (?, ?, ?)",
            (case.predecessor_fact_id, case.profile_id, occurred_at),
        )
    else:
        conn.execute(
            "UPDATE correction_cases SET predecessor_temporal_existed=1, "
            "predecessor_valid_from=?, predecessor_valid_until=?, "
            "predecessor_system_created_at=?, predecessor_system_expired_at=?, "
            "predecessor_invalidated_by=?, predecessor_invalidation_reason=? "
            "WHERE case_id=?",
            (*tuple(prior), case.case_id),
        )
    cursor = conn.execute(
        "UPDATE fact_temporal_validity SET "
        "valid_until=COALESCE(?, valid_until), system_expired_at=?, "
        "invalidated_by=?, invalidation_reason=? WHERE fact_id=? AND profile_id=?",
        (
            case.event_valid_until,
            occurred_at,
            case.successor_fact_id,
            case.reason_code,
            case.predecessor_fact_id,
            case.profile_id,
        ),
    )
    if cursor.rowcount != 1:
        # Defence in depth, not a known live failure: the read above and the
        # insert-if-missing use this same (fact_id, profile_id), inside one
        # serialized transaction, so a zero-row update should be unreachable.
        # It is asserted because the alternative is review_correction returning
        # "applied" having retired nothing -- and supersession is the ONE
        # signal every read path now resolves its answer from. Its sibling
        # _restore_predecessor_temporal has always checked; this one did not.
        raise CorrectionNotFoundError(
            "predecessor temporal record was not superseded"
        )


def _restore_predecessor_temporal(conn: sqlite3.Connection, case: CorrectionCase) -> None:
    """Restore the snapshot captured at apply; history rows themselves remain."""
    if case.predecessor_temporal_existed is None:
        raise CorrectionCaseError("approved case has no predecessor temporal snapshot")
    if not case.predecessor_temporal_existed:
        conn.execute(
            "DELETE FROM fact_temporal_validity WHERE fact_id=? AND profile_id=?",
            (case.predecessor_fact_id, case.profile_id),
        )
        return
    cursor = conn.execute(
        "UPDATE fact_temporal_validity SET valid_from=?, valid_until=?, system_created_at=?, "
        "system_expired_at=?, invalidated_by=?, invalidation_reason=? "
        "WHERE fact_id=? AND profile_id=?",
        (
            case.predecessor_valid_from,
            case.predecessor_valid_until,
            case.predecessor_system_created_at,
            case.predecessor_system_expired_at,
            case.predecessor_invalidated_by,
            case.predecessor_invalidation_reason,
            case.predecessor_fact_id,
            case.profile_id,
        ),
    )
    if cursor.rowcount != 1:
        raise CorrectionNotFoundError("predecessor temporal record is no longer available")


class _CorrectionTransaction:
    """Small transaction guard: all case/event writes commit or roll back together."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None
        # Standalone lifecycle operations are rare, but must serialize with
        # DatabaseManager writes just like coordinator-owned operations do.
        self._lock = get_write_lock(path)

    def __enter__(self) -> sqlite3.Connection:
        self._lock.acquire()
        try:
            conn = sqlite3.connect(str(self._path), timeout=5, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("BEGIN IMMEDIATE")
                self._conn = conn
            except BaseException:
                # Setup failed: close it now, or the file stays open (Windows
                # then cannot delete or replace it) for as long as the error lives.
                conn.close()
                raise
            return conn
        except Exception:
            self._lock.release()
            raise

    def __exit__(self, exc_type, exc, _traceback) -> None:
        assert self._conn is not None
        try:
            self._conn.execute("ROLLBACK" if exc_type is not None else "COMMIT")
        finally:
            try:
                self._conn.close()
            finally:
                self._lock.release()


class _CorrectionReadConnection:
    """Small read-only guard kept out of the canonical writer domain."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=0.25)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA query_only=ON")
            self._conn = conn
        except BaseException:
            # Setup failed: close it now, or the file stays open (Windows
            # then cannot delete or replace it) for as long as the error lives.
            conn.close()
            raise
        return conn

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        assert self._conn is not None
        self._conn.close()


def _case_from_row(
    row: sqlite3.Row | tuple[object, ...] | Mapping[str, object],
) -> CorrectionCase:
    if not isinstance(row, sqlite3.Row):
        columns = (
            "case_id", "profile_id", "scope", "predecessor_fact_id", "successor_fact_id",
            "reason_code", "status", "version", "idempotency_key", "proposed_by_actor_id",
            "proposed_by_actor_kind", "proposed_by_trust_tier", "created_at", "updated_at",
            "reviewed_by_actor_id", "reviewed_at", "applied_at", "system_effective_at",
            "event_valid_from", "event_valid_until", "predecessor_temporal_existed",
            "predecessor_valid_from", "predecessor_valid_until", "predecessor_system_created_at",
            "predecessor_system_expired_at", "predecessor_invalidated_by",
            "predecessor_invalidation_reason",
        )
        row = dict(zip(columns, row, strict=True))
    return CorrectionCase(
        case_id=row["case_id"], profile_id=row["profile_id"], scope=row["scope"],
        predecessor_fact_id=row["predecessor_fact_id"], successor_fact_id=row["successor_fact_id"],
        reason_code=row["reason_code"], status=row["status"], version=int(row["version"]),
        idempotency_key=row["idempotency_key"], created_at=row["created_at"],
        updated_at=row["updated_at"], reviewed_by_actor_id=row["reviewed_by_actor_id"],
        reviewed_at=row["reviewed_at"], applied_at=row["applied_at"],
        system_effective_at=row["system_effective_at"], event_valid_from=row["event_valid_from"],
        event_valid_until=row["event_valid_until"],
        predecessor_temporal_existed=(
            None if row["predecessor_temporal_existed"] is None
            else bool(row["predecessor_temporal_existed"])
        ),
        predecessor_valid_from=row["predecessor_valid_from"],
        predecessor_valid_until=row["predecessor_valid_until"],
        predecessor_system_created_at=row["predecessor_system_created_at"],
        predecessor_system_expired_at=row["predecessor_system_expired_at"],
        predecessor_invalidated_by=row["predecessor_invalidated_by"],
        predecessor_invalidation_reason=row["predecessor_invalidation_reason"],
    )


def _validate_case_input(
    case_id: str, profile_id: str, scope: str, predecessor_fact_id: str, successor_fact_id: str,
    reason_code: str, actor: CorrectionActor, idempotency_key: str,
    event_valid_from: str | None, event_valid_until: str | None,
) -> None:
    for name, value in (
        ("case_id", case_id),
        ("profile_id", profile_id),
        ("predecessor_fact_id", predecessor_fact_id),
        ("successor_fact_id", successor_fact_id), ("reason_code", reason_code),
        ("idempotency_key", idempotency_key),
    ):
        _validate_field(name, value)
    if predecessor_fact_id == successor_fact_id:
        raise ValueError("predecessor and successor facts must differ")
    if scope not in _SCOPES:
        raise ValueError("scope is unsupported")
    _validate_actor(actor)
    _validate_timestamp("event_valid_from", event_valid_from)
    _validate_timestamp("event_valid_until", event_valid_until)
    if event_valid_from and event_valid_until and event_valid_from >= event_valid_until:
        raise ValueError("event-valid interval must be ordered")


def _validate_actor(actor: CorrectionActor) -> None:
    for name, value in (
        ("actor_id", actor.actor_id), ("actor_kind", actor.actor_kind),
        ("trust_tier", actor.trust_tier),
    ):
        _validate_field(name, value)


def _validate_field(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_FIELD_LENGTH:
        raise ValueError(f"{name} must be a non-empty bounded identifier")
    if any(ch.isspace() for ch in value) or "\x00" in value:
        raise ValueError(f"{name} must be a safe identifier")


def _validate_timestamp(name: str, value: str | None) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an RFC3339 timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC3339 timestamp") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
