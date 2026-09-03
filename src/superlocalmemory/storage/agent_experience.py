"""Bounded, profile-scoped receipt persistence for the learning plane.

This module never opens ``memory.db``.  A caller must admit a profile before a
receipt transaction starts; profile deletion closes that admission and drains
these short transactions before its cross-store erasure saga continues.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from superlocalmemory.contracts.v402 import validate_agent_experience, validate_cognitive_turn

_T = TypeVar("_T")
_WRITE_DEADLINE_SECONDS = 0.90
_PROCESS_LOCKS: dict[str, threading.Lock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()
_PROFILE_GATES: dict[str, "_ProfileAdmissionGate"] = {}


class AgentExperienceConflictError(ValueError):
    """An opaque receipt identifier already names different evidence."""


class CognitiveTurnTransitionError(ValueError):
    """A cognitive turn was missing or attempted an invalid transition."""


class LearningWriteBusyError(RuntimeError):
    """Learning receipt admission could not acquire SQLite before its deadline."""


class ProfileAdmissionError(ValueError):
    """The profile is inactive or currently closing for erasure."""


class _ProfileAdmissionGate:
    """Path-scoped admission leases which make receipt erasure race-free."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._closing: set[str] = set()
        self._inflight: dict[str, int] = {}

    def admit(self, profile_id: str, is_active: Callable[[str], bool]) -> None:
        if not is_active(profile_id):
            raise ProfileAdmissionError("profile is inactive or closing for erasure")
        with self._condition:
            if profile_id in self._closing or not is_active(profile_id):
                raise ProfileAdmissionError("profile is inactive or closing for erasure")
            self._inflight[profile_id] = self._inflight.get(profile_id, 0) + 1

    def release(self, profile_id: str) -> None:
        with self._condition:
            remaining = self._inflight.get(profile_id, 0) - 1
            if remaining > 0:
                self._inflight[profile_id] = remaining
            else:
                self._inflight.pop(profile_id, None)
                self._condition.notify_all()

    def close_and_drain(self, profile_id: str, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            self._closing.add(profile_id)
            while self._inflight.get(profile_id, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LearningWriteBusyError("profile receipt drain deadline exceeded")
                self._condition.wait(remaining)


def _canonical(payload: dict[str, Any]) -> tuple[str, str]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentExperienceStore:
    """Persist durable learning receipts without entering recall's lock domain."""

    def __init__(
        self, learning_db_path: str | Path, *, is_profile_active: Callable[[str], bool]
    ) -> None:
        self._path = Path(learning_db_path).resolve()
        self._is_profile_active = is_profile_active
        with _PROCESS_LOCKS_GUARD:
            self._lock = _PROCESS_LOCKS.setdefault(str(self._path), threading.Lock())
            self._gate = _PROFILE_GATES.setdefault(str(self._path), _ProfileAdmissionGate())

    def record_experience(self, payload: dict[str, Any]) -> bool:
        validate_agent_experience(payload)
        profile_id = payload["profile_id"]
        self._admit(profile_id)
        _, digest = _canonical(payload)

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_profile_open(conn, profile_id)
            row = self._experience_row(payload, digest)
            cursor = conn.execute(
                "INSERT INTO agent_experiences ("
                "profile_id, experience_id, occurred_at, task_class, project_scope, "
                "route_json, verification_authority, verification_digest, "
                "verification_reference, producer_claim, "
                "terminal_status, failure_class, human_intervention, lessons, receipt_digest, "
                "artifact_digests_json, payload_sha256, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(profile_id, experience_id) DO NOTHING",
                row,
            )
            if cursor.rowcount:
                return True
            existing = self._get_experience_conn(
                conn, payload["profile_id"], payload["experience_id"]
            )
            if existing == payload:
                return False
            raise AgentExperienceConflictError("receipt ID already names different evidence")

        try:
            return self._write(write)
        finally:
            self._gate.release(profile_id)

    def get_experience(self, profile_id: str, experience_id: str) -> dict[str, Any] | None:
        conn = self._read_connection()
        try:
            return self._get_experience_conn(conn, profile_id, experience_id)
        finally:
            conn.close()

    def create_cognitive_turn(self, payload: dict[str, Any]) -> bool:
        validate_cognitive_turn(payload)
        if payload["state"] != "open":
            raise CognitiveTurnTransitionError("new cognitive turns must be open")
        profile_id = payload["profile_id"]
        self._admit(profile_id)
        _, digest = _canonical(payload)

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_profile_open(conn, profile_id)
            cursor = conn.execute(
                "INSERT INTO cognitive_turn_receipts ("
                "profile_id, receipt_id, task_id, project_scope, query_digest, "
                "fact_decisions_json, "
                "state, outcome_json, payload_sha256, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?, ?) "
                "ON CONFLICT(profile_id, receipt_id) DO NOTHING",
                (
                    payload["profile_id"],
                    payload["receipt_id"],
                    payload["task_id"],
                    payload["project_scope"],
                    payload["query_digest"],
                    self._json(payload["fact_decisions"]),
                    digest,
                    _now(),
                    _now(),
                ),
            )
            if cursor.rowcount:
                return True
            existing = self._get_turn_conn(conn, payload["profile_id"], payload["receipt_id"])
            if existing == payload:
                return False
            raise AgentExperienceConflictError("receipt ID already names different evidence")

        try:
            return self._write(write)
        finally:
            self._gate.release(profile_id)

    def get_cognitive_turn(self, profile_id: str, receipt_id: str) -> dict[str, Any] | None:
        conn = self._read_connection()
        try:
            return self._get_turn_conn(conn, profile_id, receipt_id)
        finally:
            conn.close()

    def finalize_cognitive_turn(
        self, profile_id: str, receipt_id: str, outcome: dict[str, Any]
    ) -> bool:
        self._admit(profile_id)

        def write(conn: sqlite3.Connection) -> bool:
            self._assert_profile_open(conn, profile_id)
            current = self._get_turn_conn(conn, profile_id, receipt_id)
            if current is None:
                raise CognitiveTurnTransitionError("cognitive turn not found for this profile")
            finalized = {**current, "state": "finalized", "outcome": outcome}
            validate_cognitive_turn(finalized)
            _, digest = _canonical(finalized)
            cursor = conn.execute(
                "UPDATE cognitive_turn_receipts SET state='finalized', outcome_json=?, "
                "payload_sha256=?, updated_at=? WHERE profile_id=? AND receipt_id=? "
                "AND state='open'",
                (self._json(outcome), digest, _now(), profile_id, receipt_id),
            )
            if cursor.rowcount:
                return True
            existing = self._get_turn_conn(conn, profile_id, receipt_id)
            if existing == finalized:
                return False
            raise AgentExperienceConflictError("cognitive turn finalized differently")

        try:
            return self._write(write)
        finally:
            self._gate.release(profile_id)

    def erase_profile(self, profile_id: str, *, close_profile: bool = True) -> int:
        """Purge all receipts, permanently closing admission for profile erasure.

        A standalone learning reset can opt out of closing because the memory
        profile remains active and should be able to collect new evidence.
        """
        if close_profile:
            self._gate.close_and_drain(profile_id)

        def erase(conn: sqlite3.Connection) -> int:
            # This durable closure is checked inside every receipt write
            # transaction. SQLite's writer serialization makes an erasure
            # followed by a stale process's write fail closed across processes.
            if close_profile:
                conn.execute(
                    "INSERT INTO agent_receipt_profile_closures (profile_id, closed_at) "
                    "VALUES (?, ?) ON CONFLICT(profile_id) "
                    "DO UPDATE SET closed_at=excluded.closed_at",
                    (profile_id, _now()),
                )
            experience_count = conn.execute(
                "DELETE FROM agent_experiences WHERE profile_id=?", (profile_id,)
            ).rowcount
            turn_count = conn.execute(
                "DELETE FROM cognitive_turn_receipts WHERE profile_id=?", (profile_id,)
            ).rowcount
            external_count = 0
            has_external = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='external_evidence_receipts'"
            ).fetchone() is not None
            if has_external:
                external_count = conn.execute(
                    "DELETE FROM external_evidence_receipts WHERE profile_id=?", (profile_id,)
                ).rowcount
            execution_count = 0
            execution_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name IN ('execution_learning_receipts', 'execution_learning_events')"
                )
            }
            if execution_tables and execution_tables != {
                "execution_learning_receipts", "execution_learning_events"
            }:
                raise sqlite3.OperationalError("incomplete execution-learning receipt schema")
            has_execution = bool(execution_tables)
            if has_execution:
                execution_count += conn.execute(
                    "DELETE FROM execution_learning_events WHERE profile_id=?", (profile_id,)
                ).rowcount
                execution_count += conn.execute(
                    "DELETE FROM execution_learning_receipts WHERE profile_id=?", (profile_id,)
                ).rowcount
            receipt_tables = ["agent_experiences", "cognitive_turn_receipts"]
            if has_external:
                receipt_tables.append("external_evidence_receipts")
            if has_execution:
                receipt_tables.extend([
                    "execution_learning_receipts", "execution_learning_events",
                ])
            residue = sum(
                int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE profile_id=?", (profile_id,)
                    ).fetchone()[0]
                )
                for table in receipt_tables
            )
            if residue:
                raise RuntimeError("learning receipt erasure left profile residue")
            return experience_count + turn_count + external_count + execution_count

        return self._write(erase)

    def _admit(self, profile_id: str) -> None:
        self._gate.admit(profile_id, self._is_profile_active)

    @staticmethod
    def _assert_profile_open(conn: sqlite3.Connection, profile_id: str) -> None:
        closed = conn.execute(
            "SELECT 1 FROM agent_receipt_profile_closures WHERE profile_id=?", (profile_id,)
        ).fetchone()
        if closed is not None:
            raise ProfileAdmissionError("profile is inactive or closing for erasure")

    def _write(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        deadline = time.monotonic() + _WRITE_DEADLINE_SECONDS
        if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise LearningWriteBusyError("learning receipt write deadline exceeded")
        try:
            while True:
                conn: sqlite3.Connection | None = None
                try:
                    conn = sqlite3.connect(str(self._path), timeout=0, isolation_level=None)
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA busy_timeout=0")
                    conn.execute("BEGIN IMMEDIATE")
                    result = operation(conn)
                    conn.execute("COMMIT")
                    return result
                except sqlite3.OperationalError as exc:
                    if conn is not None and conn.in_transaction:
                        conn.execute("ROLLBACK")
                    if not self._is_busy(exc) or time.monotonic() >= deadline:
                        if self._is_busy(exc):
                            raise LearningWriteBusyError(
                                "learning receipt write deadline exceeded"
                            ) from exc
                        raise
                    time.sleep(min(deadline - time.monotonic(), 0.01 + random.random() * 0.02))
                except Exception:
                    if conn is not None and conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                finally:
                    if conn is not None:
                        conn.close()
        finally:
            self._lock.release()

    def _read_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=0.5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=500")
        return conn

    @staticmethod
    def _is_busy(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _experience_row(self, payload: dict[str, Any], digest: str) -> tuple[Any, ...]:
        verification = payload["verification"]
        return (
            payload["profile_id"],
            payload["experience_id"],
            payload["occurred_at"],
            payload["task_class"],
            payload["project_scope"],
            self._json(payload["route"]),
            verification["authority"],
            verification["evidence_digest"],
            verification.get("reference"),
            payload["producer_claim"],
            payload["terminal_status"],
            payload.get("failure_class"),
            None if "human_intervention" not in payload else int(payload["human_intervention"]),
            payload.get("lessons"),
            payload.get("receipt_digest"),
            self._json(payload.get("artifact_digests", [])),
            digest,
            _now(),
        )

    def _get_experience_conn(
        self, conn: sqlite3.Connection, profile_id: str, experience_id: str
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM agent_experiences WHERE profile_id=? AND experience_id=?",
            (profile_id, experience_id),
        ).fetchone()
        if row is None:
            return None
        result = {
            "experience_id": row["experience_id"],
            "profile_id": row["profile_id"],
            "occurred_at": row["occurred_at"],
            "task_class": row["task_class"],
            "project_scope": row["project_scope"],
            "route": json.loads(row["route_json"]),
            "verification": {
                "authority": row["verification_authority"],
                "evidence_digest": row["verification_digest"],
            },
            "producer_claim": row["producer_claim"],
            "terminal_status": row["terminal_status"],
        }
        if row["verification_reference"] is not None:
            result["verification"]["reference"] = row["verification_reference"]
        for key in ("failure_class", "lessons", "receipt_digest"):
            if row[key] is not None:
                result[key] = row[key]
        if row["human_intervention"] is not None:
            result["human_intervention"] = bool(row["human_intervention"])
        artifacts = json.loads(row["artifact_digests_json"])
        if artifacts:
            result["artifact_digests"] = artifacts
        return result

    def _get_turn_conn(
        self, conn: sqlite3.Connection, profile_id: str, receipt_id: str
    ) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM cognitive_turn_receipts WHERE profile_id=? AND receipt_id=?",
            (profile_id, receipt_id),
        ).fetchone()
        if row is None:
            return None
        result = {
            "receipt_id": row["receipt_id"],
            "task_id": row["task_id"],
            "profile_id": row["profile_id"],
            "project_scope": row["project_scope"],
            "query_digest": row["query_digest"],
            "fact_decisions": json.loads(row["fact_decisions_json"]),
            "state": row["state"],
        }
        if row["outcome_json"] is not None:
            result["outcome"] = json.loads(row["outcome_json"])
        return result


def purge_profile_receipts(
    learning_db_path: str | Path, profile_id: str, *, close_profile: bool = True
) -> int:
    """Purge M040 evidence for a profile before its memory profile is deleted.

    A database from an older release simply has no receipt tables and is
    already clean.  A half-present schema is corruption and must stop profile
    deletion rather than silently strand unerasable evidence.
    """
    path = Path(learning_db_path)
    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('agent_experiences', 'cognitive_turn_receipts', "
                "'agent_receipt_profile_closures', 'external_evidence_receipts', "
                "'execution_learning_receipts', 'execution_learning_events')"
            )
        }
    if not tables:
        return 0
    expected = {
        "agent_experiences", "cognitive_turn_receipts", "agent_receipt_profile_closures"
    }
    if tables != expected:
        optional = {"external_evidence_receipts"}
        execution = {"execution_learning_receipts", "execution_learning_events"}
        if tables in (expected | optional, expected | execution, expected | optional | execution):
            from superlocalmemory.storage.migrations import M041_external_evidence_receipts as m041

            with sqlite3.connect(path) as conn:
                # Erasure needs a valid table, not its optional performance indexes.
                # A damaged index must never strand profile-scoped evidence.
                external_ok = (
                    "external_evidence_receipts" not in tables or m041._table_is_valid(conn)
                )
                execution_ok = (
                    not execution.intersection(tables)
                    or execution <= tables
                )
                if external_ok and execution_ok:
                    return AgentExperienceStore(
                        path, is_profile_active=lambda _: True
                    ).erase_profile(profile_id, close_profile=close_profile)
        raise sqlite3.OperationalError("incomplete Agent Experience receipt schema")
    return AgentExperienceStore(
        path, is_profile_active=lambda _: True
    ).erase_profile(profile_id, close_profile=close_profile)


def get_profile_receipt_summary(
    learning_db_path: str | Path, profile_id: str
) -> dict[str, Any]:
    """Return the small, read-only receipt view used by every host surface.

    This intentionally performs only indexed aggregates against ``learning.db``.
    It is safe to call from MCP, CLI, HTTP, and the dashboard without opening a
    memory engine or entering the recall/remember writer domains.
    """
    unavailable: dict[str, Any] = {
        "is_real": False,
        "availability": "unavailable",
        "experiences_total": 0,
        "turns_total": 0,
        "turns_by_state": {},
        "claimed_evidence_experiences": 0,
        "source": "learning.db:agent_experiences,cognitive_turn_receipts",
    }
    path = Path(learning_db_path)
    if not path.exists():
        return unavailable
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.5)
        try:
            experience = conn.execute(
                "SELECT COUNT(*) FROM agent_experiences WHERE profile_id=?", (profile_id,)
            ).fetchone()
            claimed = conn.execute(
                "SELECT COUNT(*) FROM agent_experiences "
                "WHERE profile_id=? AND verification_authority != 'bounded_loop_receipt'",
                (profile_id,),
            ).fetchone()
            rows = conn.execute(
                "SELECT state, COUNT(*) FROM cognitive_turn_receipts "
                "WHERE profile_id=? GROUP BY state",
                (profile_id,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return unavailable
    turns_by_state = {str(state): int(count) for state, count in rows}
    return {
        **unavailable,
        "is_real": True,
        "availability": "available",
        "experiences_total": int(experience[0]) if experience else 0,
        "claimed_evidence_experiences": int(claimed[0]) if claimed else 0,
        "turns_total": sum(turns_by_state.values()),
        "turns_by_state": turns_by_state,
    }
