"""Private enrollment metadata, separate from the canonical memory database.

An acknowledgment is always pending, never proof of an authorized live client.
Only a configured local enrollment service constructs this journal. Local-only
startup must not import a companion runtime or create remote enrollment state.
"""

from __future__ import annotations

import builtins
import json
import os
import re
import sqlite3
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
from uuid import uuid4

HOSTS = frozenset({"muse", "chatgpt", "claude_web", "claude_code_web", "composio", "other_mcp"})
_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")


class JournalConflict(RuntimeError):
    """Bounded public error code; no SQL, credentials or payload details."""


@dataclass(frozen=True)
class Enrollment:
    installation_id: str
    connection_id: str
    host: str
    state: str
    version: int
    requested: bool
    cleanup_pending: bool
    intent_json: str
    authorization_url: str | None = None
    intent_key: str = ""

    @property
    def intent(self) -> dict:
        return json.loads(self.intent_json)


@dataclass(frozen=True)
class DispatchLease:
    token: str
    version: int


def _identity(value: str) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise ValueError("invalid_identity")
    return value


def _intent(profile: str, payload: dict) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "host",
        "profile_id",
        "remote_opt_in",
        "permissions",
    }:
        raise ValueError("invalid_consent")
    permissions = payload["permissions"]
    if (
        not isinstance(payload["host"], str)
        or payload["host"] not in HOSTS
        or payload["profile_id"] != profile
        or payload["remote_opt_in"] is not True
        or not isinstance(permissions, dict)
        or set(permissions) != {"read", "write", "correction", "session"}
        or any(type(value) is not bool for value in permissions.values())
        or not permissions["read"]
        or (permissions["correction"] and not permissions["write"])
    ):
        raise ValueError("invalid_consent")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class EnrollmentJournal:
    """Atomic intent creation, dispatch leases and cancellation/version fences."""

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time):
        root = Path(root)
        if root.is_symlink():
            raise ValueError("unsafe_journal_path")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.path = root / "enrollment.sqlite3"
        self._clock = clock
        if self.path.is_symlink():
            raise ValueError("unsafe_journal_path")
        from superlocalmemory.remote_connections.private_state import protect_directory

        protect_directory(root)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or (
                os.name == "posix" and info.st_uid != os.getuid()
            ):
                raise ValueError("unsafe_journal_path")
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        if os.name != "posix":
            from superlocalmemory.infra.owner_only_acl import restrict_to_owner

            restrict_to_owner(self.path)
            for suffix in ("-journal", "-wal", "-shm"):
                sidecar = self.path.with_name(self.path.name + suffix)
                if sidecar.is_symlink():
                    raise ValueError("unsafe_journal_path")
                if sidecar.exists():
                    restrict_to_owner(sidecar)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS enrollments (
                owner TEXT NOT NULL, profile TEXT NOT NULL, intent_key TEXT NOT NULL,
                connection_id TEXT NOT NULL UNIQUE, intent TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', version INTEGER NOT NULL DEFAULT 1,
                requested INTEGER NOT NULL DEFAULT 0, lease_token TEXT,
                lease_until REAL NOT NULL DEFAULT 0, remote_reference TEXT,
                cleanup_pending INTEGER NOT NULL DEFAULT 0,
                authorization_url TEXT,
                PRIMARY KEY(owner, profile, intent_key))""")
            if "authorization_url" not in {
                row[1] for row in db.execute("PRAGMA table_info(enrollments)")
            }:
                db.execute("ALTER TABLE enrollments ADD COLUMN authorization_url TEXT")
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('installation_id', ?)", (uuid4().hex,)
            )
            self.installation_id = db.execute(
                "SELECT value FROM metadata WHERE key='installation_id'"
            ).fetchone()[0]
            _identity(self.installation_id)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _record(self, row: sqlite3.Row) -> Enrollment:
        return Enrollment(
            self.installation_id,
            row["connection_id"],
            json.loads(row["intent"])["host"],
            row["state"],
            row["version"],
            bool(row["requested"]),
            bool(row["cleanup_pending"]),
            row["intent"],
            row["authorization_url"],
            row["intent_key"],
        )

    @staticmethod
    def _owned(db: sqlite3.Connection, owner: str, profile: str, connection_id: str) -> sqlite3.Row:
        _identity(owner)
        _identity(profile)
        _identity(connection_id)
        row = db.execute(
            "SELECT * FROM enrollments WHERE owner=? AND profile=? AND connection_id=?",
            (owner, profile, connection_id),
        ).fetchone()
        if row is None:
            raise JournalConflict("not_found")
        return row

    def begin(self, owner: str, profile: str, key: str, payload: dict) -> Enrollment:
        _identity(owner)
        _identity(profile)
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key):
            raise ValueError("invalid_intent_key")
        immutable = _intent(profile, payload)
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM enrollments WHERE owner=? AND profile=? AND intent_key=?",
                (owner, profile, key),
            ).fetchone()
            if row is not None:
                if row["intent"] != immutable:
                    raise JournalConflict("intent_conflict")
                return self._record(row)
            if db.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0] >= 2000:
                raise JournalConflict("capacity_exhausted")
            if (
                db.execute(
                    "SELECT COUNT(*) FROM enrollments WHERE owner=? AND state='pending'", (owner,)
                ).fetchone()[0]
                >= 32
            ):
                raise JournalConflict("capacity_exhausted")
            identifier = uuid4().hex
            db.execute(
                "INSERT INTO enrollments(owner,profile,intent_key,connection_id,intent) "
                "VALUES(?,?,?,?,?)",
                (owner, profile, key, identifier, immutable),
            )
            return self._record(self._owned(db, owner, profile, identifier))

    def get(self, owner: str, profile: str, identifier: str) -> Enrollment:
        with self._transaction() as db:
            return self._record(self._owned(db, owner, profile, identifier))

    def list(self, owner: str, profile: str) -> list[Enrollment]:
        _identity(owner)
        _identity(profile)
        with self._transaction() as db:
            return [
                self._record(row)
                for row in db.execute(
                    "SELECT * FROM enrollments WHERE owner=? AND profile=? ORDER BY rowid",
                    (owner, profile),
                )
            ]

    def pending_owners(self, profile: str) -> builtins.list[str]:
        """Only explicit, non-terminal local opt-ins can trigger recovery."""
        _identity(profile)
        with self._transaction() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT owner FROM enrollments "
                    "WHERE profile=? AND state='pending' ORDER BY owner LIMIT 128",
                    (profile,),
                )
            ]

    def claim(self, owner: str, profile: str, identifier: str) -> DispatchLease | None:
        with self._transaction() as db:
            row = self._owned(db, owner, profile, identifier)
            if row["state"] != "pending" or row["requested"] or row["lease_until"] > self._clock():
                return None
            token, version = uuid4().hex, row["version"] + 1
            db.execute(
                "UPDATE enrollments SET lease_token=?,lease_until=?,version=? "
                "WHERE connection_id=?",
                (token, self._clock() + 30, version, identifier),
            )
            return DispatchLease(token, version)

    def acknowledge(
        self,
        owner: str,
        profile: str,
        identifier: str,
        token: str,
        version: int,
        remote_reference: str,
        authorization_url: str | None = None,
    ) -> bool:
        _identity(remote_reference)
        if authorization_url is not None:
            from superlocalmemory.remote_connections.service import validate_sign_in

            validate_sign_in(authorization_url, identifier)
        with self._transaction() as db:
            row = self._owned(db, owner, profile, identifier)
            if (
                row["state"] != "pending"
                or row["version"] != version
                or row["lease_token"] != token
                or row["lease_until"] <= self._clock()
            ):
                return False
            db.execute(
                """UPDATE enrollments SET requested=1, remote_reference=?,
                authorization_url=?,version=version+1,
                lease_token=NULL, lease_until=0 WHERE connection_id=?""",
                (remote_reference, authorization_url, identifier),
            )
            return True

    def cancel(
        self, owner: str, profile: str, identifier: str, expected_version: int
    ) -> Enrollment:
        with self._transaction() as db:
            row = self._owned(db, owner, profile, identifier)
            if row["state"] == "cancelled":
                return self._record(row)
            if type(expected_version) is not int or row["version"] != expected_version:
                raise JournalConflict("version_conflict")
            cleanup = bool(row["lease_token"] or row["requested"])
            db.execute(
                """UPDATE enrollments SET state='cancelled',version=version+1,
                cleanup_pending=?,lease_token=NULL,lease_until=0 WHERE connection_id=?""",
                (cleanup, identifier),
            )
            return self._record(self._owned(db, owner, profile, identifier))

    def clear_cleanup(self, owner: str, profile: str, identifier: str) -> None:
        """Called only after the authenticated cloud revocation acknowledgement."""
        with self._transaction() as db:
            row = self._owned(db, owner, profile, identifier)
            if row["state"] != "cancelled":
                raise JournalConflict("connection_not_cancelled")
            if row["cleanup_pending"]:
                db.execute(
                    "UPDATE enrollments SET cleanup_pending=0,version=version+1 "
                    "WHERE connection_id=?",
                    (identifier,),
                )
