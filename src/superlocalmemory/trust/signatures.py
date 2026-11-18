# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Signed facts (WS-5, D-16): an HMAC per fact that a copy of the file cannot forge.

Signature = HMAC-SHA256 (``trust/signing_key``) over a canonical serialization of
``(profile id, fact id, sha256(content), created_at)``, kept in ``fact_signatures`` (M082).
The threat, what it detects and what it does not are in ``signing_key.py``; the short form:
it detects a fact changed, added or moved by something that has the file but not the user's
login session. It does not stop a process running as the user.

*When it signs.* At admission, when the daemon holds the key and ``trust.signing`` observes
(:func:`admit`, called beside ``fact_trust.admit``); and in bounded background batches for
facts above the **signing mark** (:func:`maintenance_step`), so a fact saved by a hook or a
CLI process, or before the flag went on, is signed by the daemon later. A signature is
written once and never changed by a pass: a changed fact stays "modified" until a caller
decides otherwise.

*The signing mark (A-12).* ``signing_progress`` holds one row, MACed under the same key: the
highest fact row number the daemon has signed up to, written when signing first ran and
advanced by each pass. A pass signs only unsigned facts *above* the mark. An unsigned fact
*at or below* it had a signature once (or was meant to), so it reads "signature missing", is
counted as a violation by ``slm doctor --truth`` and is never signed by a pass: only a
recorded ``slm signing resign`` (any caller, D-49) signs it again. Someone with the file can change a text and
delete its signature row, but cannot make a mark that checks, so they cannot make the pass
treat the memory as new. A missing, forged or other-key mark makes the pass sign nothing.
Admission still signs a memory saved now: it is new by construction.

*What a profile fold does.* The fold moves the fact (``profile_id`` changes, nothing else),
so a signature over the old profile no longer matches the new one. The row keeps
``signed_profile_id`` (the profile the signature covers). The daemon's pass verifies the
old signature over ``signed_profile_id`` and, only if it holds, signs again for the new
profile (**re-signed in the daemon's pass, not in the fold**: the fold runs without the
key). A bad old signature stays bad. Until then the fact is counted ``moved``, not modified.

*What it does not cover.* Signing a store for the first time seals what is in it: a change
made before that is sealed in. The mark is a row number, so someone who also replaces a
memory by a new row above the mark (new text, new number, no signature) is treated as a memory
saved later; and a copy of an older, valid mark put back lowers the protection to that point.
Only the text, the creation time, the id and the profile are signed: ``pinned``, the kind and
the taint class are not (docs/signed-memories.md says why and what that leaves open).

Reads never write. Counts and ids only; no memory text is read into a report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from superlocalmemory.trust import signing_key as sk

logger = logging.getLogger(__name__)

DOMAIN = "slm-fact-signature-v1"
BATCH = 200
#: How many facts one ``slm doctor --truth`` run checks by default. Routing-only: it bounds
#: the work of a doctor run (500 HMACs take milliseconds) and is not a statistical claim.
#: A sample finds one modified fact among N with probability 500/N; ``--signatures-all``
#: checks every fact.
SAMPLE = 500
SAMPLE_IDS = 5

OK = "ok"
MODIFIED = "modified"
OTHER_KEY = "other_key"
MOVED = "moved"
UNSIGNED = "unsigned"
#: An unsigned fact at or below the signing mark: its signature was removed.
MISSING = "missing"

PROGRESS_DOMAIN = "slm-signing-progress-v1"
#: The state of the signing mark itself.
RECORD_NONE = "none"  # never signed anything: nothing to protect yet
RECORD_OK = "ok"
RECORD_MISSING = "missing"  # signatures exist but the mark was removed
RECORD_FORGED = "forged"  # the mark does not verify under this key
RECORD_OTHER_KEY = "other_key"  # made under a key this process does not hold
RECORD_UNCHECKED = "unchecked"  # no key here: not looked at


def canonical(profile_id: str, fact_id: str, content: str, created_at: str) -> bytes:
    """The bytes that are signed. Field boundaries cannot move: it is one JSON object."""
    record = {"domain": DOMAIN, "profile": profile_id, "fact": fact_id,
              "content_sha256": hashlib.sha256((content or "").encode("utf-8")).hexdigest(),
              "created_at": created_at or ""}
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


_FACT = "SELECT profile_id, content, created_at, rowid FROM atomic_facts WHERE fact_id = ?"
_INSERT = ("INSERT OR IGNORE INTO fact_signatures (fact_id, profile_id, signed_profile_id, "
           "sig, key_id, signed_at) VALUES (?, ?, ?, ?, ?, ?)")


def _sign_one(run: Callable[[str, tuple], Any], signer: sk.Signer, fact_id: str,
              row: tuple) -> bool:
    profile_id, content, created_at = row
    sig = signer.mac(canonical(profile_id, fact_id, content, created_at))
    run(_INSERT, (fact_id, profile_id, profile_id, sig, signer.key_id, _now()))
    return True


# -- the signing mark -----------------------------------------------------------------------------

_MARK_GET = ("SELECT key_id, epoch_at, signed_through, mac FROM signing_progress WHERE singleton = 1")
_MARK_CREATE = ("INSERT OR IGNORE INTO signing_progress (singleton, key_id, epoch_at, "
                "signed_through, updated_at, mac) VALUES (1, ?, ?, ?, ?, ?)")
_MARK_ADVANCE = ("UPDATE signing_progress SET key_id = ?, epoch_at = ?, signed_through = ?, "
                 "updated_at = ?, mac = ? WHERE singleton = 1")
_MARK_REPLACE = ("INSERT INTO signing_progress (singleton, key_id, epoch_at, signed_through, "
                 "updated_at, mac) VALUES (1, ?, ?, ?, ?, ?) ON CONFLICT(singleton) DO UPDATE SET "
                 "key_id = excluded.key_id, epoch_at = excluded.epoch_at, "
                 "signed_through = excluded.signed_through, updated_at = excluded.updated_at, "
                 "mac = excluded.mac")


@dataclass(frozen=True)
class Mark:
    """The signing mark as this process can read it: ``status`` is a ``RECORD_*``, ``through``
    the highest fact row number signed (0 unless the mark verifies), ``epoch_at`` when signing
    first ran for this store."""

    status: str
    through: int = 0
    epoch_at: str = ""


def _runner(conn: sqlite3.Connection) -> Callable[[str, tuple], Any]:
    return lambda sql, params=(): conn.execute(sql, params).fetchall()


def _mark_bytes(key_id: str, epoch_at: str, through: int) -> bytes:
    return json.dumps({"domain": PROGRESS_DOMAIN, "key": key_id, "epoch": epoch_at,
                       "through": int(through)}, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def progress_state(run: Callable[[str, tuple], Any], signer: sk.Signer) -> Mark:
    """What the signing mark says, checked under ``signer``. Reads only."""
    try:
        rows = run(_MARK_GET, ())
    except Exception:  # noqa: BLE001 - a store before the table: no mark
        rows = []
    if not rows:
        try:
            signed_any = bool(run("SELECT 1 FROM fact_signatures LIMIT 1", ()))
        except Exception:  # noqa: BLE001
            signed_any = False
        return Mark(RECORD_MISSING if signed_any else RECORD_NONE)
    key_id, epoch_at, through, mac = rows[0][0], rows[0][1], int(rows[0][2]), rows[0][3]
    if key_id != signer.key_id:
        return Mark(RECORD_OTHER_KEY, epoch_at=epoch_at)
    if not signer.matches(_mark_bytes(key_id, epoch_at, through), mac):
        return Mark(RECORD_FORGED, epoch_at=epoch_at)
    return Mark(RECORD_OK, through, epoch_at)


def _mark_params(signer: sk.Signer, through: int, epoch_at: str) -> tuple:
    return (signer.key_id, epoch_at, int(through), _now(),
            signer.mac(_mark_bytes(signer.key_id, epoch_at, through)))


def create_progress(run: Callable[[str, tuple], Any], signer: sk.Signer, through: int) -> None:
    """The first mark of a store (signing first ran). Never replaces an existing one."""
    run(_MARK_CREATE, _mark_params(signer, through, _now()))


def advance_progress(run: Callable[[str, tuple], Any], signer: sk.Signer, through: int,
                     epoch_at: str) -> None:
    """Move the mark (the daemon's pass, under the writer lock)."""
    run(_MARK_ADVANCE, _mark_params(signer, through, epoch_at or _now()))


def _first_signature(db: Any, signer: sk.Signer) -> None:
    """At the first admission-time signature of a store, record that signing began, so a
    later pass can tell a memory that was signed from one that is new. A mark that already
    exists, or signatures without a mark, are left as they are."""
    if getattr(db, "_signing_mark_key", None) == signer.key_id:
        return
    from superlocalmemory.storage.migrations import M082_fact_signatures as m082

    def run(sql: str, params: tuple = ()) -> Any:
        return db.execute(sql, params)

    if not run("SELECT 1 FROM sqlite_master WHERE type='table' AND name='signing_progress'", ()):
        for statement in m082.PROGRESS_DDL.split(";"):
            if statement.strip():
                run(statement, ())
    status = progress_state(run, signer).status
    if status == RECORD_NONE:
        create_progress(run, signer, 0)
    if status in (RECORD_NONE, RECORD_OK):
        db._signing_mark_key = signer.key_id


def _observing() -> bool:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.infra.data_root import canonical_data_root

    flags = FeatureFlags(canonical_data_root())
    return (not flags.killed) and flags.observing("trust.signing")


def admit(db: Any, fact: Any) -> None:
    """At admission: sign the fact just written, if this process holds the key. Never raises.

    A process without the key (a hook, a CLI save) does nothing here; the daemon's pass
    signs the fact later. Memory-only check first, so a save costs what it did before when
    signing is off.
    """
    try:
        signer = sk.current()
        if signer is None or not _observing():
            return
        if not getattr(db, "_fact_signatures_ready", False):
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                              "AND name='fact_signatures'", ()):
                return  # the deferred M082 has not run yet
            db._fact_signatures_ready = True
        _first_signature(db, signer)
        rows = db.execute(_FACT, (fact.fact_id,))
        if rows:
            r = rows[0]
            _sign_one(lambda sql, params: db.execute(sql, params), signer, fact.fact_id,
                      (r[0], r[1], r[2]))
            _advance_at_admission(db, signer, int(r[3]))
    except Exception as exc:  # noqa: BLE001 - a signature is evidence, never a lost save
        logger.warning("fact not signed %s (%s)", getattr(fact, "fact_id", "?"),
                       type(exc).__name__)


_GAP = ("SELECT 1 FROM atomic_facts f LEFT JOIN fact_signatures s ON s.fact_id = f.fact_id "
        "WHERE s.fact_id IS NULL AND f.rowid > ? AND f.rowid < ? LIMIT 1")


def _advance_at_admission(db: Any, signer: sk.Signer, rowid: int) -> None:
    """A fact signed at admission moves the mark up to itself when every fact below it is
    signed, so a memory saved through the daemon is covered at once, not at the next pass.
    Row numbers only grow, so nothing can appear below ``rowid`` later."""
    def run(sql: str, params: tuple = ()) -> Any:
        return db.execute(sql, params)

    mark = progress_state(run, signer)
    if mark.status == RECORD_OK and rowid > mark.through and not run(_GAP, (mark.through, rowid)):
        advance_progress(run, signer, rowid, mark.epoch_at)


# -- background batches ---------------------------------------------------------------------------

_UNSIGNED_ABOVE = (
    "SELECT f.rowid, f.fact_id, f.profile_id, f.content, f.created_at FROM atomic_facts f "
    "LEFT JOIN fact_signatures s ON s.fact_id = f.fact_id "
    "WHERE s.fact_id IS NULL AND f.rowid > ? ORDER BY f.rowid LIMIT ?")
_UNSIGNED_AT_OR_BELOW = (
    "SELECT f.fact_id FROM atomic_facts f LEFT JOIN fact_signatures s ON s.fact_id = f.fact_id "
    "WHERE s.fact_id IS NULL AND f.rowid <= ? ORDER BY f.rowid LIMIT ?")
_COUNT_AT_OR_BELOW = (
    "SELECT COUNT(*) FROM atomic_facts f LEFT JOIN fact_signatures s ON s.fact_id = f.fact_id "
    "WHERE s.fact_id IS NULL AND f.rowid <= ?")
_HEAD = "SELECT COALESCE(MAX(rowid), 0) FROM atomic_facts"
_MOVED = ("SELECT s.fact_id, s.signed_profile_id, s.sig, s.key_id, f.profile_id, f.content, "
          "f.created_at FROM fact_signatures s JOIN atomic_facts f ON f.fact_id = s.fact_id "
          "WHERE s.signed_profile_id != f.profile_id AND s.profile_id = f.profile_id "
          "ORDER BY f.rowid LIMIT ?")


def sign_missing(conn: sqlite3.Connection, signer: sk.Signer, *, limit: int = BATCH) -> int:
    """Sign up to ``limit`` unsigned facts above the signing mark and advance it. Returns how
    many were signed. Signs nothing unless the mark is this key's, or there is none because
    nothing was ever signed (the first run creates it). Facts at or below the mark are never
    signed here (A-12)."""
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")  # the mark and the facts it covers move together
    run = _runner(conn)
    mark = progress_state(run, signer)
    if mark.status == RECORD_NONE:
        create_progress(run, signer, 0)
        through = 0
    elif mark.status == RECORD_OK:
        through = mark.through
    else:
        return 0
    head = int(run(_HEAD, ())[0][0])
    through = min(through, head)  # erased newest facts hand their row numbers to later saves
    rows = conn.execute(_UNSIGNED_ABOVE, (through, int(limit))).fetchall()
    for _rowid, fact_id, profile_id, content, created_at in rows:
        _sign_one(conn.execute, signer, fact_id, (profile_id, content, created_at))
    reached = head if len(rows) < int(limit) else int(rows[-1][0])
    advance_progress(run, signer, max(through, reached), mark.epoch_at)
    return len(rows)


def reseal_moved(conn: sqlite3.Connection, signer: sk.Signer, *, limit: int = BATCH) -> int:
    """Sign again for the new profile the facts a fold moved, if the old signature holds.

    A signature that does not verify over its ``signed_profile_id`` is left alone (it is
    reported as modified); one made under another key is left alone too.
    """
    done = 0
    for (fact_id, signed_profile, sig, key_id, profile_id, content,
         created_at) in conn.execute(_MOVED, (int(limit),)).fetchall():
        if key_id != signer.key_id or not signer.matches(
                canonical(signed_profile, fact_id, content, created_at), sig):
            continue
        fresh = signer.mac(canonical(profile_id, fact_id, content, created_at))
        conn.execute("UPDATE fact_signatures SET signed_profile_id = ?, sig = ?, signed_at = ? "
                     "WHERE fact_id = ? AND sig = ?",
                     (profile_id, fresh, _now(), fact_id, sig))
        done += 1
    return done


def _ensure_mark_table(conn: sqlite3.Connection) -> None:
    """A store whose M082 ran before the signing mark existed gets the table now."""
    from superlocalmemory.storage.migrations import M082_fact_signatures as m082

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='signing_progress'").fetchone():
        for statement in m082.PROGRESS_DDL.split(";"):
            if statement.strip():
                conn.execute(statement)


def maintenance_step(db: Any, *, max_batches: int = 5) -> int:
    """One maintenance cycle's share, only while the daemon holds the key and the flag observes."""
    signer = sk.current()
    if signer is None or not _observing():
        return 0
    total = 0
    for _ in range(max_batches):
        # One short write per batch (WS-7): the writer lock is let go between batches, so a
        # save waits for at most one batch of 200 HMACs, never for the whole run.
        try:
            with db.raw_connection() as conn:
                _ensure_mark_table(conn)
                moved = reseal_moved(conn, signer, limit=BATCH)
                fresh = sign_missing(conn, signer, limit=BATCH)
        except sqlite3.OperationalError:  # a store before M082
            return 0
        total += moved + fresh
        if moved < BATCH and fresh < BATCH:
            break
    return total


# -- reads (never write) --------------------------------------------------------------------------

@dataclass(frozen=True)
class Report:
    """What a check found. Counts and ids only."""

    key_status: str
    facts: int = 0
    signed: int = 0
    unsigned: int = 0
    checked: int = 0
    modified: int = 0
    other_key: int = 0
    moved: int = 0
    modified_ids: tuple[str, ...] = field(default_factory=tuple)
    exhaustive: bool = False
    #: Unsigned facts at or below the signing mark: their signature was removed (A-12).
    missing: int = 0
    missing_ids: tuple[str, ...] = field(default_factory=tuple)
    #: The signing mark's own state (a ``RECORD_*``).
    record: str = RECORD_UNCHECKED

    @property
    def ok(self) -> bool:
        """Nothing found modified, no signature removed, and the signing mark is not missing
        or forged. Unsigned (new) and other-key facts are reported, not failures."""
        return (self.modified == 0 and self.missing == 0
                and self.record not in (RECORD_MISSING, RECORD_FORGED))

    def as_dict(self) -> dict[str, Any]:
        return {"key_status": self.key_status, "key_words": sk.words(self.key_status),
                "facts": self.facts, "signed": self.signed, "unsigned": self.unsigned,
                "checked": self.checked, "modified": self.modified,
                "signed_under_another_key": self.other_key, "moved_by_a_fold": self.moved,
                "modified_ids": list(self.modified_ids), "exhaustive": self.exhaustive,
                "signature_missing": self.missing, "missing_ids": list(self.missing_ids),
                "record": self.record, "record_words": record_words(self.record)}


_RECORD_WORDS = {
    RECORD_NONE: "Nothing has been signed yet.",
    RECORD_OK: "The signing record checks.",
    RECORD_MISSING: ("The signing record was removed while signed memories remain, so a removed "
                     "signature cannot be told from a new memory. Run `slm signing resign`."),
    RECORD_FORGED: ("The signing record does not check under your key: it was changed outside "
                    "SuperLocalMemory. Run `slm signing resign` once you have looked at why."),
    RECORD_OTHER_KEY: ("The signing record was made under a key this computer does not hold. "
                       "Run `slm signing resign` to move it to the current key."),
    RECORD_UNCHECKED: "The signing record was not checked (no key in this process).",
}


def record_words(status: str) -> str:
    return _RECORD_WORDS.get(status, status)


def state_of(signer: sk.Signer, row: tuple) -> str:
    """``row`` = (fact_id, current profile, content, created_at, sig, key_id, signed_profile,
    signature profile). One of OK, MODIFIED, OTHER_KEY, MOVED."""
    fact_id, profile_id, content, created_at, sig, key_id, signed_profile, sig_profile = row
    if key_id != signer.key_id:
        return OTHER_KEY
    if signer.matches(canonical(signed_profile, fact_id, content, created_at), sig):
        if signed_profile == profile_id:
            return OK
        return MOVED if sig_profile == profile_id else MODIFIED
    return MODIFIED


_JOIN = ("SELECT f.fact_id, f.profile_id, f.content, f.created_at, s.sig, s.key_id, "
         "s.signed_profile_id, s.profile_id FROM fact_signatures s "
         "JOIN atomic_facts f ON f.fact_id = s.fact_id")


def counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """``(facts, signed)`` by the database; a store before M082 has none signed."""
    facts = int(conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0])
    try:
        signed = int(conn.execute(
            "SELECT COUNT(*) FROM fact_signatures s JOIN atomic_facts f "
            "ON f.fact_id = s.fact_id").fetchone()[0])
    except sqlite3.OperationalError:
        signed = 0
    return facts, signed


def verify(conn: sqlite3.Connection, signer: sk.Signer | None, *, key_status: str = sk.READY,
           sample: int | None = SAMPLE) -> Report:
    """Count signed and unsigned facts and verify ``sample`` of the signed ones (all when ``None``).

    Without a key the counts are still given and nothing is verified (``checked`` is 0).
    The sample is random (``ORDER BY RANDOM()``): routing-only, see :data:`SAMPLE`.
    """
    facts, signed = counts(conn)
    unsigned = facts - signed
    if signer is None:
        return Report(key_status, facts, signed, unsigned)
    mark = progress_state(_runner(conn), signer)
    gone, gone_ids = _removed(conn, mark)
    if signed == 0:
        return Report(key_status, facts, signed, unsigned, missing=gone, missing_ids=gone_ids,
                      record=mark.status)
    sql = _JOIN + (" ORDER BY RANDOM() LIMIT ?" if sample is not None else "")
    rows = conn.execute(sql, (int(sample),) if sample is not None else ()).fetchall()
    bad: list[str] = []
    other = moved = modified = 0
    for row in rows:
        state = state_of(signer, row)
        if state == MODIFIED:
            modified += 1
            if len(bad) < SAMPLE_IDS:
                bad.append(row[0])
        elif state == OTHER_KEY:
            other += 1
        elif state == MOVED:
            moved += 1
    return Report(key_status, facts, signed, unsigned, len(rows), modified, other, moved,
                  tuple(bad), exhaustive=sample is None or len(rows) >= signed,
                  missing=gone, missing_ids=gone_ids, record=mark.status)


def _removed(conn: sqlite3.Connection, mark: Mark) -> tuple[int, tuple[str, ...]]:
    """Unsigned facts at or below a mark that checks: ``(count, first few ids)``."""
    if mark.status != RECORD_OK:
        return 0, ()
    try:
        n = int(conn.execute(_COUNT_AT_OR_BELOW, (mark.through,)).fetchone()[0])
        ids = tuple(r[0] for r in conn.execute(_UNSIGNED_AT_OR_BELOW,
                                               (mark.through, SAMPLE_IDS)).fetchall()) if n else ()
    except sqlite3.OperationalError:
        return 0, ()
    return n, ids


def for_fact(conn: sqlite3.Connection, fact_id: str, signer: sk.Signer | None) -> dict[str, Any]:
    """One fact's signature state for ``slm why``. Never reads the keychain: ``signer`` is
    whatever the caller already holds (the daemon's key, or ``None``)."""
    try:
        sig = conn.execute("SELECT key_id, signed_at FROM fact_signatures WHERE fact_id = ?",
                           (fact_id,)).fetchone()
    except sqlite3.OperationalError:
        return {"state": "not_available", "words": "This store does not keep signatures yet."}
    if sig is None:
        return _unsigned(conn, fact_id, signer)
    base = {"key_id": sig[0], "signed_at": sig[1]}
    if signer is None:
        return {**base, "state": "signed_unchecked",
                "words": "Signed. This process does not hold the key, so it was not checked."}
    row = conn.execute(_JOIN + " WHERE s.fact_id = ?", (fact_id,)).fetchone()
    if row is None:
        return {**base, "state": UNSIGNED, "words": "Not signed (yet)."}
    state = state_of(signer, row)
    text = {OK: "Signed, and unchanged since it was signed.",
            MODIFIED: "Signed, but the memory no longer matches its signature.",
            OTHER_KEY: "Signed under another key, so it cannot be checked with this one.",
            MOVED: "Signed; a profile move is waiting for the daemon to sign it again."}[state]
    return {**base, "state": state, "words": text}


def _unsigned(conn: sqlite3.Connection, fact_id: str, signer: sk.Signer | None) -> dict[str, Any]:
    """A fact with no signature row: new (waiting for the daemon) or removed (A-12)."""
    if signer is not None:
        mark = progress_state(_runner(conn), signer)
        row = conn.execute("SELECT rowid FROM atomic_facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if mark.status == RECORD_OK and row is not None and int(row[0]) <= mark.through:
            return {"state": MISSING, "words": (
                "Its signature was removed: this memory was signed once and no longer is. "
                "It may have been changed. Run `slm signing resign` once you have checked.")}
    return {"state": UNSIGNED, "words": "Not signed (yet)."}


def resign(conn: sqlite3.Connection, signer: sk.Signer, *, only_other_key: bool = True,
           limit: int | None = None, include_missing: bool = False) -> int:
    """Sign again, with the current key, facts whose signature cannot be checked now.

    A recorded decision (``slm signing resign``, any caller, D-49): it blesses what is in the
    store as it stands, so it is never done by a pass. ``only_other_key`` limits it to
    signatures made under a key this computer no longer holds (a lost or replaced key).
    ``include_missing`` also signs every memory that has no signature (the ones a removed
    signature left, and the ones not yet signed) and writes the signing mark afresh under
    this key, covering everything now signed (A-12). Returns how many memories were signed.
    """
    done = _resign_signed(conn, signer, only_other_key, limit)
    if include_missing:
        run = _runner(conn)
        epoch = progress_state(run, signer).epoch_at or _now()
        for _rowid, fact_id, profile_id, content, created_at in conn.execute(
                _UNSIGNED_ABOVE, (-1, 2**62)).fetchall():
            _sign_one(conn.execute, signer, fact_id, (profile_id, content, created_at))
            done += 1
        head = int(run(_HEAD, ())[0][0])
        run(_MARK_REPLACE, _mark_params(signer, head, epoch))
    return done


def _resign_signed(conn: sqlite3.Connection, signer: sk.Signer, only_other_key: bool,
                   limit: int | None) -> int:
    where = "WHERE s.key_id != ?" if only_other_key else "WHERE 1 = 1"
    params: tuple = (signer.key_id,) if only_other_key else ()
    rows = conn.execute(
        "SELECT f.fact_id, f.profile_id, f.content, f.created_at FROM fact_signatures s "
        f"JOIN atomic_facts f ON f.fact_id = s.fact_id {where} ORDER BY f.rowid"
        + (" LIMIT ?" if limit is not None else ""),
        params + ((int(limit),) if limit is not None else ())).fetchall()
    for fact_id, profile_id, content, created_at in rows:
        sig = signer.mac(canonical(profile_id, fact_id, content, created_at))
        conn.execute("UPDATE fact_signatures SET profile_id = ?, signed_profile_id = ?, sig = ?, "
                     "key_id = ?, signed_at = ? WHERE fact_id = ?",
                     (profile_id, profile_id, sig, signer.key_id, _now(), fact_id))
    return len(rows)


def forget_signatures(run: Callable[[str, tuple], Any], fact_ids: Iterable[str]) -> None:
    """A legitimate rewrite of a fact's text (a repair) drops its signature: the old one covers
    text that is gone. The memory then reads "signature missing" until a caller signs it again
    (``slm signing resign``): a pass cannot tell this from someone removing a signature, and
    must not sign it (A-12). Callers pass their own ``execute``; a store before M082 is skipped."""
    for fact_id in fact_ids:
        try:
            run("DELETE FROM fact_signatures WHERE fact_id = ?", (fact_id,))
        except Exception:  # noqa: BLE001 - a repair must never fail on a signature
            return  # a store before M082 has no table; the daemon signs what is there


__all__ = ["BATCH", "MISSING", "MODIFIED", "MOVED", "Mark", "OK", "OTHER_KEY", "RECORD_FORGED",
           "RECORD_MISSING", "RECORD_NONE", "RECORD_OK", "RECORD_OTHER_KEY", "RECORD_UNCHECKED",
           "Report", "SAMPLE", "UNSIGNED", "admit", "advance_progress", "canonical", "counts",
           "create_progress", "forget_signatures", "for_fact", "maintenance_step",
           "progress_state", "record_words", "reseal_moved", "resign", "sign_missing", "state_of",
           "verify"]
