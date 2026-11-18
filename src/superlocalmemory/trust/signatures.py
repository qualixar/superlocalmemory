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
facts that have no signature yet (:func:`maintenance_step`), so a fact saved by a hook or
a CLI process, or before the flag went on, is signed by the daemon later. A signature is
written once and never changed by a pass: a changed fact stays "modified" until a person
decides otherwise.

*What a profile fold does.* The fold moves the fact (``profile_id`` changes, nothing else),
so a signature over the old profile no longer matches the new one. The row keeps
``signed_profile_id`` (the profile the signature covers). The daemon's pass verifies the
old signature over ``signed_profile_id`` and, only if it holds, signs again for the new
profile (**re-signed in the daemon's pass, not in the fold**: the fold runs without the
key). A bad old signature stays bad. Until then the fact is counted ``moved``, not modified.

*What it does not cover.* Removing a signature row is reported as ``unsigned`` (never as
intact), but the next pass signs the fact as it then stands: a signature attests "unchanged
since it was signed", not "never altered before". Signing a store for the first time seals
what is in it.

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


def canonical(profile_id: str, fact_id: str, content: str, created_at: str) -> bytes:
    """The bytes that are signed. Field boundaries cannot move: it is one JSON object."""
    record = {"domain": DOMAIN, "profile": profile_id, "fact": fact_id,
              "content_sha256": hashlib.sha256((content or "").encode("utf-8")).hexdigest(),
              "created_at": created_at or ""}
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


_FACT = "SELECT profile_id, content, created_at FROM atomic_facts WHERE fact_id = ?"
_INSERT = ("INSERT OR IGNORE INTO fact_signatures (fact_id, profile_id, signed_profile_id, "
           "sig, key_id, signed_at) VALUES (?, ?, ?, ?, ?, ?)")


def _sign_one(run: Callable[[str, tuple], Any], signer: sk.Signer, fact_id: str,
              row: tuple) -> bool:
    profile_id, content, created_at = row
    sig = signer.mac(canonical(profile_id, fact_id, content, created_at))
    run(_INSERT, (fact_id, profile_id, profile_id, sig, signer.key_id, _now()))
    return True


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
        rows = db.execute(_FACT, (fact.fact_id,))
        if rows:
            r = rows[0]
            _sign_one(lambda sql, params: db.execute(sql, params), signer, fact.fact_id,
                      (r[0], r[1], r[2]))
    except Exception as exc:  # noqa: BLE001 - a signature is evidence, never a lost save
        logger.warning("fact not signed %s (%s)", getattr(fact, "fact_id", "?"),
                       type(exc).__name__)


# -- background batches ---------------------------------------------------------------------------

_UNSIGNED = ("SELECT f.fact_id, f.profile_id, f.content, f.created_at FROM atomic_facts f "
             "LEFT JOIN fact_signatures s ON s.fact_id = f.fact_id "
             "WHERE s.fact_id IS NULL ORDER BY f.rowid LIMIT ?")
_MOVED = ("SELECT s.fact_id, s.signed_profile_id, s.sig, s.key_id, f.profile_id, f.content, "
          "f.created_at FROM fact_signatures s JOIN atomic_facts f ON f.fact_id = s.fact_id "
          "WHERE s.signed_profile_id != f.profile_id AND s.profile_id = f.profile_id "
          "ORDER BY f.rowid LIMIT ?")


def sign_missing(conn: sqlite3.Connection, signer: sk.Signer, *, limit: int = BATCH) -> int:
    """Sign up to ``limit`` facts that have no signature. Returns how many were signed."""
    done = 0
    for fact_id, profile_id, content, created_at in conn.execute(_UNSIGNED, (int(limit),)).fetchall():
        _sign_one(conn.execute, signer, fact_id, (profile_id, content, created_at))
        done += 1
    return done


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


def maintenance_step(db: Any, *, max_batches: int = 5) -> int:
    """One maintenance cycle's share, only while the daemon holds the key and the flag observes."""
    signer = sk.current()
    if signer is None or not _observing():
        return 0
    total = 0
    with db.raw_connection() as conn:
        try:
            for _ in range(max_batches):
                moved = reseal_moved(conn, signer)
                fresh = sign_missing(conn, signer)
                total += moved + fresh
                if moved < BATCH and fresh < BATCH:
                    break
        except sqlite3.OperationalError:  # a store before M082
            return 0
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

    @property
    def ok(self) -> bool:
        """Nothing found modified. Unsigned and other-key facts are reported, not failures."""
        return self.modified == 0

    def as_dict(self) -> dict[str, Any]:
        return {"key_status": self.key_status, "key_words": sk.words(self.key_status),
                "facts": self.facts, "signed": self.signed, "unsigned": self.unsigned,
                "checked": self.checked, "modified": self.modified,
                "signed_under_another_key": self.other_key, "moved_by_a_fold": self.moved,
                "modified_ids": list(self.modified_ids), "exhaustive": self.exhaustive}


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
    if signer is None or signed == 0:
        return Report(key_status, facts, signed, unsigned)
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
                  tuple(bad), exhaustive=sample is None or len(rows) >= signed)


def for_fact(conn: sqlite3.Connection, fact_id: str, signer: sk.Signer | None) -> dict[str, Any]:
    """One fact's signature state for ``slm why``. Never reads the keychain: ``signer`` is
    whatever the caller already holds (the daemon's key, or ``None``)."""
    try:
        sig = conn.execute("SELECT key_id, signed_at FROM fact_signatures WHERE fact_id = ?",
                           (fact_id,)).fetchone()
    except sqlite3.OperationalError:
        return {"state": "not_available", "words": "This store does not keep signatures yet."}
    if sig is None:
        return {"state": UNSIGNED, "words": "Not signed (yet)."}
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


def resign(conn: sqlite3.Connection, signer: sk.Signer, *, only_other_key: bool = True,
           limit: int | None = None) -> int:
    """Sign again, with the current key, facts whose signature cannot be checked now.

    A person's decision (``slm signing resign``, at a terminal): it blesses what is in the
    store as it stands, so it is never done by a pass. ``only_other_key`` limits it to
    signatures made under a key this computer no longer holds (a lost or replaced key).
    """
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
    """A legitimate rewrite of a fact's text (a repair) drops its signature so the daemon
    signs the new text. Callers pass their own ``execute``; a store before M082 is skipped."""
    for fact_id in fact_ids:
        try:
            run("DELETE FROM fact_signatures WHERE fact_id = ?", (fact_id,))
        except Exception:  # noqa: BLE001 - a repair must never fail on a signature
            return  # a store before M082 has no table; the daemon signs what is there


__all__ = ["BATCH", "MODIFIED", "MOVED", "OK", "OTHER_KEY", "Report", "SAMPLE", "UNSIGNED",
           "admit", "canonical", "counts", "forget_signatures", "for_fact", "maintenance_step",
           "reseal_moved", "resign", "sign_missing", "state_of", "verify"]
