# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Mesh broker security helpers — startup integrity, state guards, fencing.

Pure functions so they can be unit-tested independently of the broker.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger("superlocalmemory.mesh")

# ---------------------------------------------------------------------------
# 3a-1  Per-message HMAC identity (sign / verify / replay defense)
# ---------------------------------------------------------------------------

#: Seconds of acceptable clock skew between signer and verifier.
MESH_SIG_SKEW_SECONDS: int = int(os.environ.get("MESH_SIG_SKEW_SECONDS", "300"))

#: Hard cap on the in-memory nonce table to prevent memory exhaustion.
_NONCE_STORE_MAX: int = 10_000

_nonce_lock = threading.Lock()
# Maps nonce → expiry monotonic timestamp (in-memory fast path)
_nonce_store: dict[str, float] = {}
# SQLite DB path wired by MeshBroker.__init__ for durable replay defense (SEC-3)
_nonce_db_path: str | None = None


def _set_nonce_db_path(db_path: str) -> None:
    """Wire the SQLite path for durable nonce storage. Called from MeshBroker.__init__."""
    global _nonce_db_path
    _nonce_db_path = db_path


def _clear_nonce_store() -> None:
    """Purge the nonce table — for use in tests only. Clears both memory and SQLite."""
    global _nonce_db_path
    with _nonce_lock:
        _nonce_store.clear()
        if _nonce_db_path:
            try:
                conn = sqlite3.connect(_nonce_db_path, timeout=3)
                conn.execute("DELETE FROM mesh_nonces")
                conn.commit()
                conn.close()
            except sqlite3.Error:
                pass


def _prune_nonces(now: float) -> None:
    """Evict expired entries (caller must hold _nonce_lock). SEC-1: strict < not <=."""
    expired = [n for n, exp in _nonce_store.items() if exp < now]
    for n in expired:
        del _nonce_store[n]


def _register_nonce(nonce: str, now: float) -> bool:
    """Register *nonce*; return False if already seen (replay), True if fresh.

    SEC-3: When SQLite path is wired, persists to mesh_nonces for durability
    across process restarts. The INSERT OR IGNORE is atomic; rowcount==0 means
    a prior process already registered this nonce (cross-restart replay detected).
    """
    with _nonce_lock:
        _prune_nonces(now)
        if nonce in _nonce_store:
            return False
        # Emergency eviction when the store is full (bounded)
        if len(_nonce_store) >= _NONCE_STORE_MAX:
            oldest = min(_nonce_store, key=lambda k: _nonce_store[k])
            del _nonce_store[oldest]
        exp = now + MESH_SIG_SKEW_SECONDS
        # SEC-3: persist to SQLite when wired — atomic INSERT OR IGNORE detects cross-restart replays
        if _nonce_db_path:
            try:
                conn = sqlite3.connect(_nonce_db_path, timeout=3)
                # P2: bound table growth — drop nonces whose replay window has passed.
                conn.execute("DELETE FROM mesh_nonces WHERE expires_at < ?", (now,))
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO mesh_nonces (nonce, expires_at) VALUES (?, ?)",
                    (nonce, exp),
                )
                conn.commit()
                rowcount = cursor.rowcount
                conn.close()
                if rowcount == 0:
                    return False  # SQLite already has this nonce — cross-restart replay
            except sqlite3.Error:
                # P1: FAIL CLOSED. A durable nonce store is configured but the write
                # could not be confirmed, so replay defense cannot be guaranteed for
                # this signed message — reject rather than silently degrade to
                # process-local memory (which reopens the multi-worker replay hole).
                return False
        _nonce_store[nonce] = exp
        return True


def _canonical_payload(
    from_peer: str, to: str, content: str, nonce: str, ts: str
) -> str:
    """Deterministic string signed by HMAC. Covers SHA-256 of content for integrity.

    SEC-2: Fields joined with NUL byte (\\x00) to prevent field injection collisions
    that pipe (|) allows. E.g. ('a|b','c') and ('a','b|c') produce identical pipe
    payloads but distinct NUL payloads — no forgery via field boundaries.
    """
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return "\x00".join([from_peer, to, content_hash, nonce, ts])


def sign_mesh_message(
    secret: str,
    from_peer: str,
    to: str,
    content: str,
    nonce: str,
    ts: str,
) -> str:
    """Return hex HMAC-SHA256 over the canonical mesh payload."""
    payload = _canonical_payload(from_peer, to, content, nonce, ts)
    return _hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def verify_mesh_message(
    secret: str,
    from_peer: str,
    to: str,
    content: str,
    nonce: str,
    ts: str,
    sig: str,
) -> bool:
    """Constant-time compare of presented sig vs expected. Returns False on any mismatch."""
    try:
        expected = sign_mesh_message(secret, from_peer, to, content, nonce, ts)
        return _hmac.compare_digest(expected, sig)
    except Exception:
        return False


def is_strict_identity(config: object | None = None) -> bool:
    """Return True if strict per-message HMAC is required for inbound remote messages.

    SEC-7: Checks env vars first (SLM_MESH_PRODUCTION=1 or SLM_MESH_STRICT_IDENTITY=1),
    then falls back to config.mesh_strict_identity if a config object is passed.
    Default False — preserves backward compat for existing deployments.
    """
    _truthy = frozenset({"1", "true", "yes", "on", "production", "prod"})
    if os.environ.get("SLM_MESH_PRODUCTION", "").strip().lower() in _truthy:
        return True
    if os.environ.get("SLM_MESH_STRICT_IDENTITY", "").strip().lower() in _truthy:
        return True
    if config is not None and getattr(config, "mesh_strict_identity", False):
        return True
    return False


def _has_forbidden_chars(field: str) -> bool:
    """Return True if the field contains NUL bytes or ASCII control characters < 0x20.

    SEC-2: These characters could be used to inject extra NUL-delimited fields into
    the canonical payload, bypassing per-field binding. Reject them at the gate.
    """
    return any(ord(c) < 0x20 for c in field)


def check_mesh_message_signature(
    secret: str | None,
    from_peer: str,
    to: str,
    content: str,
    sig_header: str | None,
    nonce_header: str | None,
    ts_header: str | None,
    *,
    is_loopback: bool,
    strict: bool,
) -> dict | None:
    """Gate inbound mesh message signatures.

    Returns None on acceptance, or ``{"ok": False, "error": "..."}`` on rejection.

    Backward-compat rules (NON-NEGOTIABLE):
      - Loopback is always trusted; no sig required regardless of strict.
      - strict=False + no sig → accepted (unsigned legacy remote).
      - strict=False + bad sig present → rejected (reject known-bad, not silent swallow).
      - strict=True + no sig → rejected.
      - strict=True + valid sig → accepted.
    """
    if is_loopback:
        return None  # Always trusted; existing require_write_actor path unchanged.

    # SEC-2: Reject control characters (including NUL) in canonicalized fields.
    # NUL is the canonical delimiter — a NUL in a field injects a phantom field boundary.
    for field_name, field_val in (("from_peer", from_peer), ("nonce", nonce_header or ""),
                                  ("to", to)):
        if _has_forbidden_chars(field_val):
            return {"ok": False, "error": f"field '{field_name}' contains forbidden control characters"}

    has_sig = bool(sig_header)

    if not has_sig:
        if strict:
            return {"ok": False, "error": "missing message signature (strict identity mode)"}
        return None  # unsigned legacy remote accepted in compat mode

    # Signature present — must be well-formed and valid regardless of strict flag.
    if not secret:
        # A sig was presented but we have no secret to verify against.
        if strict:
            return {"ok": False, "error": "signature present but no shared secret configured"}
        return None  # compat mode: can't verify, treat as unsigned legacy

    if not nonce_header or not ts_header:
        return {"ok": False, "error": "X-Mesh-Sig present but X-Mesh-Nonce or X-Mesh-Ts missing"}

    # Timestamp skew check
    try:
        ts_float = float(ts_header)
    except (ValueError, TypeError):
        return {"ok": False, "error": "X-Mesh-Ts is not a valid unix timestamp"}

    now = time.time()
    if abs(now - ts_float) > MESH_SIG_SKEW_SECONDS:
        return {"ok": False, "error": "message timestamp outside acceptable skew window"}

    # HMAC verification (before nonce registration — prevent timing oracle)
    if not verify_mesh_message(secret, from_peer, to, content, nonce_header, ts_header, sig_header):
        return {"ok": False, "error": "message signature verification failed"}

    # Nonce replay check (register after HMAC so only valid sigs consume a slot)
    if not _register_nonce(nonce_header, now):
        return {"ok": False, "error": "message nonce has been used before (replay rejected)"}

    return None  # All checks passed


# ---------------------------------------------------------------------------
# 3a-2  Content scrub helper
# ---------------------------------------------------------------------------


def scrub_message_content(content: str) -> str:
    """Redact known secret patterns from message content before durable storage.

    Fail-open: on any import/runtime error the original content is returned
    unchanged so a storage failure never loses a message.
    """
    try:
        from superlocalmemory.core.security_primitives import redact_secrets

        return redact_secrets(content)
    except Exception:
        return content


# ---------------------------------------------------------------------------
# 3a-3  Restart-safe fencing counter seed
# ---------------------------------------------------------------------------


def seed_fencing_counter(db_path: str) -> int:
    """Return MAX(fencing_token) from mesh_locks so post-restart tokens exceed
    any surviving DB value.

    SEC-6 fail-closed semantics:
      - DB connect error OR table missing → return 0 (first-run / degraded, safe default).
      - Table EXISTS but MAX query errors → raise RuntimeError (split-brain prevention:
        starting at 0 after a query failure on an existing lock table would issue
        fencing tokens below the highest previously-issued value, invalidating live locks).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return 0  # Can't even open the DB — first-run or degraded

    try:
        # Check if the table exists first so we can distinguish "no table" from "query error"
        table_exists_row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='mesh_locks'"
        ).fetchone()
        if not table_exists_row:
            return 0  # First run — no table yet, starting from 0 is safe
        # Table exists: query MUST succeed. Fail-closed if it errors.
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(COALESCE(fencing_token, 0)), 0) AS max_tok "
                "FROM mesh_locks"
            ).fetchone()
            return int(row["max_tok"]) if row else 0
        except sqlite3.OperationalError as exc:
            err_lower = str(exc).lower()
            if "no such column" in err_lower:
                # Table exists but fencing_token column not yet added (pre-migration schema).
                # Starting at 0 is safe because no tokens were ever issued.
                return 0
            raise RuntimeError(
                f"mesh_locks table exists but MAX(fencing_token) query failed: {exc}. "
                "Cannot seed fencing counter safely — refusing to start at 0 (split-brain risk)."
            ) from exc
    finally:
        conn.close()

# Pattern matches key names that imply secret material.
import re as _re
STATE_SECRET_KEY = _re.compile(
    r"(?:^|[_\-.])(api[_\-.]?key|secret|token|password|credential)(?:$|[_\-.]|$)",
    _re.IGNORECASE,
)

_SCHEMA_ALTERS = (
    "ALTER TABLE mesh_locks ADD COLUMN fencing_token INTEGER DEFAULT 0",
    "ALTER TABLE mesh_state ADD COLUMN revision INTEGER DEFAULT 0",
    "ALTER TABLE mesh_peers ADD COLUMN peer_key TEXT",  # SEC-4: per-peer HMAC key
    # 3c-1: LWW provenance. '' = local-origin (set by broker.set_state);
    # a merged row stores the winning remote node_id. See mesh/state_sync.py.
    "ALTER TABLE mesh_state ADD COLUMN origin_node TEXT NOT NULL DEFAULT ''",
)
_SENT_OPS_DDL = """
CREATE TABLE IF NOT EXISTS mesh_sent_ops (
    operation_id TEXT PRIMARY KEY,
    message_id   INTEGER NOT NULL,
    created_at   TEXT NOT NULL
)"""
# SEC-3: durable nonce store — survives process restarts
_NONCES_DDL = """
CREATE TABLE IF NOT EXISTS mesh_nonces (
    nonce      TEXT PRIMARY KEY,
    expires_at REAL NOT NULL
)"""


def ensure_db_healthy(db_path: str) -> bool:
    """Return True (degraded) if the DB was corrupt and had to be quarantined.

    Quarantine = rename to ``<name>.quarantine-<ms>``.  The original bytes
    are preserved; the caller receives a fresh empty DB on the same path.
    A missing DB is a normal first-run situation and is not an error.
    """
    path = Path(db_path)
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT count(*) FROM sqlite_master")
        conn.close()
        return False
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
        ts = int(time.monotonic() * 1_000)
        quarantine = path.with_name(f"{path.name}.quarantine-{ts}")
        try:
            path.rename(quarantine)
        except OSError as rename_err:
            logger.error("mesh db corrupt and could not be quarantined: %s", rename_err)
            return False
        logger.warning(
            "mesh db corrupt (%s); quarantined to %s; starting fresh",
            exc, quarantine.name,
        )
        return True


def apply_security_schema(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema additions (fencing_token, revision, mesh_sent_ops, mesh_nonces, peer_key)."""
    for sql in _SCHEMA_ALTERS:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists
    try:
        conn.executescript(_SENT_OPS_DDL)
    except sqlite3.OperationalError:
        pass
    try:
        conn.executescript(_NONCES_DDL)
    except sqlite3.OperationalError:
        pass
    conn.commit()


def get_or_create_peer_key(
    conn: sqlite3.Connection, peer_id: str, profile_id: str
) -> str:
    """Return the stored peer_key for *peer_id*, minting a new one if absent.

    SEC-4: Each registered peer gets a unique 32-byte (256-bit) HMAC key that only
    they receive at registration time. Signatures in strict mode are verified against
    this key — not the shared fleet secret — so knowing the fleet secret cannot forge
    another peer's identity.
    """
    row = conn.execute(
        "SELECT peer_key FROM mesh_peers WHERE peer_id=? AND profile_id=?",
        (peer_id, profile_id),
    ).fetchone()
    if row and row["peer_key"]:
        return str(row["peer_key"])
    key = secrets.token_hex(32)  # 256-bit random key
    conn.execute(
        "UPDATE mesh_peers SET peer_key=? WHERE peer_id=? AND profile_id=?",
        (key, peer_id, profile_id),
    )
    conn.commit()
    return key


def reject_secret_state(key: str, value: str) -> dict | None:
    """Return an error dict if key or value looks like a secret, else None."""
    if STATE_SECRET_KEY.search(key):
        return {"ok": False, "error": "mesh state is coordination metadata; secret key names are prohibited"}
    try:
        from superlocalmemory.core.security_primitives import redact_secrets
        if redact_secrets(value) != value:
            return {"ok": False, "error": "mesh state is coordination metadata; secret values are prohibited"}
    except ImportError:
        pass
    return None


def check_cross_profile_sender(
    conn: sqlite3.Connection, from_peer: str, profile_id: str
) -> dict | None:
    """Return an error dict if from_peer is a known peer in a different profile.

    Arbitrary label strings (not registered anywhere) are allowed — they are
    metadata, not identity claims.  Only a server-assigned peer_id that
    belongs to a different profile is rejected (cross-profile impersonation).
    """
    if not from_peer:
        return None
    row = conn.execute(
        "SELECT profile_id FROM mesh_peers WHERE peer_id=? LIMIT 1",
        (from_peer,),
    ).fetchone()
    if row is not None and row["profile_id"] != profile_id:
        return {"ok": False, "error": "from_peer belongs to a different profile"}
    return None


def validate_lock_fence_query(
    conn: sqlite3.Connection,
    file_path: str,
    fencing_token: int,
    profile_id: str,
) -> dict:
    """Compare presented fencing_token against the current lock record."""
    row = conn.execute(
        "SELECT COALESCE(fencing_token, 0) AS fencing_token "
        "FROM mesh_locks WHERE profile_id=? AND file_path=?",
        (profile_id, file_path),
    ).fetchone()
    if row is None:
        return {"ok": False, "error": "no lock held for this resource"}
    current = row["fencing_token"]
    if fencing_token < current:
        return {"ok": False, "error": f"fencing token {fencing_token} is stale; current token is {current}"}
    return {"ok": True, "fencing_token": current}
