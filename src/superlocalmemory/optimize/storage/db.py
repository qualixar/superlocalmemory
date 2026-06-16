"""CacheDB — wraps DatabaseManager for llmcache.db operations.

INTERFACE-CONTRACT §1 CONFORMANCE: Every method below is the canonical API.
No implementer may add aliases. Consumers (LLD-02/03/04/08) import this exact API.

REUSE: DatabaseManager from superlocalmemory/src/superlocalmemory/storage/database.py
  - WAL mode (database.py:66-74)
  - busy_timeout=10000ms (database.py:41)
  - _MAX_RETRIES=5 exponential backoff (database.py:127-148)
  - _connect() with row_factory (database.py:95-100)
  - transaction() context manager (database.py:102-116)
  - execute() with retry (database.py:118-148)

ENCRYPTION (resolves SEC-C-01 / CWE-312, NEW-M-01, NEW-M-02):
  - All value BLOBs (llmcache_entries.value_blob) are AES-256-GCM encrypted.
  - CCR original_blob is ALSO AES-256-GCM encrypted.
  - Key storage: a single MACHINE-WIDE key file (~/.superlocalmemory/opt-key.bin,
    0o600) is generated once and reused for all cache DBs on the machine. (The
    per-DB salt below is persisted for provenance but does NOT make the AES key
    per-DB — a single install has one llmcache.db, so a machine-wide key is the
    intended model. A tampered/rotated key now degrades to a cache MISS, not a
    crash — see _decrypt fail-open, v3.6.12 cache-1.)
  - Salt: os.urandom(32) generated ONCE at DB creation, stored in
    llmcache_schema_version.description='salt:<hex>'. NO hardcoded salt.
  - Nonce (12 bytes random) prepended to each ciphertext.
  - llmcache.db file permissions set to 0o600 at creation.

ISOLATION GUARANTEE: CacheDB MUST NOT import from superlocalmemory.storage.models.
Enforced by test_no_memory_db_import() in test_db.py.

FAIL-OPEN CONTRACT: Every public method catches sqlite3.Error, logs at WARNING,
and returns a safe default (None, [], False, 0) rather than raising.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import sqlite3
import struct
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.optimize.storage import schema as _schema

logger = logging.getLogger(__name__)

_DEFAULT_TENANT: str = "default"
_TENANT_HEX64 = re.compile(r"[0-9a-f]{64}")


def _normalize_tenant_id(tenant_id: str) -> str:
    """Match CacheManager.build_key tenant normalization (v3.6.10 fix).

    The proxy stores entries under SHA-256(tenant) when the tenant is not
    already a 64-char hex digest. Public tenant-scoped helpers (entry_count,
    clear_tenant, entry_exists) MUST apply the same hashing or they query the
    wrong tenant — the cause of the dashboard reporting "0 entries" while the
    cache is in fact populated.
    """
    tid = tenant_id or _DEFAULT_TENANT
    if _TENANT_HEX64.fullmatch(tid):
        return tid
    import hashlib as _hashlib
    return _hashlib.sha256(tid.encode()).hexdigest()


_ZLIB_LEVEL: int = 6
_AES_NONCE_BYTES: int = 12
_PBKDF2_ITERATIONS: int = 100_000

LLMCACHE_DIRNAME: str = ".superlocalmemory"
LLMCACHE_DBNAME: str = "llmcache.db"
MID_FILENAME: str = ".llmcache_key"
SALT_PREFIX: str = "salt:"

# C-06: persisted AES key — survives machine-id changes after first run
_KEY_FILE: Path = Path.home() / LLMCACHE_DIRNAME / "opt-key.bin"

_FORBIDDEN_MEMORY_TABLES: frozenset[str] = frozenset({
    "memories", "atomic_facts", "profiles", "canonical_entities",
    "entity_aliases", "consolidation_log", "trust_scores", "bm25_tokens",
    "fact_retention", "core_memory_blocks", "ccq_consolidated_blocks",
})


# ---------------------------------------------------------------------------
# MetricsSnapshot — single definition (INTERFACE-CONTRACT v2.2 §6)
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    """Mirror of llmcache_metrics columns — names MUST match exactly.

    S-03 / M-03 note: compress_bytes_original and compress_bytes_after store
    WORD-COUNT proxy values (len(text.split())), NOT byte counts. The column
    names use "bytes" for DB schema backward compatibility. Treat these fields
    as token-count proxies, not literal byte measurements.
    """
    id: int = 1
    hits: int = 0
    misses: int = 0
    calls_skipped: int = 0
    tokens_saved_input: int = 0
    tokens_saved_output: int = 0
    tokens_saved_compress: int = 0
    evictions: int = 0
    latency_overhead_ms_sum: float = 0.0
    latency_samples: int = 0
    compress_runs: int = 0
    compress_bytes_original: int = 0  # unit: word-count proxy (see S-03 note above)
    compress_bytes_after: int = 0     # unit: word-count proxy (see S-03 note above)
    cache_size_bytes: int = 0
    cache_entry_count: int = 0
    updated_at: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_latency_overhead_ms(self) -> float:
        return (
            self.latency_overhead_ms_sum / self.latency_samples
            if self.latency_samples > 0 else 0.0
        )

    @property
    def compression_ratio(self) -> float:
        if self.compress_bytes_original == 0:
            return 1.0
        return self.compress_bytes_after / self.compress_bytes_original


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CacheRow:
    """Return type of CacheDB.get() and get_by_id().

    All fields are optional via default so that SELECT * on a table with
    a different schema version does not blow up the constructor. The
    caller is expected to read attributes they need; missing fields
    return the default value.
    """
    entry_id: str = ""
    cache_key: str = ""
    tenant_id: str = "default"
    model: str = ""
    provider: str = ""
    value: bytes = b""
    created_at: str = ""
    last_hit_at: str | None = None
    ttl_expires: float | None = None
    hit_count: int = 0
    byte_size: int = 0
    tag_json: str = "[]"
    cache_tier: str = "exact"
    compressed: int = 1
    tags: list[str] = field(default_factory=list)


@dataclass
class BoundaryRow:
    """Per-item vCache MLE boundary record."""
    entry_id: str
    logistic_t: float = 0.95
    logistic_gamma: float = 10.0
    sample_count: int = 0
    updated_at: float = 0.0


# ---------------------------------------------------------------------------
# CacheDB
# ---------------------------------------------------------------------------

class CacheDB:
    """SQLite cache store backed by DatabaseManager. Canonical API per INTERFACE-CONTRACT §1."""

    _default_instance: "CacheDB | None" = None
    _default_lock: Any = None  # lazy-import threading.Lock on first use

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / LLMCACHE_DIRNAME / LLMCACHE_DBNAME
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # If the file exists but is not a valid SQLite database (e.g. user
        # accidentally pointed us at a binary blob), delete it and start fresh
        # so that the proxy never gets wedged on init. The proxy is the
        # critical path — we must be able to start even if the cache file
        # is corrupted. The previous file is left in a `.corrupt` sidecar
        # for forensic inspection.
        if self._db_path.exists():
            try:
                import sqlite3 as _sq
                test_conn = _sq.connect(str(self._db_path))
                test_conn.execute("PRAGMA schema_version")
                test_conn.close()
            except Exception as exc:
                corrupt_sidecar = self._db_path.with_suffix(
                    self._db_path.suffix + ".corrupt"
                )
                try:
                    if corrupt_sidecar.exists():
                        corrupt_sidecar.unlink()
                    self._db_path.replace(corrupt_sidecar)
                    try:
                        os.chmod(corrupt_sidecar, 0o600)
                    except OSError:
                        pass
                    logger.error(
                        "CacheDB: %s was not a SQLite file (%s) — moved to %s. "
                        "Starting with a fresh cache.",
                        self._db_path, exc, corrupt_sidecar,
                    )
                except OSError as move_exc:
                    logger.error(
                        "CacheDB: %s is not a SQLite file and could not be moved: %s",
                        self._db_path, move_exc,
                    )
                    raise
        self._db = DatabaseManager(self._db_path)
        self._db.initialize(_schema)
        # chmod 600 (SEC-C-01)
        try:
            os.chmod(self._db_path, 0o600)
        except OSError as exc:
            logger.warning("CacheDB: could not chmod 600 on %s: %s", self._db_path, exc)
        self._salt = self._load_or_create_salt()
        self._aes_key = self._get_or_persist_aes_key(self._salt)
        self.assert_no_memory_db_tables()

    # ---- context manager ----
    def __enter__(self) -> "CacheDB":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """No-op (per-call connection model)."""
        self._db.close()

    @property
    def db_path(self) -> str:
        return str(self._db_path)

    # ---- encryption helpers ----

    def _load_or_create_salt(self) -> bytes:
        rows = self._db.execute(
            "SELECT description FROM llmcache_schema_version "
            "WHERE description LIKE 'salt:%' LIMIT 1"
        )
        if rows:
            hex_salt = dict(rows[0])["description"][len(SALT_PREFIX):]
            if len(hex_salt) != 64:
                raise RuntimeError(
                    f"llmcache.db: salt row is malformed (len={len(hex_salt)}) — DB may be corrupted."
                )
            return bytes.fromhex(hex_salt)
        salt = os.urandom(32)
        self._db.execute(
            "INSERT INTO llmcache_schema_version (version, description) VALUES (?, ?)",
            (1, f"{SALT_PREFIX}{salt.hex()}"),
        )
        return salt

    def _get_machine_id(self) -> str:
        """Return a stable machine identifier for AES key derivation."""
        system = os.uname().sysname
        mid: str | None = None
        if system == "Darwin":
            try:
                import plistlib
                import subprocess
                out = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, timeout=2,
                )
                if out.returncode == 0:
                    text = out.stdout.decode("utf-8", errors="ignore")
                    for line in text.splitlines():
                        if "IOPlatformUUID" in line and "=" in line:
                            mid = line.split("=", 1)[1].strip().strip('"')
                            if mid:
                                break
            except Exception:
                mid = None
        elif system == "Linux":
            try:
                mid = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
            except OSError:
                mid = None
        if not mid:
            mid_file = Path.home() / LLMCACHE_DIRNAME / MID_FILENAME
            if mid_file.exists():
                try:
                    mid = mid_file.read_text(encoding="utf-8").strip()
                except OSError:
                    mid = None
            if not mid:
                mid = uuid.uuid4().hex
                try:
                    mid_file.parent.mkdir(parents=True, exist_ok=True)
                    mid_file.write_text(mid, encoding="utf-8")
                    os.chmod(mid_file, 0o600)
                except OSError as exc:
                    logger.warning("CacheDB: could not persist machine id: %s", exc)
        return mid

    def _get_or_persist_aes_key(self, salt: bytes) -> bytes:
        """C-06: Load persisted key from disk, or derive + persist on first run.

        Surviving a machine-id change: after first derivation the key is saved to
        opt-key.bin (0600). On subsequent starts the file is read directly,
        so changing the underlying machine-id string cannot invalidate existing
        cache entries.
        """
        try:
            if _KEY_FILE.exists():
                key = _KEY_FILE.read_bytes()
                if len(key) == 32:
                    return key
        except Exception as exc:
            logger.warning("CacheDB: could not read persisted AES key: %s", exc)

        # First run (or corrupted file): derive from machine-id and persist.
        machine_id = self._get_machine_id()
        key = self._derive_aes_key(machine_id, salt)
        try:
            _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
            _KEY_FILE.write_bytes(key)
            os.chmod(_KEY_FILE, 0o600)
        except Exception as exc:
            logger.warning("CacheDB: could not persist AES key (fail-open): %s", exc)
        return key

    def _derive_aes_key(self, machine_id: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=_PBKDF2_ITERATIONS,
        )
        return kdf.derive(machine_id.encode("utf-8"))

    def _encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(_AES_NONCE_BYTES)
        aesgcm = AESGCM(self._aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
        return nonce + ciphertext

    def _decrypt(self, blob: bytes) -> bytes:
        if len(blob) <= _AES_NONCE_BYTES:
            raise ValueError("Encrypted blob too short")
        nonce = blob[:_AES_NONCE_BYTES]
        ciphertext = blob[_AES_NONCE_BYTES:]
        aesgcm = AESGCM(self._aes_key)
        # v3.6.12 (cache-1): AES-GCM raises cryptography.exceptions.InvalidTag
        # (NOT a ValueError subclass) on a tampered/wrong-key blob. Every caller
        # catches ValueError to fail-open; convert InvalidTag -> ValueError here
        # at the single chokepoint so a corrupt/rotated-key cache entry degrades
        # to a miss instead of raising out of get()/get_value()/ccr_get().
        try:
            return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        except InvalidTag as exc:
            raise ValueError(f"AES-GCM authentication failed: {exc}") from exc

    # ---- assertion ----

    def assert_no_memory_db_tables(self) -> None:
        """Raise RuntimeError if memory.db tables are present."""
        rows = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        names = {dict(r)["name"] for r in rows}
        found = _FORBIDDEN_MEMORY_TABLES & names
        if found:
            raise RuntimeError(
                f"ISOLATION VIOLATION: llmcache.db contains memory.db tables: {sorted(found)}. "
                "Wrong DB file opened. Aborting."
            )

    # ---- exact CRUD ----

    def get(self, key: str, tenant_id: str) -> CacheRow | None:
        try:
            rows = self._db.execute(
                "SELECT * FROM llmcache_entries "
                "WHERE cache_key = ? AND tenant_id = ? "
                "AND (ttl_expires IS NULL OR ttl_expires > ?) "
                "LIMIT 1",
                (key, tenant_id, time.time()),
            )
            if not rows:
                return None
            row = dict(rows[0])
            # increment hit stats
            self._db.execute(
                "UPDATE llmcache_entries SET hit_count = hit_count + 1, "
                "last_hit_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE cache_key = ? AND tenant_id = ?",
                (key, tenant_id),
            )
            row["hit_count"] = row["hit_count"] + 1
            # decrypt + decompress value
            try:
                plaintext = self._decrypt(row["value_blob"])
                if row.get("compressed", 0):
                    plaintext = zlib.decompress(plaintext)
                row["value"] = plaintext
            except (ValueError, zlib.error) as exc:
                logger.warning("CacheDB.get decrypt failed (cache miss): %s", exc)
                return None
            return self._row_to_cacherow(row)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.get failed (cache miss): %s", exc)
            return None

    def get_value(self, cache_key: str, tenant_id: str) -> bytes | None:
        """Pure value lookup — no hit_count increment (unlike get()).

        Used by MCP KV tools which manage their own hit/miss counters.
        Caller must have already normalized tenant_id. Fail-open: returns None on error.
        """
        try:
            rows = self._db.execute(
                "SELECT value_blob, compressed FROM llmcache_entries "
                "WHERE cache_key = ? AND tenant_id = ? "
                "AND (ttl_expires IS NULL OR ttl_expires > ?) LIMIT 1",
                (cache_key, tenant_id, time.time()),
            )
            if not rows:
                return None
            row = dict(rows[0])
            plaintext = self._decrypt(row["value_blob"])
            if row.get("compressed", 0):
                plaintext = zlib.decompress(plaintext)
            return plaintext
        except (sqlite3.Error, ValueError, zlib.error) as exc:
            logger.warning("CacheDB.get_value failed (fail-open): %s", exc)
            return None

    def set(
        self,
        key: str,
        tenant_id: str,
        value: bytes,
        *,
        model: str,
        ttl_expires: float | None,
        tags: list[str],
    ) -> None:
        try:
            # 1) zlib-compress
            compressed_blob = zlib.compress(value, level=_ZLIB_LEVEL)
            # 2) AES-256-GCM encrypt
            encrypted_blob = self._encrypt(compressed_blob)
            entry_id = uuid.uuid4().hex
            self._db.execute(
                "INSERT INTO llmcache_entries "
                "(entry_id, cache_key, tenant_id, model, provider, value_blob, compressed, "
                " ttl_expires, tag_json, byte_size) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?) "
                "ON CONFLICT(cache_key, tenant_id) DO UPDATE SET "
                "  value_blob=excluded.value_blob, model=excluded.model, provider=excluded.provider, "
                "  ttl_expires=excluded.ttl_expires, tag_json=excluded.tag_json, "
                "  byte_size=excluded.byte_size, hit_count=0, last_hit_at=NULL, "
                "  created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (
                    entry_id, key, tenant_id, model, "",
                    encrypted_blob, ttl_expires,
                    json.dumps(tags or []), len(encrypted_blob),
                ),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.set failed (fail-open): %s", exc)

    def set_with_entry_id(
        self,
        key: str,
        tenant_id: str,
        value: bytes,
        entry_id: str,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Like set(), but writes a caller-supplied entry_id instead of auto-generating one.

        Used by semantic layer (index_entry) which pre-computes the entry_id from
        the embedding vector to enable O(1) semantic-to-cache cross-reference.
        """
        try:
            compressed_blob = zlib.compress(value, level=_ZLIB_LEVEL)
            encrypted_blob = self._encrypt(compressed_blob)
            ttl_expires = (time.time() + ttl_seconds) if ttl_seconds else None
            self._db.execute(
                "INSERT INTO llmcache_entries "
                "(entry_id, cache_key, tenant_id, model, provider, value_blob, compressed, "
                " ttl_expires, tag_json, byte_size) "
                "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?) "
                "ON CONFLICT(cache_key, tenant_id) DO UPDATE SET "
                "  entry_id=excluded.entry_id, value_blob=excluded.value_blob, "
                "  model=excluded.model, provider=excluded.provider, "
                "  ttl_expires=excluded.ttl_expires, tag_json=excluded.tag_json, "
                "  byte_size=excluded.byte_size, hit_count=0, last_hit_at=NULL, "
                "  created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (
                    entry_id, key, tenant_id, "", "",
                    encrypted_blob, ttl_expires,
                    json.dumps(tags or []), len(encrypted_blob),
                ),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.set_with_entry_id failed (fail-open): %s", exc)

    def get_by_id(self, entry_id: str) -> CacheRow | None:
        try:
            rows = self._db.execute(
                "SELECT * FROM llmcache_entries WHERE entry_id = ? LIMIT 1",
                (entry_id,),
            )
            if not rows:
                return None
            row = dict(rows[0])
            try:
                plaintext = self._decrypt(row["value_blob"])
                if row.get("compressed", 0):
                    plaintext = zlib.decompress(plaintext)
                row["value"] = plaintext
            except (ValueError, zlib.error) as exc:
                logger.warning("CacheDB.get_by_id decrypt failed: %s", exc)
                return None
            return self._row_to_cacherow(row)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.get_by_id failed: %s", exc)
            return None

    def delete(self, key: str, tenant_id: str) -> None:
        try:
            with self._db.transaction():
                rows = self._db.execute(
                    "SELECT entry_id FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ?",
                    (key, tenant_id),
                )
                entry_ids = [dict(r)["entry_id"] for r in rows]
                if entry_ids:
                    placeholders = ",".join("?" for _ in entry_ids)
                    self._db.execute(
                        f"DELETE FROM llmcache_semantic_vectors "
                        f"WHERE entry_id IN ({placeholders})",
                        tuple(entry_ids),
                    )
                self._db.execute(
                    "DELETE FROM llmcache_tags WHERE cache_key = ? AND tenant_id = ?",
                    (key, tenant_id),
                )
                self._db.execute(
                    "DELETE FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ?",
                    (key, tenant_id),
                )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.delete failed (fail-open): %s", exc)

    def delete_batch(self, keys: list[tuple[str, str]]) -> int:
        if not keys:
            return 0
        try:
            deleted = 0
            with self._db.transaction():
                for key, tenant_id in keys:
                    rows = self._db.execute(
                        "SELECT entry_id FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ?",
                        (key, tenant_id),
                    )
                    entry_ids = [dict(r)["entry_id"] for r in rows]
                    if entry_ids:
                        placeholders = ",".join("?" for _ in entry_ids)
                        self._db.execute(
                            f"DELETE FROM llmcache_semantic_vectors "
                            f"WHERE entry_id IN ({placeholders})",
                            tuple(entry_ids),
                        )
                    self._db.execute(
                        "DELETE FROM llmcache_tags WHERE cache_key = ? AND tenant_id = ?",
                        (key, tenant_id),
                    )
                    self._db.execute(
                        "DELETE FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ?",
                        (key, tenant_id),
                    )
                    deleted += 1
            return deleted
        except sqlite3.Error as exc:
            logger.warning("CacheDB.delete_batch failed: %s", exc)
            return 0

    def sweep_expired(self, now: float) -> int:
        try:
            with self._db.transaction():
                rows = self._db.execute(
                    "SELECT entry_id FROM llmcache_entries "
                    "WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
                    (now,),
                )
                entry_ids = [dict(r)["entry_id"] for r in rows]
                entries_deleted = len(entry_ids)
                if entry_ids:
                    placeholders = ",".join("?" for _ in entry_ids)
                    self._db.execute(
                        f"DELETE FROM llmcache_semantic_vectors "
                        f"WHERE entry_id IN ({placeholders})",
                        tuple(entry_ids),
                    )
                    self._db.execute(
                        f"DELETE FROM llmcache_tags "
                        f"WHERE cache_key IN ("
                        f"  SELECT cache_key FROM llmcache_entries WHERE entry_id IN ({placeholders})"
                        f")",
                        tuple(entry_ids),
                    )
                self._db.execute(
                    "DELETE FROM llmcache_entries "
                    "WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
                    (now,),
                )
                ccr_row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM llmcache_ccr_originals "
                    "WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
                    (now,),
                )
                ccr_count = int(dict(ccr_row[0])["n"]) if ccr_row else 0
                self._db.execute(
                    "DELETE FROM llmcache_ccr_originals "
                    "WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
                    (now,),
                )
            return entries_deleted + ccr_count
        except sqlite3.Error as exc:
            logger.warning("CacheDB.sweep_expired failed: %s", exc)
            return 0

    # ---- tags ----

    def tag_register(self, key: str, tenant_id: str, tags: list[str]) -> None:
        if not tags:
            return
        try:
            with self._db.transaction():
                for tag in tags:
                    self._db.execute(
                        "INSERT OR IGNORE INTO llmcache_tags (tag, cache_key, tenant_id) "
                        "VALUES (?, ?, ?)",
                        (tag, key, tenant_id),
                    )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.tag_register failed: %s", exc)

    def tag_keys(self, tag: str) -> list[tuple[str, str]]:
        try:
            rows = self._db.execute(
                "SELECT cache_key, tenant_id FROM llmcache_tags WHERE tag = ?",
                (tag,),
            )
            return [(dict(r)["cache_key"], dict(r)["tenant_id"]) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("CacheDB.tag_keys failed: %s", exc)
            return []

    def invalidate_by_tag(self, tag: str) -> int:
        try:
            with self._db.transaction():
                rows = self._db.execute(
                    "SELECT DISTINCT e.entry_id FROM llmcache_entries e "
                    "JOIN llmcache_tags t ON t.cache_key = e.cache_key AND t.tenant_id = e.tenant_id "
                    "WHERE t.tag = ?",
                    (tag,),
                )
                entry_ids = [dict(r)["entry_id"] for r in rows]
                if not entry_ids:
                    self._db.execute("DELETE FROM llmcache_tags WHERE tag = ?", (tag,))
                    return 0
                placeholders = ",".join("?" for _ in entry_ids)
                self._db.execute(
                    f"DELETE FROM llmcache_semantic_vectors WHERE entry_id IN ({placeholders})",
                    tuple(entry_ids),
                )
                self._db.execute(
                    f"DELETE FROM llmcache_entries WHERE entry_id IN ({placeholders})",
                    tuple(entry_ids),
                )
                self._db.execute("DELETE FROM llmcache_tags WHERE tag = ?", (tag,))
                return len(entry_ids)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.invalidate_by_tag failed: %s", exc)
            return 0

    # ---- semantic vectors ----

    def vec_add(self, entry_id: str, tenant_id: str, vector: bytes, meta: dict) -> None:
        try:
            dim = int(meta.get("dim", len(vector) // 4))
            model_name = str(meta.get("model", "nomic-ai/nomic-embed-text-v1.5"))
            context_fp = str(meta.get("context_fp", ""))  # C-10: persist context fingerprint
            self._db.execute(
                "INSERT OR REPLACE INTO llmcache_semantic_vectors "
                "(entry_id, tenant_id, vector_blob, vector_dim, model_name, context_fp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, tenant_id, vector, dim, model_name, context_fp),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.vec_add failed: %s", exc)

    def vec_search(self, tenant_id: str, vector: bytes, top_k: int) -> list[tuple[str, float]]:
        try:
            rows = self._db.execute(
                "SELECT entry_id, vector_blob FROM llmcache_semantic_vectors "
                "WHERE tenant_id = ?",
                (tenant_id,),
            )
            try:
                import numpy as _np
            except ImportError:
                return []
            q = _np.frombuffer(vector, dtype=_np.float32)
            q_norm = float(_np.linalg.norm(q))
            if q_norm == 0:
                return []
            scored: list[tuple[str, float]] = []
            for r in rows:
                rd = dict(r)
                v = _np.frombuffer(rd["vector_blob"], dtype=_np.float32)
                v_norm = float(_np.linalg.norm(v))
                if v_norm == 0:
                    continue
                cos = float(_np.dot(q, v) / (q_norm * v_norm))
                scored.append((rd["entry_id"], cos))
            scored.sort(key=lambda t: t[1], reverse=True)
            return scored[: max(0, top_k)]
        except sqlite3.Error as exc:
            logger.warning("CacheDB.vec_search failed: %s", exc)
            return []

    def vec_delete(self, entry_id: str) -> None:
        try:
            self._db.execute(
                "DELETE FROM llmcache_semantic_vectors WHERE entry_id = ?",
                (entry_id,),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.vec_delete failed: %s", exc)

    # ---- vCache boundary + centroids (Phase 3 hooks, stub-implemented) ----

    def boundary_get(self, entry_id: str) -> BoundaryRow | None:
        try:
            rows = self._db.execute(
                "SELECT * FROM llmcache_boundaries WHERE entry_id = ? LIMIT 1",
                (entry_id,),
            )
            if not rows:
                return None
            d = dict(rows[0])
            return BoundaryRow(**d)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.boundary_get failed: %s", exc)
            return None

    def boundary_upsert(self, entry_id: str, row: BoundaryRow) -> None:
        try:
            self._db.execute(
                "INSERT INTO llmcache_boundaries "
                "(entry_id, logistic_t, logistic_gamma, sample_count, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(entry_id) DO UPDATE SET "
                "  logistic_t=excluded.logistic_t, logistic_gamma=excluded.logistic_gamma, "
                "  sample_count=excluded.sample_count, updated_at=excluded.updated_at",
                (
                    entry_id, row.logistic_t, row.logistic_gamma,
                    row.sample_count, row.updated_at,
                ),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.boundary_upsert failed: %s", exc)

    def centroid_get(self, tenant_id: str) -> bytes | None:
        try:
            rows = self._db.execute(
                "SELECT centroid_blob FROM llmcache_centroids WHERE tenant_id = ? LIMIT 1",
                (tenant_id,),
            )
            if not rows:
                return None
            return dict(rows[0])["centroid_blob"]
        except sqlite3.Error as exc:
            logger.warning("CacheDB.centroid_get failed: %s", exc)
            return None

    def centroid_update(self, tenant_id: str, centroid: bytes, n: int) -> None:
        try:
            self._db.execute(
                "INSERT INTO llmcache_centroids (tenant_id, centroid_blob, n) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(tenant_id) DO UPDATE SET "
                "  centroid_blob=excluded.centroid_blob, n=excluded.n, "
                "  updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (tenant_id, centroid, n),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.centroid_update failed: %s", exc)

    # ---- v2.2 supplementary boundary methods (LLD-03 Phase 3) ----

    def get_all_boundaries(self) -> list[dict[str, Any]]:
        """Return all vCache MLE boundary records (warm-start / rebuild).

        LLD-03 §4.4: called at VCacheSemantic startup to populate the
        in-memory boundary cache.

        Returns:
            List of dicts with keys: entry_id, t_hat, gamma_hat,
            sample_count, updated_at. [] on error.
        """
        try:
            rows = self._db.execute(
                "SELECT entry_id, logistic_t, logistic_gamma, sample_count, updated_at "
                "FROM llmcache_boundaries"
            )
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.warning("CacheDB.get_all_boundaries failed: %s", exc)
            return []

    def delete_boundary(self, entry_id: str) -> None:
        """Delete a vCache boundary record. Fail-open."""
        try:
            self._db.execute(
                "DELETE FROM llmcache_boundaries WHERE entry_id = ?",
                (entry_id,),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.delete_boundary failed: %s", exc)

    def get_entry_by_id(self, entry_id: str) -> dict[str, Any] | None:
        """Fetch a cache entry by entry_id and return its decoded response dict.

        LLD-03 §4.5 _fetch_response: needed to retrieve the response for a
        semantically matched entry_id. Decrypts + decompresses the value_blob.

        Returns:
            Response dict (parsed JSON), or None on miss / error.
        """
        try:
            rows = self._db.execute(
                "SELECT value_blob, compressed FROM llmcache_entries "
                "WHERE entry_id = ? LIMIT 1",
                (entry_id,),
            )
            if not rows:
                return None
            row = dict(rows[0])
            try:
                plaintext = self._decrypt(row["value_blob"])
                if row.get("compressed", 0):
                    plaintext = zlib.decompress(plaintext)
            except (ValueError, zlib.error) as exc:
                logger.warning("CacheDB.get_entry_by_id decrypt failed: %s", exc)
                return None
            try:
                import json as _json
                return _json.loads(plaintext.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"raw_bytes": plaintext.hex()}
        except sqlite3.Error as exc:
            logger.warning("CacheDB.get_entry_by_id failed: %s", exc)
            return None

    # ---- CCR originals ----

    def ccr_put(
        self,
        ccr_id: str,
        original: bytes,
        ttl_expires: float | None = None,
    ) -> None:
        import hashlib
        try:
            compressed = zlib.compress(original, level=_ZLIB_LEVEL)
            encrypted = self._encrypt(compressed)
            self._db.execute(
                "INSERT OR REPLACE INTO llmcache_ccr_originals "
                "(ccr_id, original_blob, compressed_hash, byte_size_orig, byte_size_comp, ttl_expires) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ccr_id, encrypted,
                    hashlib.sha256(original).hexdigest(),
                    len(original), len(compressed),
                    ttl_expires,
                ),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.ccr_put failed: %s", exc)

    def ccr_get(self, ccr_id: str) -> bytes | None:
        try:
            rows = self._db.execute(
                "SELECT original_blob FROM llmcache_ccr_originals "
                "WHERE ccr_id = ? "
                "AND (ttl_expires IS NULL OR ttl_expires > ?)",
                (ccr_id, time.time()),
            )
            if not rows:
                return None
            blob = dict(rows[0])["original_blob"]
            plaintext = self._decrypt(blob)
            return zlib.decompress(plaintext)
        except (sqlite3.Error, ValueError, zlib.error) as exc:
            logger.warning("CacheDB.ccr_get failed: %s", exc)
            return None

    def ccr_update_compressed(self, ccr_id: str, compressed: bytes) -> None:
        import hashlib
        try:
            self._db.execute(
                "UPDATE llmcache_ccr_originals "
                "SET compressed_hash = ?, byte_size_comp = ? WHERE ccr_id = ?",
                (hashlib.sha256(compressed).hexdigest(), len(compressed), ccr_id),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.ccr_update_compressed failed: %s", exc)

    def ccr_delete(self, ccr_id: str) -> None:
        """Delete a CCR row by ccr_id. Idempotent — warns on sqlite error, never raises.

        WP-10 D6: defensive infra + sweep parity. Deleting a non-existent row is a no-op.
        """
        try:
            self._db.execute(
                "DELETE FROM llmcache_ccr_originals WHERE ccr_id = ?",
                (ccr_id,),
            )
        except sqlite3.Error as exc:
            logger.warning("CacheDB.ccr_delete failed (non-fatal): %s", exc)

    def ccr_count(self) -> int:
        """Return UNFILTERED count of rows in llmcache_ccr_originals.

        WP-10 CRIT-2: Do NOT reuse TTL-filtered count at :646. A fresh no-expiry row
        has ttl_expires=None, so the TTL filter returns 0 and D6 orphan tests would
        falsely pass. This unfiltered count is test infrastructure only.
        """
        try:
            rows = self._db.execute(
                "SELECT COUNT(*) AS n FROM llmcache_ccr_originals",
                (),
            )
            return int(dict(rows[0])["n"]) if rows else 0
        except sqlite3.Error as exc:
            logger.warning("CacheDB.ccr_count failed: %s", exc)
            return 0

    # ---- v2 additions ----

    def get_all_vectors(self, tenant_id: str) -> list[tuple[str, bytes, str]]:
        """Return (entry_id, vector_blob, context_fp) for all vectors in a tenant.

        C-10: context_fp is included so _lazy_warm_tenant() can restore it without
        recomputing embeddings from messages that may no longer be in scope.
        """
        try:
            rows = self._db.execute(
                "SELECT entry_id, vector_blob, context_fp FROM llmcache_semantic_vectors "
                "WHERE tenant_id = ?",
                (tenant_id,),
            )
            return [
                (dict(r)["entry_id"], dict(r)["vector_blob"], dict(r).get("context_fp", ""))
                for r in rows
            ]
        except sqlite3.Error as exc:
            logger.warning("CacheDB.get_all_vectors failed: %s", exc)
            return []

    # ---- metrics ----

    def metrics_load(self) -> MetricsSnapshot:
        try:
            rows = self._db.execute("SELECT * FROM llmcache_metrics WHERE id = 1")
            if not rows:
                return MetricsSnapshot()
            d = dict(rows[0])
            return MetricsSnapshot(**d)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.metrics_load failed: %s", exc)
            return MetricsSnapshot()

    def metrics_flush(self, snap: MetricsSnapshot) -> None:
        try:
            params = dataclasses.asdict(snap)
            sql = (
                "INSERT INTO llmcache_metrics ("
                "  id, hits, misses, calls_skipped, "
                "  tokens_saved_input, tokens_saved_output, tokens_saved_compress, "
                "  evictions, latency_overhead_ms_sum, latency_samples, "
                "  compress_runs, compress_bytes_original, compress_bytes_after, "
                "  cache_size_bytes, cache_entry_count, updated_at) "
                "VALUES ("
                "  :id, :hits, :misses, :calls_skipped, "
                "  :tokens_saved_input, :tokens_saved_output, :tokens_saved_compress, "
                "  :evictions, :latency_overhead_ms_sum, :latency_samples, "
                "  :compress_runs, :compress_bytes_original, :compress_bytes_after, "
                "  :cache_size_bytes, :cache_entry_count, :updated_at) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  hits=excluded.hits, misses=excluded.misses, calls_skipped=excluded.calls_skipped, "
                "  tokens_saved_input=excluded.tokens_saved_input, "
                "  tokens_saved_output=excluded.tokens_saved_output, "
                "  tokens_saved_compress=excluded.tokens_saved_compress, "
                "  evictions=excluded.evictions, "
                "  latency_overhead_ms_sum=excluded.latency_overhead_ms_sum, "
                "  latency_samples=excluded.latency_samples, "
                "  compress_runs=excluded.compress_runs, "
                "  compress_bytes_original=excluded.compress_bytes_original, "
                "  compress_bytes_after=excluded.compress_bytes_after, "
                "  cache_size_bytes=excluded.cache_size_bytes, "
                "  cache_entry_count=excluded.cache_entry_count, "
                "  updated_at=excluded.updated_at"
            )
            with self._db.transaction():
                self._db.execute(sql, params)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.metrics_flush failed: %s", exc)

    # ---- convenience / non-contract helpers ----

    def entry_exists(self, cache_key: str, tenant_id: str = _DEFAULT_TENANT) -> bool:
        tenant_id = _normalize_tenant_id(tenant_id)
        try:
            rows = self._db.execute(
                "SELECT 1 FROM llmcache_entries WHERE cache_key = ? AND tenant_id = ? LIMIT 1",
                (cache_key, tenant_id),
            )
            return bool(rows)
        except sqlite3.Error:
            return False

    def clear_tenant(self, tenant_id: str) -> int:
        tenant_id = _normalize_tenant_id(tenant_id)
        try:
            with self._db.transaction():
                rows = self._db.execute(
                    "SELECT entry_id FROM llmcache_entries WHERE tenant_id = ?",
                    (tenant_id,),
                )
                entry_ids = [dict(r)["entry_id"] for r in rows]
                if entry_ids:
                    placeholders = ",".join("?" for _ in entry_ids)
                    self._db.execute(
                        f"DELETE FROM llmcache_semantic_vectors "
                        f"WHERE entry_id IN ({placeholders})",
                        tuple(entry_ids),
                    )
                self._db.execute(
                    "DELETE FROM llmcache_tags WHERE tenant_id = ?", (tenant_id,),
                )
                self._db.execute(
                    "DELETE FROM llmcache_entries WHERE tenant_id = ?", (tenant_id,),
                )
                return len(entry_ids)
        except sqlite3.Error as exc:
            logger.warning("CacheDB.clear_tenant failed: %s", exc)
            return 0

    def entry_count(self, tenant_id: str = _DEFAULT_TENANT) -> int:
        tenant_id = _normalize_tenant_id(tenant_id)
        try:
            rows = self._db.execute(
                "SELECT COUNT(*) AS n FROM llmcache_entries "
                "WHERE tenant_id = ? AND (ttl_expires IS NULL OR ttl_expires > ?)",
                (tenant_id, time.time()),
            )
            return int(dict(rows[0])["n"]) if rows else 0
        except sqlite3.Error:
            return 0

    def db_size_bytes(self) -> int:
        try:
            rows = self._db.execute("PRAGMA page_count")
            page_count = int(dict(rows[0])["page_count"]) if rows else 0
            rows = self._db.execute("PRAGMA page_size")
            page_size = int(dict(rows[0])["page_size"]) if rows else 0
            return page_count * page_size
        except sqlite3.Error:
            return 0

    # ---- internal helpers ----

    def _row_to_cacherow(self, row: dict) -> "CacheRow":
        """Build a CacheRow from a sqlite3.Row dict, robust to extra/missing keys."""
        kwargs: dict[str, Any] = {}
        for fname in CacheRow.__dataclass_fields__:
            if fname == "value":
                continue
            if fname in row:
                kwargs[fname] = row[fname]
        kwargs["value"] = row.get("value", b"")
        try:
            kwargs["tags"] = json.loads(row.get("tag_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            kwargs["tags"] = []
        return CacheRow(**kwargs)

    # ---- singleton (INTERFACE-CONTRACT v2.2 §1) ----

    @classmethod
    def get_default(cls) -> "CacheDB":
        if cls._default_instance is None:
            import threading as _t
            if cls._default_lock is None:
                cls._default_lock = _t.Lock()
            with cls._default_lock:
                if cls._default_instance is None:
                    cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the singleton (testing only)."""
        if cls._default_instance is not None:
            try:
                cls._default_instance.close()
            except Exception:
                pass
        cls._default_instance = None
