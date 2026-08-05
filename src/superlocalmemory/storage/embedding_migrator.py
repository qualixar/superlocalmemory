# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Embedding migration on mode/model switch.

When a user switches modes (e.g., Mode B Ollama -> Mode A sentence-transformers),
the embeddings live in different vector spaces. This module detects the mismatch
and flags facts for progressive re-embedding.

Key table: ``embedding_metadata.model_name`` stores the model used for each fact.
A config-level field in ``config.json`` stores the current model signature.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os as _os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backfill constants
# ---------------------------------------------------------------------------

#: Default batch size for backfill_missing_embeddings.
_BACKFILL_BATCH_SIZE = 50

#: Cooperative yield between per-fact writes in backfill_missing_embeddings.
#: After every fact's UPDATE + INSERT pair, the write-back loop releases
#: db._lock for this many seconds, giving concurrent user writes a guaranteed
#: acquisition window.  Tunable via SLM_SELFHEAL_WRITE_DELAY_S; set to 0 to
#: disable (only do this on single-user dev databases with no concurrency).
#: Default 5 ms is imperceptible for humans but visible to the OS scheduler.
_SELFHEAL_WRITE_DELAY_S: float = float(_os.environ.get("SLM_SELFHEAL_WRITE_DELAY_S", "0.005"))

#: Max characters embedded per fact during backfill. The embedding model
#: (nomic-embed-text-v1.5) truncates at ~8192 tokens anyway, but a raw
#: oversized document (observed up to 107 KB on a real DB) makes the shared
#: single-worker embedder busy for 15-20s on ONE fact — starving foreground
#: recall during a self-heal pass. Bounding the input keeps every fact's embed
#: fast and the worker responsive; the leading slice captures the fact's gist
#: for semantic recall. Facts this large are documents that were almost
#: certainly NULL because they failed to embed at ingestion for the same reason.
_MAX_EMBED_CHARS = 8000

# Sentinel stored in config.json when no model has been set yet.
_NO_MODEL = ""

# Batch size for progressive re-embedding.
_REINDEX_BATCH_SIZE = 50


def _activate_staged_vectors(
    config: SLMConfig,
    db: Any,
    stage_path: Path,
    expected_count: int,
) -> None:
    """Atomically replace canonical and sqlite-vec embeddings from a shadow DB."""
    db_path = getattr(db, "db_path", None)
    if not isinstance(db_path, (str, Path)):
        raise RuntimeError("embedding migration requires an authoritative db_path")

    import sqlite_vec

    from superlocalmemory.storage.write_lock import get_write_lock

    db_path = Path(db_path)
    with get_write_lock(db_path), sqlite3.connect(stage_path) as stage:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA busy_timeout=10000")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("BEGIN IMMEDIATE")
            canonical = conn.execute(
                "SELECT fact_id, profile_id, content "
                "FROM atomic_facts ORDER BY fact_id"
            )
            staged = stage.execute(
                "SELECT fact_id, profile_id, content_hash "
                "FROM staged_embeddings ORDER BY fact_id"
            )
            for current, shadow in itertools.zip_longest(canonical, staged):
                if current is None or shadow is None:
                    raise RuntimeError("canonical fact set changed during migration")
                current_key = (str(current[0]), str(current[1]))
                shadow_key = (str(shadow[0]), str(shadow[1]))
                current_hash = hashlib.sha256(str(current[2]).encode("utf-8")).hexdigest()
                if current_key != shadow_key or current_hash != str(shadow[2]):
                    raise RuntimeError(
                        f"canonical fact changed during migration: {current_key[0]}"
                    )
            conn.execute("DROP TABLE IF EXISTS embedding_metadata")
            conn.execute("DROP TABLE IF EXISTS vector_row_map")
            conn.execute("DROP TABLE IF EXISTS fact_embeddings")
            conn.execute(
                "CREATE VIRTUAL TABLE fact_embeddings USING vec0("
                "profile_id TEXT PARTITION KEY, "
                f"embedding float[{config.embedding.dimension}] distance_metric=cosine)"
            )
            conn.execute(
                "CREATE TABLE embedding_metadata ("
                "vec_rowid INTEGER PRIMARY KEY, fact_id TEXT NOT NULL UNIQUE, "
                "profile_id TEXT NOT NULL DEFAULT 'default', "
                "model_name TEXT NOT NULL DEFAULT '', "
                "dimension INTEGER NOT NULL DEFAULT 768, "
                "created_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            conn.execute(
                "CREATE INDEX idx_embmeta_fact ON embedding_metadata (fact_id)"
            )
            conn.execute(
                "CREATE INDEX idx_embmeta_profile ON embedding_metadata (profile_id)"
            )
            conn.execute(
                "CREATE TABLE vector_row_map ("
                "fact_id TEXT NOT NULL PRIMARY KEY, profile_id TEXT NOT NULL, "
                "vec_rowid INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX idx_vector_row_map_profile "
                "ON vector_row_map (profile_id)"
            )

            first_probe: tuple[bytes, str] | None = None
            activated = 0
            for rowid, (fact_id, profile_id, embedding_json) in enumerate(
                stage.execute(
                    "SELECT fact_id, profile_id, embedding "
                    "FROM staged_embeddings ORDER BY fact_id"
                ),
                start=1,
            ):
                vector = json.loads(embedding_json)
                vec_bytes = np.asarray(vector, dtype=np.float32).tobytes()
                conn.execute(
                    "INSERT INTO fact_embeddings(rowid, profile_id, embedding) "
                    "VALUES (?, ?, ?)",
                    (rowid, profile_id, vec_bytes),
                )
                conn.execute(
                    "INSERT INTO embedding_metadata "
                    "(vec_rowid, fact_id, profile_id, model_name, dimension) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        rowid,
                        fact_id,
                        profile_id,
                        config.embedding.model_name,
                        config.embedding.dimension,
                    ),
                )
                conn.execute(
                    "INSERT INTO vector_row_map (fact_id, profile_id, vec_rowid) "
                    "VALUES (?, ?, ?)",
                    (fact_id, profile_id, rowid),
                )
                updated = conn.execute(
                    "UPDATE atomic_facts SET embedding = ? "
                    "WHERE fact_id = ? AND profile_id = ?",
                    (embedding_json, fact_id, profile_id),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(
                        f"canonical fact changed during migration: {fact_id}"
                    )
                if first_probe is None:
                    first_probe = (vec_bytes, profile_id)
                activated += 1

            counts = [
                int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "fact_embeddings",
                    "embedding_metadata",
                    "vector_row_map",
                )
            ]
            if activated != expected_count or any(c != expected_count for c in counts):
                raise RuntimeError(
                    f"vector activation incomplete: activated={activated}, "
                    f"projection_counts={counts}, expected={expected_count}"
                )
            if first_probe is not None:
                probe = conn.execute(
                    "SELECT rowid FROM fact_embeddings "
                    "WHERE embedding MATCH ? AND profile_id = ? AND k = 1",
                    (first_probe[0], first_probe[1]),
                ).fetchone()
                if probe is None:
                    raise RuntimeError("post-activation KNN probe returned no row")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _model_signature(config: SLMConfig) -> str:
    """Derive a deterministic signature from the active embedding config.

    V3.3.4: Only model_name + dimension matter. Provider (sentence-transformers
    vs ollama) doesn't change the embedding space when the model is the same.
    This prevents spurious re-indexing when switching Mode A ↔ B.
    """
    emb = config.embedding
    return f"{emb.model_name}::{emb.dimension}"


def _normalize_signature(signature: str) -> str:
    """Normalize a signature for equivalence comparison.

    v3.8.2 self-healing: the SAME embedding model has been recorded under
    different name strings across releases — notably the HuggingFace org
    prefix drifted (``nomic-ai/nomic-embed-text-v1.5`` vs the bare
    ``nomic-embed-text-v1.5``). A prefix-only difference does NOT change the
    embedding vector space, so it must not trigger a full multi-hour re-embed
    when a non-technical user upgrades. This collapses the model name to its
    basename (segment after the last ``/``) while keeping the ``::dimension``
    suffix — a genuine model change (different basename OR dimension) still
    differs and still triggers migration.
    """
    model, sep, dim = signature.partition("::")
    model = model.rsplit("/", 1)[-1].strip()
    return f"{model}{sep}{dim}" if sep else model


def _read_stored_signature(config_dir: Path) -> str:
    """Read the last-used embedding model signature from config.json."""
    config_path = config_dir / "config.json"
    if not config_path.exists():
        return _NO_MODEL
    try:
        data = json.loads(config_path.read_text())
        return data.get("embedding_signature", _NO_MODEL)
    except (json.JSONDecodeError, OSError):
        return _NO_MODEL


def _write_stored_signature(config_dir: Path, signature: str) -> None:
    """Persist the current embedding model signature to config.json."""
    config_path = config_dir / "config.json"
    data: dict[str, Any] = {}
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    data["embedding_signature"] = signature
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2))


def check_embedding_migration(config: SLMConfig) -> bool:
    """Check if embedding model changed since last run.

    Returns True if re-indexing is needed (model signature differs).
    Returns False if signatures match or this is the first run.
    """
    current_sig = _model_signature(config)
    stored_sig = _read_stored_signature(config.base_dir)

    if stored_sig == _NO_MODEL:
        # First run — store signature, no migration needed.
        _write_stored_signature(config.base_dir, current_sig)
        logger.info("Embedding signature initialized: %s", current_sig)
        return False

    if stored_sig == current_sig:
        return False

    # v3.8.2 self-healing: a prefix-only model-name drift (e.g. the nomic-ai/
    # org prefix appearing/disappearing between releases) is the SAME vector
    # space — absorb the transition by refreshing the stored signature to the
    # current form, with NO re-embed. This spares non-technical users a
    # multi-hour full re-index on a cosmetic upgrade.
    if _normalize_signature(stored_sig) == _normalize_signature(current_sig):
        _write_stored_signature(config.base_dir, current_sig)
        logger.info(
            "Embedding signature normalized (no re-embed): %s ~= %s",
            stored_sig,
            current_sig,
        )
        return False

    logger.warning(
        "Embedding model changed: %s -> %s. Re-indexing required.",
        stored_sig,
        current_sig,
    )
    return True


def run_embedding_migration(
    config: SLMConfig,
    db: Any,
    embedder: Any,
) -> int:
    """Stage and atomically activate embeddings for the current model.

    Embedding is performed in bounded batches into a temporary shadow store.
    Canonical rows are changed only after every target fact has a valid vector;
    the activation itself runs in one database transaction.  A failed batch,
    malformed vector set, or database write therefore leaves both the old
    embeddings and the old model signature active.
    """
    if embedder is None:
        logger.warning("No embedder available. Skipping re-indexing.")
        return 0

    current_sig = _model_signature(config)
    # Embedding configuration is database-wide. Rebuild every profile so one
    # vec0 table never contains vectors from mixed model spaces.
    rows = db.execute(
        "SELECT fact_id, profile_id, content FROM atomic_facts ORDER BY created_at",
    )
    facts = [
        (dict(r)["fact_id"], dict(r)["profile_id"], dict(r)["content"])
        for r in rows
    ]
    total = len(facts)

    if total == 0:
        _write_stored_signature(config.base_dir, current_sig)
        return 0

    logger.info(
        "Re-embedding %d facts with model %s (batch_size=%d)",
        total,
        current_sig,
        _REINDEX_BATCH_SIZE,
    )

    config.base_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="embedding-migration-",
            dir=config.base_dir,
        ) as stage_dir:
            stage_path = Path(stage_dir) / "shadow.sqlite3"
            with sqlite3.connect(stage_path) as stage:
                stage.execute(
                    "CREATE TABLE staged_embeddings ("
                    "fact_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, "
                    "content_hash TEXT NOT NULL, embedding TEXT NOT NULL)"
                )
                for i in range(0, total, _REINDEX_BATCH_SIZE):
                    batch = facts[i : i + _REINDEX_BATCH_SIZE]
                    texts = [content for _, _, content in batch]
                    fact_ids = [fid for fid, _, _ in batch]
                    profile_ids = [pid for _, pid, _ in batch]
                    vectors = list(embedder.embed_batch(texts))
                    if len(vectors) != len(batch):
                        raise ValueError(
                            "embedder returned "
                            f"{len(vectors)} vectors for {len(batch)} facts"
                        )
                    staged_rows: list[tuple[str, str, str, str]] = []
                    for fid, profile_id, content, vec in zip(
                        fact_ids, profile_ids, texts, vectors, strict=True
                    ):
                        if vec is None or len(vec) != config.embedding.dimension:
                            raise ValueError(
                                f"invalid embedding for fact {fid[:16]}: "
                                f"expected dimension {config.embedding.dimension}"
                            )
                        staged_rows.append(
                            (
                                fid,
                                profile_id,
                                hashlib.sha256(content.encode("utf-8")).hexdigest(),
                                json.dumps([float(value) for value in vec]),
                            )
                        )
                    stage.executemany(
                        "INSERT INTO staged_embeddings "
                        "(fact_id, profile_id, content_hash, embedding) "
                        "VALUES (?, ?, ?, ?)",
                        staged_rows,
                    )
                    stage.commit()

                staged_count = int(
                    stage.execute("SELECT COUNT(*) FROM staged_embeddings").fetchone()[0]
                )
                if staged_count != total:
                    raise ValueError(
                        f"shadow migration incomplete: {staged_count}/{total} facts"
                    )

                _activate_staged_vectors(config, db, stage_path, total)
    except Exception as exc:
        logger.error(
            "Embedding migration aborted; previous embedding space remains active: %s",
            exc,
        )
        return 0

    _write_stored_signature(config.base_dir, current_sig)
    logger.info(
        "Embedding migration complete: %d/%d facts re-embedded.",
        total,
        total,
    )
    return total


# ---------------------------------------------------------------------------
# Backfill: embed facts that were NEVER embedded (embedding IS NULL)
# ---------------------------------------------------------------------------


def _count_null_embeddings(
    db: Any,
    profile_id: str,
    all_profiles: bool,
) -> int:
    """Return count of atomic_facts rows with NULL embedding."""
    if all_profiles:
        rows = db.execute(
            "SELECT count(*) AS c FROM atomic_facts WHERE embedding IS NULL",
        )
    else:
        rows = db.execute(
            "SELECT count(*) AS c FROM atomic_facts WHERE embedding IS NULL AND profile_id = ?",
            (profile_id,),
        )
    return int(rows[0]["c"]) if rows else 0


def backfill_missing_embeddings(
    config: "SLMConfig",
    db: Any,
    embedder: Any,
    batch_size: int = _BACKFILL_BATCH_SIZE,
    limit: int | None = None,
    all_profiles: bool = False,
) -> dict[str, int]:
    """Embed atomic_facts rows whose ``embedding`` column is NULL.

    Unlike :func:`run_embedding_migration` (which re-embeds on model-signature
    change), this function handles facts that were *never* embedded — for
    example facts stored while the embedder was unavailable.

    Resumable and idempotent: re-running after a partial run only processes
    the remaining NULLs.  Fail-open per-fact: a single bad fact logs a warning
    and is skipped; the batch continues.

    Writes mirror :func:`run_embedding_migration` exactly:
    * ``atomic_facts.embedding`` ← ``json.dumps(vector)``
    * sqlite-vec + ``embedding_metadata`` ← one atomic projection pair

    Args:
        config: Active SLMConfig (provides profile_id, model name, dimension).
        db: DatabaseManager (or duck-compatible object with ``.execute()``).
        embedder: Object implementing ``embed_batch(texts) -> list[vec|None]``
            and (optionally) ``embed(text) -> vec|None``.  Pass ``None`` to
            make this a no-op (returns zero counts).
        batch_size: Facts per embed_batch() call.  Defaults to 50.
        limit: Maximum facts to embed in this call.  ``None`` means no cap —
            all NULL-embedding facts are processed.  Use a bounded limit for
            the maintenance self-healing path so each pass is quick.
        all_profiles: When ``True``, processes facts from every profile in the
            database.  When ``False`` (default), scopes to
            ``config.active_profile``.

    Returns:
        ``{"scanned": int, "embedded": int, "remaining_null": int}``

        *scanned*: total NULL-embedding facts found before applying *limit*.
        *embedded*: facts successfully written in this call.
        *remaining_null*: NULL count after the call (includes facts not yet
            reached because of *limit*).
    """
    profile_id = config.active_profile

    if embedder is None:
        logger.warning("backfill_missing_embeddings: no embedder available — skipping.")
        return {"scanned": 0, "embedded": 0, "remaining_null": 0}

    # ------------------------------------------------------------------
    # 1. Fetch all NULL-embedding facts (cheap query; only reads IDs + content)
    # ------------------------------------------------------------------
    if all_profiles:
        rows = db.execute(
            "SELECT fact_id, content, profile_id FROM atomic_facts "
            "WHERE embedding IS NULL ORDER BY created_at",
        )
    else:
        rows = db.execute(
            "SELECT fact_id, content, profile_id FROM atomic_facts "
            "WHERE embedding IS NULL AND profile_id = ? ORDER BY created_at",
            (profile_id,),
        )

    facts: list[tuple[str, str, str]] = [
        (dict(r)["fact_id"], dict(r)["content"], dict(r)["profile_id"]) for r in rows
    ]
    scanned = len(facts)

    if scanned == 0:
        return {"scanned": 0, "embedded": 0, "remaining_null": 0}

    # Apply call-level limit (resumability: next call picks up where this left off)
    if limit is not None:
        facts = facts[:limit]

    current_model = config.embedding.model_name
    current_dim = config.embedding.dimension
    embedded = 0
    vector_store = None
    try:
        from superlocalmemory.retrieval.vector_store import (
            VectorStore,
            VectorStoreConfig,
        )

        db_path = getattr(db, "db_path", None)
        if db_path is not None:
            vector_store = VectorStore(
                db_path,
                VectorStoreConfig(
                    dimension=current_dim,
                    model_name=current_model,
                ),
            )
    except Exception as exc:
        logger.debug("backfill: vector store unavailable: %s", exc)

    # ------------------------------------------------------------------
    # 2. Batch embed and write back
    # ------------------------------------------------------------------
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start : batch_start + batch_size]
        # Bound per-fact input so an oversized document doesn't monopolize the
        # shared embedding worker (starving foreground recall during self-heal).
        texts = [(content or "")[:_MAX_EMBED_CHARS] for _, content, _ in batch]
        fact_ids = [fid for fid, _, _ in batch]
        prof_ids = [pid for _, _, pid in batch]

        # Attempt batch embed; fall back to per-fact on batch failure. Mark the
        # whole inference burst as background work so the shared embedding
        # service yields between items if a recall begins after the daemon's
        # initial in-flight check.
        from superlocalmemory.core.recall_gate import background_work

        with background_work():
            try:
                vectors: list[Any] = embedder.embed_batch(texts)
            except Exception as exc:
                logger.warning(
                    "backfill: batch embed failed for facts %d-%d: %s — retrying per-fact.",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                vectors = []
                for text in texts:
                    try:
                        vec = embedder.embed(text)
                        vectors.append(vec)
                    except Exception as per_fact_exc:
                        logger.warning(
                            "backfill: per-fact embed failed for '%s...': %s",
                            text[:40],
                            per_fact_exc,
                        )
                        vectors.append(None)

        # Write each successfully-embedded fact back to the DB.
        for fid, vec, pid in zip(fact_ids, vectors, prof_ids):
            if vec is None:
                logger.warning("backfill: null vector for fact %s — skipping.", fid[:16])
                continue
            try:
                embedding_json = json.dumps(vec)
                # Metadata is not an independent record: it is the pointer to
                # a sqlite-vec row. Creating it before the vector payload leaves
                # semantic recall permanently blind while reporting success.
                # VectorStore owns the atomic pair and repairs legacy orphans.
                projection_written = False
                if vector_store is not None and getattr(vector_store, "available", False):
                    projection_written = vector_store.upsert(
                        fid,
                        pid,
                        vec,
                        model_name=current_model,
                    )
                    if not projection_written:
                        # Keep the canonical embedding NULL so the next
                        # bounded self-heal pass retries this fact.  The old
                        # order wrote JSON first, permanently removed the fact
                        # from the backfill query, and silently stranded recall
                        # without its sqlite-vec projection.
                        logger.warning(
                            "backfill: vector projection failed for fact %s; "
                            "leaving it pending for retry",
                            fid[:16],
                        )
                        continue

                try:
                    # Publish the canonical JSON only after the derived vector
                    # projection is durable.  When sqlite-vec is unavailable,
                    # this remains the supported JSON-only fallback path.
                    db.execute(
                        "UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                        (embedding_json, fid),
                    )
                except Exception:
                    if projection_written and vector_store is not None:
                        vector_store.delete(fid)
                    raise
                embedded += 1
                # Cooperative yield: release db._lock briefly so concurrent
                # user writes can acquire it between facts.  Without this,
                # the write-back loop holds db._lock in rapid succession
                # (Python RLock has no fairness guarantee), potentially
                # starving user POST /remember writes for seconds on large DBs.
                if _SELFHEAL_WRITE_DELAY_S > 0:
                    time.sleep(_SELFHEAL_WRITE_DELAY_S)
            except Exception as exc:
                logger.warning("backfill: failed to write fact %s: %s", fid[:16], exc)

    # ------------------------------------------------------------------
    # 3. Count remaining NULLs (accounts for the limit; tells caller how
    #    many passes remain before full convergence).
    # ------------------------------------------------------------------
    remaining = _count_null_embeddings(db, profile_id, all_profiles)

    logger.info(
        "Embedding backfill: %d/%d facts embedded, %d remaining NULL.",
        embedded,
        scanned,
        remaining,
    )
    return {"scanned": scanned, "embedded": embedded, "remaining_null": remaining}
