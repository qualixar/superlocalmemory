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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backfill constants
# ---------------------------------------------------------------------------

#: Default batch size for backfill_missing_embeddings.
_BACKFILL_BATCH_SIZE = 50

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
            stored_sig, current_sig,
        )
        return False

    logger.warning(
        "Embedding model changed: %s -> %s. Re-indexing required.",
        stored_sig, current_sig,
    )
    return True


def run_embedding_migration(
    config: SLMConfig,
    db: Any,
    embedder: Any,
) -> int:
    """Re-embed all facts with the current model. Returns count re-embedded.

    Processes facts in batches to avoid memory spikes. Updates the
    embedding_metadata table and vector store for each fact.

    This is idempotent — can be interrupted and resumed safely.
    """
    if embedder is None:
        logger.warning("No embedder available. Skipping re-indexing.")
        return 0

    current_sig = _model_signature(config)
    profile_id = config.active_profile

    # Get all fact IDs that need re-embedding (all facts for the profile).
    rows = db.execute(
        "SELECT fact_id, content FROM atomic_facts "
        "WHERE profile_id = ? ORDER BY created_at",
        (profile_id,),
    )
    facts = [(dict(r)["fact_id"], dict(r)["content"]) for r in rows]
    total = len(facts)

    if total == 0:
        _write_stored_signature(config.base_dir, current_sig)
        return 0

    logger.info(
        "Re-embedding %d facts with model %s (batch_size=%d)",
        total, current_sig, _REINDEX_BATCH_SIZE,
    )

    reindexed = 0
    for i in range(0, total, _REINDEX_BATCH_SIZE):
        batch = facts[i : i + _REINDEX_BATCH_SIZE]
        texts = [content for _, content in batch]
        fact_ids = [fid for fid, _ in batch]

        try:
            vectors = embedder.embed_batch(texts)
        except Exception as exc:
            logger.error(
                "Re-embedding batch %d-%d failed: %s. Stopping migration.",
                i, i + len(batch), exc,
            )
            break

        for j, (fid, vec) in enumerate(zip(fact_ids, vectors)):
            if vec is None:
                continue
            # Update embedding in the database (embedding column on atomic_facts).
            try:
                embedding_json = json.dumps(vec)
                db.execute(
                    "UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                    (embedding_json, fid),
                )
                # Update embedding_metadata with new model name.
                db.execute(
                    "UPDATE embedding_metadata SET model_name = ? "
                    "WHERE fact_id = ?",
                    (config.embedding.model_name, fid),
                )
                reindexed += 1
            except Exception as exc:
                logger.warning(
                    "Failed to update embedding for fact %s: %s",
                    fid[:16], exc,
                )

    # Update stored signature after successful migration.
    _write_stored_signature(config.base_dir, current_sig)
    logger.info(
        "Embedding migration complete: %d/%d facts re-embedded.",
        reindexed, total,
    )
    return reindexed


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
            "SELECT count(*) AS c FROM atomic_facts "
            "WHERE embedding IS NULL AND profile_id = ?",
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
    * ``embedding_metadata`` ← upserted row with current model name + dimension

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
        logger.warning(
            "backfill_missing_embeddings: no embedder available — skipping."
        )
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
        (dict(r)["fact_id"], dict(r)["content"], dict(r)["profile_id"])
        for r in rows
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

        # Attempt batch embed; fall back to per-fact on batch failure.
        try:
            vectors: list[Any] = embedder.embed_batch(texts)
        except Exception as exc:
            logger.warning(
                "backfill: batch embed failed for facts %d-%d: %s — "
                "retrying per-fact.",
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
                logger.warning(
                    "backfill: null vector for fact %s — skipping.", fid[:16]
                )
                continue
            try:
                embedding_json = json.dumps(vec)
                # Mirror run_embedding_migration's write path exactly.
                db.execute(
                    "UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                    (embedding_json, fid),
                )
                # Upsert embedding_metadata.  NULL-embedding facts have no row
                # here yet, so we INSERT; if a row somehow exists, update it.
                db.execute(
                    "INSERT INTO embedding_metadata"
                    "    (fact_id, profile_id, model_name, dimension)"
                    " VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(fact_id) DO UPDATE SET"
                    "    model_name = excluded.model_name",
                    (fid, pid, current_model, current_dim),
                )
                embedded += 1
            except Exception as exc:
                logger.warning(
                    "backfill: failed to write fact %s: %s", fid[:16], exc
                )

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
