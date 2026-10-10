# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Every engine is built on the embedding space the store actually holds.

Until 4.1.21 engine start-up compared config.json's embedding model with the
one the vectors were made with and, when they differed, re-embedded the whole
store right there, while every recall and remember waited. That is gone: the
re-embed is a background job (core/embedding_reindex.py).

What start-up must still guarantee is that it never pairs one model's queries
with another model's vectors, and never opens the vector table at a dimension
it was not built with (VectorStore drops and recreates the table then). So
when config.json names a different model than the live space -- a hand edit,
``slm mode``, a mode change on the dashboard -- this engine is built on the LIVE
model, and the requested one is recorded on the config as the pending target
the daemon's re-index runner turns into a job. Nothing here writes to the
store; on a store's very first start it records the signature, as before.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_PRESET_LOCK = threading.Lock()
_PRESET: dict[str, Any] = {}

PENDING_ATTR = "_pending_embedding_target"


def _preset_key(db_path: Any, signature: str) -> str:
    return f"{db_path}|{signature}"


def offer_embedder(db_path: Any, signature: str, embedder: Any) -> None:
    """Hand a warm embedder to the next engine built for this store and model."""
    with _PRESET_LOCK:
        _PRESET[_preset_key(db_path, signature)] = embedder


def take_preset_embedder(config: Any) -> Any | None:
    from superlocalmemory.storage.embedding_spaces import signature_of

    with _PRESET_LOCK:
        return _PRESET.pop(_preset_key(config.db_path, signature_of(config.embedding)), None)


def _rows(db: Any, sql: str, params: tuple = ()) -> list[Any]:
    try:
        return list(db.execute(sql, params))
    except Exception as exc:  # a store without the table is a store without the record
        if "no such table" in str(exc).lower():
            return []
        raise


def _vectors_model(db: Any) -> tuple[str, int] | None:
    """The model and width the stored vectors were actually made with."""
    rows = _rows(db, "SELECT model_name, dimension FROM embedding_metadata LIMIT 2000")
    if not rows:
        return None
    (model, dim), _n = Counter((str(r[0]), int(r[1])) for r in rows).most_common(1)[0]
    return (model, dim) if model else None


def _infer_live(desired: Any, live_sig: str) -> Any:
    """A store upgraded mid-change: only the live model's name and width are known."""
    from dataclasses import replace

    from superlocalmemory.core import model_catalog

    model, _sep, dim = live_sig.partition("::")
    entry = model_catalog.find(model)
    if entry is not None and entry.provider in ("ollama", "sentence-transformers", "slm-media"):
        provider = entry.provider
    elif "/" in model:
        provider = "sentence-transformers"
    elif desired.provider == "ollama" or ":" in model:
        provider = "ollama"
    else:
        provider = desired.provider
    same_endpoint = provider == desired.provider
    return replace(
        desired, model_name=model, dimension=int(dim or desired.dimension),
        provider=provider,
        ollama_model=model if provider == "ollama" else desired.ollama_model,
        api_endpoint=desired.api_endpoint if same_endpoint else "",
        api_key=desired.api_key if same_endpoint else "",
    )


def live_signature(config: Any, db: Any) -> tuple[str | None, dict | None]:
    """``(signature, key-free config)`` of the live space; config None when unknown."""
    from superlocalmemory.storage.embedding_migrator import _read_stored_signature

    rows = _rows(db, "SELECT live_signature, live_config FROM embedding_space WHERE id = 1")
    if rows:
        return str(rows[0][0]), json.loads(rows[0][1])
    stored = _read_stored_signature(config.base_dir)
    if stored:
        return stored, None
    seen = _vectors_model(db)
    return (f"{seen[0]}::{seen[1]}", None) if seen else (None, None)


def bind_live_space(config: Any, db: Any) -> None:
    """Point ``config.embedding`` at the live space before any embedder is built."""
    try:
        _bind_live_space(config, db)
    finally:  # the stored vectors' width, for the decoder's truncated-write check
        from superlocalmemory.storage.embedding_codec import set_expected_dimension

        set_expected_dimension(config.embedding.dimension)


def _bind_live_space(config: Any, db: Any) -> None:
    from superlocalmemory.storage.embedding_migrator import _write_stored_signature
    from superlocalmemory.storage.embedding_spaces import (
        config_from_public,
        public_config,
        same_space,
        signature_of,
    )

    desired = config.embedding
    desired_sig = signature_of(desired)
    try:
        live_sig, live_public = live_signature(config, db)
    except Exception as exc:  # unreadable record: serve as configured, say so
        logger.error("embedding space record unreadable (%s); using config.json", exc)
        return
    if live_sig is None:
        try:  # first start of an empty store: what it will hold is what is configured
            _write_stored_signature(config.base_dir, desired_sig)
        except OSError as exc:
            logger.warning("could not record the embedding signature: %s", exc)
        return
    if same_space(desired_sig, live_sig):
        return
    if not _rows(db, "SELECT 1 FROM atomic_facts LIMIT 1"):
        # Nothing stored, nothing to re-embed: the configured model is the space.
        _write_stored_signature(config.base_dir, desired_sig)
        if live_public is not None:
            db.execute("UPDATE embedding_space SET live_signature = ?, live_config = ? "
                       "WHERE id = 1",
                       (desired_sig, json.dumps(public_config(desired), sort_keys=True)))
        return
    if live_public is not None:
        same_endpoint = (live_public.get("provider") == desired.provider
                         and live_public.get("api_endpoint", "") == desired.api_endpoint)
        live = config_from_public(live_public, desired.api_key if same_endpoint else "")
    else:
        live = _infer_live(desired, live_sig)
    setattr(config, PENDING_ATTR, desired)
    config.embedding = live
    logger.warning(
        "config names embedding model %s but this store's vectors are %s: serving the "
        "stored model; the daemon re-indexes in the background "
        "(see: slm embedder status)", desired_sig, live_sig)
    try:
        from superlocalmemory.core import embedding_reindex

        embedding_reindex.notify_pending(config)
    except Exception as exc:  # the runner adopts it on its next pass anyway
        logger.debug("pending embedder notice not delivered: %s", exc)


__all__ = ["PENDING_ATTR", "bind_live_space", "live_signature", "offer_embedder",
           "take_preset_embedder"]
