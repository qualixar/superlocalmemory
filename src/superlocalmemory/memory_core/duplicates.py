# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Recognise a retry that only differs because redaction was turned on."""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from superlocalmemory.memory_core.save_path import (
    ContentOrigin,
    prepare_for_save,
    prepare_metadata,
)


def _as_prepared(existing: Any) -> Any:
    """The stored operation as it would have been saved with redaction on."""
    from superlocalmemory.core.metadata_guard import strip_reserved_metadata

    text = prepare_for_save(
        existing.raw_content, origin=ContentOrigin.USER_TEXT, pii_redaction=True,
    ).text
    metadata, _ = prepare_metadata(
        strip_reserved_metadata(existing.metadata), pii_redaction=True,
    )
    return dataclasses.replace(
        existing, raw_content=text, metadata=metadata,
        source_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def find_redacted_duplicate(db: Any, request: Any, pii_redaction: bool) -> Any | None:
    """The stored operation that is ``request`` saved before redaction was on.

    An operation saved with redaction off keeps its raw text and metadata, so a
    retry of the same key conflicts once redaction is on. It is a duplicate only
    when the stored operation, prepared the same way, equals the request on every
    field (the comparison the repository itself uses); anything else is a real
    conflict and returns None so the caller re-raises it.
    """
    from superlocalmemory.core.ingestion_command import (
        IdempotencyConflict,
        IngestionOperationRepository,
    )

    if not pii_redaction:
        return None
    repository = IngestionOperationRepository(db)
    existing = repository.find_for_request(request)
    if existing is None:
        return None
    try:
        repository._assert_same_request(_as_prepared(existing), request)
    except IdempotencyConflict:
        return None
    return existing
