# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Recognise a retry that only differs because redaction was turned on."""

from __future__ import annotations

from typing import Any

from superlocalmemory.memory_core.save_path import same_after_redaction


def find_redacted_duplicate(db: Any, request: Any, pii_redaction: bool) -> Any | None:
    """The stored operation that owns ``request``'s key, when its raw text is the
    same as the request's prepared text; otherwise None.

    An operation saved before redaction was on keeps its raw text, so a retry of
    the same key now conflicts. When the two are the same text once prepared,
    the retry is a duplicate, not a failure.
    """
    from superlocalmemory.core.ingestion_command import IngestionOperationRepository

    if not pii_redaction:
        return None
    existing = IngestionOperationRepository(db).find_for_request(request)
    if existing is not None and same_after_redaction(
        existing.raw_content, request.content, pii_redaction,
    ):
        return existing
    return None
