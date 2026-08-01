"""Pre-admission scrub policy that removes secret material from content.

A single ordered stage that redacts secrets BEFORE content can reach any durable
or queryable representation (atomic facts, memory records, ingestion receipts,
journal, exports, backups, mesh). Unlike PII redaction — an opt-in operator
policy — secret scrubbing is unconditional: a credential persisted verbatim at
rest is always incorrect.

The detector is the ``security_primitives.redact_secrets`` redactor; this module
wraps it in an immutable result so callers can log the fact of redaction without
ever handling the secret value.
"""
from __future__ import annotations

from dataclasses import dataclass

from superlocalmemory.core.security_primitives import redact_secrets


@dataclass(frozen=True, slots=True)
class ScrubResult:
    """Outcome of a pre-admission scrub. Never carries the original secret."""

    content: str
    redacted: bool


def scrub_secrets_for_ingest(content: str) -> ScrubResult:
    """Redact secret material from ``content`` before durable admission.

    Returns a :class:`ScrubResult` whose ``content`` is safe to persist. When
    nothing matched, ``content`` is returned unchanged and ``redacted`` is False,
    so the caller can keep the original object and skip a log line.
    """
    if not content:
        return ScrubResult(content=content, redacted=False)
    scrubbed = redact_secrets(content)
    return ScrubResult(content=scrubbed, redacted=scrubbed != content)
