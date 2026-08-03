"""F-68 regression: the canonical write chokepoint must scrub secrets.

The canonical HTTP ``/remember`` path routes through ``CanonicalRememberRuntime``
into ``build_immediate_admission_handler.write_queryable``. Before this fix, that
writer persisted ``request.content`` verbatim, so a credential sent to ``/remember``
reached the durable store unscrubbed even though the ``canonical_store()`` Python/CLI
path scrubbed. This pins the invariant that ``write_queryable`` scrubs
unconditionally, so BOTH ingress paths share the same secret-scrub guarantee
before anything is written to ``memories`` or ``atomic_facts``.
"""

from __future__ import annotations

_AWS_KEY = "AKIAIOSFODNN7EXAMPLE"


def test_write_queryable_scrubs_secret_before_durable_store(tmp_path) -> None:
    from superlocalmemory.core.engine_ingestion import build_immediate_admission_handler
    from superlocalmemory.core.ingestion_command import IngestionRequest
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute(
        "INSERT INTO profiles(profile_id, name, description) VALUES (?, ?, ?)",
        ("p1", "p1", "test profile"),
    )

    writer = build_immediate_admission_handler(db, profile_id="p1")
    request = IngestionRequest(
        content=f"Please store my AWS access key {_AWS_KEY} for the deploy pipeline.",
        profile_id="p1",
        source_type="http-remember",
        idempotency_key="op-f68",
        trusted_actor_id="local-capability:test",
    )

    fact_ids = writer(request, "op-f68")
    assert fact_ids, "write_queryable should persist a queryable fact"

    fact_rows = db.execute(
        "SELECT content FROM atomic_facts WHERE fact_id = ?", (fact_ids[0],)
    )
    memory_rows = db.execute(
        "SELECT content FROM memories WHERE profile_id = ?", ("p1",)
    )
    assert fact_rows and memory_rows

    fact_content = str(fact_rows[0]["content"])
    memory_content = str(memory_rows[0]["content"])

    # The raw credential must never reach the durable queryable representation.
    assert _AWS_KEY not in fact_content, "raw credential leaked into atomic_facts"
    assert _AWS_KEY not in memory_content, "raw credential leaked into memories"
    # And the scrub must be observable (redaction marker present).
    assert "[REDACTED:" in memory_content, "expected a redaction marker in stored memory"
