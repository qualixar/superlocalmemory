"""Nested trusted metadata (a picture's source record) survives the write queue and is stored."""

from __future__ import annotations

from superlocalmemory.core.ingestion_command import IngestionOperationRepository
from superlocalmemory.memory_core.submit import SaveRequest, submit_memory
from superlocalmemory.memory_core import ContentOrigin

from .test_write_paths_redact import (
    M018_ingestion_operations, M032_write_coordinator_admission, M033_projection_transactions,
    M034_obligation_integrity, M042_correction_case_ledger,
)

SOURCE = {"type": "media", "media_id": "m1", "origin": "tool"}


def test_nested_trusted_metadata_is_saved_through_the_daemon_writer(engine_with_mock_deps) -> None:
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime

    engine = engine_with_mock_deps
    with engine._db.raw_connection() as conn:
        for migration in (M018_ingestion_operations, M032_write_coordinator_admission,
                          M033_projection_transactions, M034_obligation_integrity, M042_correction_case_ledger):
            migration.apply(conn)
    runtime = CanonicalRememberRuntime.for_engine(engine)
    runtime.start()
    try:
        request = SaveRequest(
            segments=(("A grey cat sleeping on a sofa in the afternoon light.", ContentOrigin.USER_TEXT),),
            profile_id=engine._profile_id, source_type="media",
            trusted_actor_id=local_trusted_actor_id("http-media"),
            trusted_metadata={"_slm_source": SOURCE}, idempotency_key="nested-1")
        receipt = submit_memory(runtime, request, config=engine._config)
        assert receipt.status in ("stored", "queryable", "complete", "accepted")
    finally:
        runtime.stop()
    ops = IngestionOperationRepository(engine._db).list_operations()
    assert [dict(o.metadata.get("_slm_source") or {}) for o in ops] == [SOURCE]
