# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A save is enriched even when the embedding worker's stdout carries bytes that
are not UTF-8, and a save that dead-lettered on that error retries cleanly.

The worker answers in JSON lines on its stdout. A model library that writes to
the same file descriptor (a warning in the console's legacy code page, a
progress line) puts raw bytes into that channel. The parent read the pipe as
strict UTF-8, so on the thread reader (Windows pipes) the next read raised
``'utf-8' codec can't decode byte 0x81 in position 2: invalid start byte`` out of
``embed()``; the materializer stored that text as the save's error ten times and
dead-lettered it. On POSIX the same bytes were swallowed as a "timeout", the
worker was killed and the save failed on its missing embedding.

These tests drive the real path: the engine's materializer, the real
``EmbeddingService`` spawning its worker with the production pipe settings, and
the real worker loop (``embedding_worker._worker_main``) with only the model
replaced by one that writes a non-UTF-8 line to file descriptor 1.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap

import pytest

from superlocalmemory.core import embeddings as E
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.engine_ingestion import (
    build_engine_ingestion_command,
    canonical_store,
    local_trusted_actor_id,
)
from superlocalmemory.core.ingestion_command import (
    _MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS,
    IngestionState,
)

CODEC_ERROR = "'utf-8' codec can't decode byte 0x81 in position 2: invalid start byte"
CONTENT = ("会議は金曜日です。Alice reviews the billing schema before the "
           "Postgres cutover, 田中さん signs off.")

# The real worker loop; only the model is replaced. Its encode() writes a line
# that is not UTF-8 straight to file descriptor 1, as a native library does.
_NOISY_WORKER = textwrap.dedent("""
    import os
    from superlocalmemory.core import embedding_worker as worker

    class _Model:
        def get_sentence_embedding_dimension(self):
            return 768

        def encode(self, texts, normalize_embeddings=True):
            os.write(1, b"ab\\x81cd\\n")
            return [[1.0] + [0.0] * 767 for _ in texts]

    worker._load_embedding_model = lambda name: (_Model(), "fake")
    worker._start_parent_watchdog = lambda: None
    worker._worker_main()
""")


class _NoFileno:
    """The worker's stdout as the reader sees a Windows pipe: no selectable fd,
    so ``_readline_with_timeout`` takes its thread reader."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def fileno(self):
        raise OSError("not selectable")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _use_thread_reader(monkeypatch) -> None:
    original = E.EmbeddingService._readline_with_timeout

    def reader(stream, timeout_seconds, *, cancelled=None):
        return original(_NoFileno(stream), timeout_seconds, cancelled=cancelled)

    monkeypatch.setattr(E.EmbeddingService, "_readline_with_timeout", staticmethod(reader))


@pytest.fixture
def noisy_worker(tmp_path, monkeypatch):
    """Spawn the noisy worker through the service's own Popen call and settings."""
    script = tmp_path / "noisy_worker.py"
    script.write_text(_NOISY_WORKER, encoding="utf-8")
    real_popen = subprocess.Popen

    def popen(args, *rest, **kwargs):
        if "superlocalmemory.core.embedding_worker" in list(args):
            args = [sys.executable, str(script)]
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(E.subprocess, "Popen", popen)
    # This test's worker only: its own lock and PID files, never another run's.
    monkeypatch.setattr(E, "_embedding_lock_file", lambda: tmp_path / "embed.lock")
    monkeypatch.setattr(E, "_embedding_pid_file", lambda: tmp_path / "embed.pid")
    monkeypatch.setattr(E.EmbeddingService, "_check_memory_pressure",
                        staticmethod(lambda worker_pid=None: True))
    services: list[E.EmbeddingService] = []

    def make() -> E.EmbeddingService:
        service = E.EmbeddingService(EmbeddingConfig())
        services.append(service)
        return service

    yield make
    for service in services:
        service.shutdown(timeout=1.0)


def _submit(engine) -> str:
    receipt = canonical_store(
        engine, CONTENT, source_type="python-api",
        trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=False, return_receipt=True,
    )
    return receipt.operation_id


def _stored_embeddings(engine, fact_ids) -> list:
    marks = ",".join("?" for _ in fact_ids)
    rows = engine._db.execute(
        f"SELECT embedding FROM atomic_facts WHERE fact_id IN ({marks})", tuple(fact_ids))
    return [dict(r)["embedding"] for r in rows]


@pytest.mark.parametrize("reader", ["selector", "thread"])
def test_save_completes_when_worker_stdout_has_non_utf8_bytes(
    engine_with_mock_deps, noisy_worker, monkeypatch, reader,
) -> None:
    engine = engine_with_mock_deps
    engine._embedder = noisy_worker()
    if reader == "thread":
        _use_thread_reader(monkeypatch)

    operation_id = _submit(engine)
    result = build_engine_ingestion_command(engine).materialize(operation_id)

    assert result.state is IngestionState.COMPLETE, result.last_error
    assert CODEC_ERROR not in result.last_error
    assert result.raw_content == CONTENT
    vectors = _stored_embeddings(engine, result.fact_ids)
    assert vectors and all(v is not None for v in vectors)


def test_embedding_worker_pipes_decode_tolerantly(noisy_worker, monkeypatch) -> None:
    """The parent never raises a decode error, whatever bytes the pipe carries."""
    seen: dict = {}
    real_popen = E.subprocess.Popen

    def recording(args, *rest, **kwargs):
        seen.update(kwargs)
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(E.subprocess, "Popen", recording)
    _use_thread_reader(monkeypatch)
    service = noisy_worker()
    service.embed("hello there")  # must not raise
    assert seen.get("encoding") == "utf-8"
    assert seen.get("errors") == "replace"


def test_dead_lettered_codec_failure_retries_and_clears(
    engine_with_mock_deps, noisy_worker, monkeypatch,
) -> None:
    from superlocalmemory.core.ops_remediation import resolve_operation

    engine = engine_with_mock_deps
    operation_id = _submit(engine)

    # The failure as it reached the store before the fix: embed() raised the
    # codec error out of the pipe, on every automatic attempt.
    def broken(_text):
        b"ab\x81cd".decode("utf-8")

    engine._embedder.embed.side_effect = broken
    command = build_engine_ingestion_command(engine)
    for _ in range(_MAX_AUTOMATIC_MATERIALIZATION_ATTEMPTS):
        failed = command.materialize(operation_id)
    assert failed.state is IngestionState.FAILED
    db_path = engine._config.db_path
    with sqlite3.connect(str(db_path)) as conn:
        errors = [r[0] for r in conn.execute(
            "SELECT error FROM dead_letter_operations WHERE original_op_id=?",
            (operation_id,))]
    assert errors == [CODEC_ERROR]

    # After the fix the same worker output no longer fails the save.
    engine._embedder = noisy_worker()
    _use_thread_reader(monkeypatch)
    outcome = resolve_operation(db_path, engine, operation_id, "retry")

    assert outcome["success"] is True, outcome
    done = command.repository.get(operation_id)
    assert done.state is IngestionState.COMPLETE, done.last_error
    assert done.raw_content == CONTENT
    with sqlite3.connect(str(db_path)) as conn:
        left = conn.execute(
            "SELECT COUNT(*) FROM dead_letter_operations WHERE original_op_id=?",
            (operation_id,)).fetchone()[0]
    assert left == 0
