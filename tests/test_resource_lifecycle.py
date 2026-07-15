"""Regression gates for owned SQLite handles and worker subprocess pipes."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.embeddings import EmbeddingService
from superlocalmemory.retrieval.reranker import CrossEncoderReranker


class _CursorConnection:
    """Mimic sqlite3's context manager: exit a txn, but do not close."""

    def __init__(self, *, interrupt: bool = False) -> None:
        self.closed = False
        self._interrupt = interrupt

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def execute(self, *_args, **_kwargs):
        if self._interrupt:
            raise KeyboardInterrupt
        return self

    def fetchall(self) -> list:
        return []

    def fetchone(self):
        return None

    def close(self) -> None:
        self.closed = True


def test_stop_hook_closes_owned_connection_on_cancellation(monkeypatch) -> None:
    from superlocalmemory.hooks import stop_outcome_hook as hook

    conn = _CursorConnection(interrupt=True)
    monkeypatch.setattr(
        hook, "read_stdin_json", lambda: {"session_id": "session-1"},
    )
    monkeypatch.setattr(hook, "open_memory_db", lambda: conn)

    with pytest.raises(KeyboardInterrupt):
        hook._inner_main()

    assert conn.closed is True


def test_rehash_lookup_closes_owned_connection(monkeypatch) -> None:
    from superlocalmemory.hooks import user_prompt_rehash_hook as hook

    conn = _CursorConnection()
    monkeypatch.setattr(hook, "open_memory_db", lambda: conn)

    assert hook._current_latest_outcome_id("session-1") is None
    assert conn.closed is True


def test_rehash_main_closes_owned_connection(monkeypatch, tmp_path: Path) -> None:
    from superlocalmemory.hooks import user_prompt_rehash_hook as hook

    conn = _CursorConnection()
    monkeypatch.setattr(
        hook,
        "read_stdin_json",
        lambda: {"session_id": "session-1", "prompt": "Where is the plan?"},
    )
    monkeypatch.setattr(hook, "session_state_file", lambda _session_id: tmp_path)
    monkeypatch.setattr(hook, "load_session_state", lambda _session_id: {})
    monkeypatch.setattr(hook, "save_session_state", lambda *_args: None)
    monkeypatch.setattr(hook, "open_memory_db", lambda: conn)

    assert hook._inner_main() == "no_rehash"
    assert conn.closed is True


class _BrokenPipe:
    def __init__(self) -> None:
        self.close_calls = 0

    def write(self, _payload: str) -> None:
        raise BrokenPipeError("child already exited")

    def flush(self) -> None:
        raise BrokenPipeError("child already exited")

    def close(self) -> None:
        self.close_calls += 1


class _ExitedProcess:
    def __init__(self) -> None:
        self.stdin = _BrokenPipe()
        self.stdout = _BrokenPipe()
        self.stderr = None
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        return 0

    def kill(self) -> None:
        self.kill_calls += 1


@pytest.mark.parametrize("worker_kind", ["embedding", "reranker"])
def test_worker_shutdown_closes_dead_child_pipes_idempotently(worker_kind: str) -> None:
    if worker_kind == "embedding":
        owner = EmbeddingService.__new__(EmbeddingService)
        owner._http_client = None
    else:
        owner = CrossEncoderReranker.__new__(CrossEncoderReranker)

    proc = _ExitedProcess()
    owner._idle_timer = None
    owner._worker_proc = proc
    owner._worker_ready = True

    owner._kill_worker()
    owner._kill_worker()

    assert owner._worker_proc is None
    assert proc.kill_calls == 0
    assert proc.stdin.close_calls == 1
    assert proc.stdout.close_calls == 1
