# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The recall worker and reranker pipes survive bytes that are not UTF-8.

Both workers answer in JSON lines on stdout, as the embedding worker does, and
had the same flaw: a library writing to file descriptor 1 put raw bytes in the
reply pipe, and the parent decoded it strictly. On the thread reader (Windows
pipes) that raised ``UnicodeDecodeError`` out of the request; on POSIX the
reply was lost as a "timeout" and the worker was killed.

Each test runs the real worker loop with ``_respond`` wrapped so that a line of
non-UTF-8 bytes reaches file descriptor 1 before every reply, and the real
parent class spawning it with its production pipe settings.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from superlocalmemory.core import worker_pool as wp
from superlocalmemory.retrieval import reranker as rr

# The real class, bound at import: the suite's session fixture replaces the
# module attribute with a mock so engines never start a model.
from superlocalmemory.retrieval.reranker import CrossEncoderReranker

_NOISY = textwrap.dedent("""
    import os
    from superlocalmemory.core import {module} as worker

    _real = worker._respond

    def _noisy(data):
        os.write(1, b"ab\\x81cd\\n")
        _real(data)

    worker._respond = _noisy
    worker._start_parent_watchdog = lambda: None
    {extra}
    worker._worker_main()
""")


class _NoFileno:
    """A pipe as the reader sees one on Windows: no selectable descriptor."""

    def __init__(self, stream) -> None:
        self._stream = stream

    def fileno(self):
        raise OSError("not selectable")

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _script(tmp_path, module: str, extra: str = "") -> str:
    path = tmp_path / f"noisy_{module}.py"
    path.write_text(_NOISY.format(module=module, extra=extra), encoding="utf-8")
    return str(path)


def _thread_reader(monkeypatch, owner, attribute: str = "_readline_with_timeout") -> None:
    original = getattr(owner, attribute)

    def reader(stream, timeout_seconds, *args, **kwargs):
        return original(_NoFileno(stream), timeout_seconds, *args, **kwargs)

    monkeypatch.setattr(owner, attribute, staticmethod(reader))


@pytest.mark.parametrize("reader", ["selector", "thread"])
def test_recall_worker_reply_survives_non_utf8_stdout(tmp_path, monkeypatch, reader) -> None:
    script = _script(tmp_path, "recall_worker",
                     extra="worker._handle_status = lambda: {'ok': True, 'state': 'fake'}")
    real_popen = subprocess.Popen

    def popen(args, *rest, **kwargs):
        if "superlocalmemory.core.recall_worker" in list(args):
            args = [sys.executable, script]
        return real_popen(args, *rest, **kwargs)

    monkeypatch.setattr(wp.subprocess, "Popen", popen)
    if reader == "thread":
        _thread_reader(monkeypatch, wp.WorkerPool)
    pool = wp.WorkerPool()
    try:
        reply = pool.status()
    finally:
        pool.shutdown()
    assert reply == {"ok": True, "state": "fake"}


@pytest.mark.parametrize("reader", ["selector", "thread"])
def test_reranker_reply_survives_non_utf8_stdout(tmp_path, monkeypatch, reader) -> None:
    script = _script(tmp_path, "reranker_worker")
    monkeypatch.setattr(CrossEncoderReranker, "_worker_argv",
                        lambda self: [sys.executable, script])
    monkeypatch.setattr(rr, "_is_reranker_worker_alive", lambda: False)
    monkeypatch.setattr(rr, "_reranker_pid_file", lambda: tmp_path / "reranker.pid")
    monkeypatch.setattr(rr, "_memory_pressed", lambda: False)
    if reader == "thread":
        _thread_reader(monkeypatch, CrossEncoderReranker)
    reranker = CrossEncoderReranker()
    try:
        reply = reranker._send_request({"cmd": "ping"}, timeout=20)
    finally:
        reranker.shutdown(timeout=1.0)
    assert isinstance(reply, dict) and reply.get("ok") is True, reply


def test_recall_and_reranker_pipes_decode_tolerantly(monkeypatch) -> None:
    seen: list[dict] = []

    class _Proc:
        pid = 1
        stdin = stdout = stderr = None

        def poll(self):
            return None

    def recording(args, *rest, **kwargs):
        seen.append(kwargs)
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", recording)
    wp.WorkerPool()._ensure_worker()
    CrossEncoderReranker._spawn_process(object.__new__(CrossEncoderReranker))
    assert [(k.get("encoding"), k.get("errors")) for k in seen] == [("utf-8", "replace")] * 2


def test_outbound_body_that_is_not_utf8_is_blocked_not_a_decode_error() -> None:
    from superlocalmemory.core.outbound_http import OutboundBlocked, outbound_content

    with pytest.raises(OutboundBlocked):
        outbound_content(b"ab\x81cd", "https://api.example.com/v1/embeddings")
    # This machine is not screened: the bytes go as they are.
    assert outbound_content(b"ab\x81cd", "http://127.0.0.1:11434/api") == b"ab\x81cd"


# -- the channel's edge cases -------------------------------------------------


class _Stream:
    """A stand-in for ``sys.stdout`` / ``sys.stderr`` backed by a real descriptor."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd

    def flush(self) -> None:
        pass


def test_a_worker_without_stdout_fails_with_a_clear_error(monkeypatch) -> None:
    from superlocalmemory.core.worker_stdio import claim_protocol_channel

    monkeypatch.setattr(sys, "stdout", None)  # as under pythonw
    with pytest.raises(RuntimeError, match="no stdout"):
        claim_protocol_channel()


def test_the_duplicated_descriptor_is_closed_when_the_redirect_fails(monkeypatch) -> None:
    import os
    from unittest.mock import patch

    from superlocalmemory.core.worker_stdio import claim_protocol_channel

    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    duplicated: list[int] = []
    real_dup = os.dup

    def dup(fd):
        new = real_dup(fd)
        duplicated.append(new)
        return new

    stdout = _Stream(out_w)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", _Stream(err_w))
    try:
        # Patched only around the call: the test runner itself uses dup2.
        with patch.object(os, "dup", dup), patch.object(os, "dup2", side_effect=OSError("no")):
            channel = claim_protocol_channel()
        assert channel is stdout  # nothing moved: the plain stream
        assert len(duplicated) == 1
        with pytest.raises(OSError):
            os.fstat(duplicated[0])  # closed, not leaked
    finally:
        for fd in (out_r, out_w, err_r, err_w):
            os.close(fd)


@pytest.mark.parametrize("module", ["embedding_worker", "reranker_worker"])
def test_sigterm_exit_flushes_the_reply_channel(monkeypatch, module) -> None:
    import importlib
    import os

    worker = importlib.import_module(f"superlocalmemory.core.{module}")
    flushed: list[str] = []

    class _Channel:
        def flush(self) -> None:
            flushed.append("reply")

    class _Left(Exception):
        pass

    def fake_exit(code):
        raise _Left

    monkeypatch.setattr(worker, "_PROTOCOL", _Channel())
    monkeypatch.setattr(os, "_exit", fake_exit)
    with pytest.raises(_Left):
        worker._exit_now()
    assert flushed == ["reply"]
