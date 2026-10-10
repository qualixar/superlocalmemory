"""What the picture worker writes to stderr reaches the daemon log, bounded, and leaves no thread behind."""

from __future__ import annotations

import io
import logging
import textwrap
import threading
import time

import pytest

from superlocalmemory.runtimes import worker_client, worker_log

LOGGER = "superlocalmemory.runtimes.worker_client"


_HEADER = "import json, sys\n"
_LOOP = textwrap.dedent("""
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("cmd") == "quit":
            break
        print(json.dumps({"id": req["id"], "ok": True, "dim": 4, "vectors": [[0.0] * 4]}), flush=True)
""")


def _script(tmp_path, body: str):
    path = tmp_path / "noisy_worker.py"
    path.write_text(_HEADER + textwrap.dedent(body) + "\n" + _LOOP)
    return path


@pytest.fixture()
def noisy(tmp_path, monkeypatch, client_factory):
    def make(body: str):
        monkeypatch.setattr(worker_client, "WORKER_PATH", _script(tmp_path, body))
        return client_factory()

    return make


def _drain_threads():
    return [t for t in threading.enumerate() if t.name == "media-worker-stderr" and t.is_alive()]


def _wait_until(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _worker_lines(caplog):
    return [r for r in caplog.records if r.getMessage().startswith("image worker: ")]


def test_a_stderr_line_reaches_the_log_at_warning_with_the_prefix(noisy, caplog):
    c = noisy('print("CUDA out of memory", file=sys.stderr, flush=True)')
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c.embed_texts(["a"], prompt="Document")
        assert _wait_until(lambda: _worker_lines(caplog))
    record = _worker_lines(caplog)[0]
    assert record.levelno == logging.WARNING and record.getMessage() == "image worker: CUDA out of memory"


def test_a_long_line_is_cut_and_its_rest_is_not_logged_as_more_lines(noisy, caplog):
    c = noisy('sys.stderr.write("x" * 100000 + "\\n"); sys.stderr.write("after\\n"); sys.stderr.flush()')
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c.embed_texts(["a"], prompt="Document")
        assert _wait_until(lambda: len(_worker_lines(caplog)) >= 2)
    first, second = (r.getMessage() for r in _worker_lines(caplog)[:2])
    assert len(first) <= len("image worker: ") + 500
    assert second == "image worker: after"


def test_a_flood_is_rate_limited(noisy, caplog):
    c = noisy('for i in range(500):\n    print("line", i, file=sys.stderr)\nsys.stderr.flush()')
    with caplog.at_level(logging.INFO, logger=LOGGER):
        c.embed_texts(["a"], prompt="Document")
        c.stop()
    warnings = [r for r in _worker_lines(caplog) if r.levelno == logging.WARNING]
    assert len(warnings) == 50
    assert any("450" in r.getMessage() and "not logged" in r.getMessage() for r in _worker_lines(caplog))


def test_undecodable_bytes_do_not_stop_the_drain(noisy, caplog):
    c = noisy('sys.stderr.buffer.write(b"bad \\xff\\xfe bytes\\n"); sys.stderr.buffer.write(b"still here\\n"); sys.stderr.flush()')
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c.embed_texts(["a"], prompt="Document")
        assert _wait_until(lambda: any("still here" in r.getMessage() for r in caplog.records))


def test_no_drain_thread_is_left_after_stop(noisy):
    c = noisy('print("hello", file=sys.stderr, flush=True)')
    c.embed_texts(["a"], prompt="Document")
    assert _drain_threads()
    c.stop()
    assert _wait_until(lambda: not _drain_threads())


def test_no_drain_thread_is_left_after_a_crash_and_restart_or_the_idle_stop(noisy):
    c = noisy('print("hello", file=sys.stderr, flush=True)')
    c.embed_texts(["a"], prompt="Document")
    import psutil
    psutil.Process(c.pid).kill()
    c.embed_texts(["b"], prompt="Document")  # restarts: the old drain must have ended
    assert _wait_until(lambda: len(_drain_threads()) == 1)
    c.stop()
    assert _wait_until(lambda: not _drain_threads())


def test_the_pipe_is_closed_when_the_worker_is_stopped(noisy):
    c = noisy('print("hello", file=sys.stderr, flush=True)')
    c.embed_texts(["a"], prompt="Document")
    proc = c._proc
    c.stop()
    assert proc.stderr.closed


# -- the drain on its own: the rate window ------------------------------------------

def test_the_window_rolls_over_and_reports_what_it_held_back(caplog):
    ticks = iter([0.0] * 60 + [61.0])  # one reading per line: 60 lines in the first minute, then one a minute later
    lines = "".join(f"l{i}\n" for i in range(60)) + "late\n"
    drain = worker_log.StderrDrain(io.StringIO(lines), clock=lambda: next(ticks), max_lines=50, window_s=60.0)
    with caplog.at_level(logging.INFO, logger=worker_log.logger.name):
        drain.run()
    texts = [r.getMessage() for r in caplog.records]
    assert sum(t.startswith("image worker: l") and t != "image worker: late" for t in texts) == 50
    assert any("10 more" in t and "not logged" in t for t in texts)
    assert texts.index("image worker: late") > next(i for i, t in enumerate(texts) if "not logged" in t)


def test_the_worker_starts_with_library_progress_bars_off(monkeypatch):
    from superlocalmemory.runtimes import worker_client

    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    client = worker_client.MediaWorkerClient.__new__(worker_client.MediaWorkerClient)
    client.rss_limit_mb = 0
    env = worker_client.MediaWorkerClient._worker_env(client)
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1" and env["TQDM_DISABLE"] == "1"
    assert env["TRANSFORMERS_VERBOSITY"] == "error"
