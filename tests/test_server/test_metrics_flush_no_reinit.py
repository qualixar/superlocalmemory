"""Regression for issue #48: the metrics flush loop must NOT construct a new
CacheDB() on every 60s tick (which re-ran schema init, the corruption check, and
AES-key derivation, spamming 'Schema initialized' once per minute forever).
The CacheDB is now built once and reused.

The flush loop is a nested async closure inside the daemon lifespan, so this
guards the regression at the source level: no CacheDB() construction inside the
periodic loop body.
"""
import inspect

from superlocalmemory.server import unified_daemon


def _flush_loop_body() -> str:
    src = inspect.getsource(unified_daemon)
    idx = src.index("_metrics_flush_loop")
    # the loop body runs until the task is scheduled
    end = src.index("_optimize_flush_task = asyncio.create_task", idx)
    return src[idx:end]


def _strip_comments(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def test_metrics_flush_loop_does_not_construct_cachedb():
    body = _strip_comments(_flush_loop_body())
    assert "while True" in body, "test anchor moved — update the slice"
    assert "CacheDB()" not in body, (
        "Regression #48: metrics flush loop constructs a new CacheDB() per tick"
    )
