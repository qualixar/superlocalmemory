# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Notices file changes in active folders so they are read within seconds, not minutes.

In-process, one observer per active source, no separate process and no token. Events are only
hints: they are collected for a quiet moment (the debounce), then handed over as the set of
changed relpaths. The scan that follows applies every rule; the 15-minute scan stays the truth.
When the operating system cannot watch (watchdog missing, inotify limit reached) the source
just keeps the 15-minute scan and its report says ``watch`` 0.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.sources.host import SourceHost

logger = logging.getLogger(__name__)

THREAD_NAME = "slm-source-watch"
DEBOUNCE_S = 2.0
#: More changed paths than this in one burst: ask for a full scan instead (``None``).
MAX_PENDING = 500

_watching: set[str] = set()
_watching_lock = threading.Lock()


def is_watching(source_id: str) -> bool:
    with _watching_lock:
        return source_id in _watching


def _default_observer() -> Any:
    from watchdog.observers import Observer  # raises ImportError when watchdog is missing

    return Observer()


class _Handler:
    """Receives watchdog events (``dispatch``) and passes the file paths on."""

    def __init__(self, root: Path, push: Callable[[list[str]], None]) -> None:
        self._root, self._push = root, push

    def dispatch(self, event: Any) -> None:
        if getattr(event, "is_directory", False):
            return  # the file events of a moved folder arrive on their own; the full scan covers the rest
        rels = [r for r in (self._rel(getattr(event, "src_path", "")),
                            self._rel(getattr(event, "dest_path", ""))) if r]
        if rels:
            self._push(rels)

    def _rel(self, raw: Any) -> str:
        if not raw:
            return ""
        try:
            return Path(os.fsdecode(raw)).relative_to(self._root).as_posix()
        except ValueError:
            return ""


class SourceWatcher:
    def __init__(self, host: SourceHost, on_change: Callable[[str, list[str] | None], None], *,
                 debounce_s: float = DEBOUNCE_S,
                 observer_factory: Callable[[], Any] | None = None) -> None:
        self._host = host
        self._on_change = on_change
        self._debounce_s = debounce_s
        self._factory = observer_factory or _default_observer
        self._cond = threading.Condition()
        self._observers: dict[str, Any] = {}
        self._pending: dict[str, set[str] | None] = {}  # None: too many, scan everything
        self._last: dict[str, float] = {}
        self._warned = False
        self._failed: set[str] = set()  # sources that could not be watched; not retried every pass
        self._closed = False
        self._thread: threading.Thread | None = None

    # -- observers ---------------------------------------------------------------------
    def sync(self, active: list[dict[str, Any]]) -> None:
        """Watch exactly the active ones among the given sources: start missing observers, stop the rest."""
        wanted = {} if self._closed else {s["source_id"]: s for s in active if s["state"] == "active"}
        self._failed &= set(wanted)
        for sid in [s for s in self._observers if s not in wanted]:
            self._drop(sid)
        for sid, source in wanted.items():
            if sid not in self._observers and sid not in self._failed:
                self._start(sid, source)

    def watching(self, source_id: str) -> bool:
        return source_id in self._observers

    def _start(self, sid: str, source: dict[str, Any]) -> None:
        observer = None
        try:
            observer = self._factory()
            handler = _Handler(Path(source["root_path"]), lambda rels, s=sid: self._push(s, rels))
            observer.schedule(handler, source["root_path"], recursive=True)
            observer.start()
        except (ImportError, OSError) as exc:
            self._give_up(sid, exc, observer)
            return
        self._observers[sid] = observer
        with _watching_lock:
            _watching.add(sid)
        self._ensure_thread()

    def _give_up(self, sid: str, exc: Exception, observer: Any) -> None:
        self._failed.add(sid)
        if not self._warned:  # once, whatever the number of folders
            self._warned = True
            logger.warning("folder watching is off (%s); folders are checked every 15 minutes",
                           type(exc).__name__)
        if observer is not None:
            self._halt(observer)

    @staticmethod
    def _halt(observer: Any) -> None:
        try:
            observer.stop()
            observer.join(2.0)
        except Exception:  # noqa: BLE001 - an observer that never started has nothing to join
            pass

    def _drop(self, sid: str) -> None:
        observer = self._observers.pop(sid, None)
        with _watching_lock:
            _watching.discard(sid)
        with self._cond:
            self._pending.pop(sid, None)
            self._last.pop(sid, None)
        if observer is not None:
            self._halt(observer)

    # -- debounce ----------------------------------------------------------------------
    def _push(self, sid: str, rels: list[str]) -> None:
        with self._cond:
            if sid not in self._observers:
                return
            bucket = self._pending.setdefault(sid, set())
            if bucket is not None:
                bucket.update(rels)
                if len(bucket) > MAX_PENDING:
                    self._pending[sid] = None
            self._last[sid] = time.monotonic()
            self._cond.notify_all()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
            self._thread.start()

    def _due(self) -> list[tuple[str, list[str] | None]]:
        now, out = time.monotonic(), []
        for sid in list(self._pending):
            if now - self._last.get(sid, now) >= self._debounce_s:
                bucket = self._pending.pop(sid)
                out.append((sid, None if bucket is None else sorted(bucket)))
        return out

    def _run(self) -> None:
        while True:
            with self._cond:
                if self._closed:
                    return
                due = self._due()
                if not due:
                    self._cond.wait(self._debounce_s / 4 if self._pending else 1.0)
                    continue
            for sid, rels in due:
                if self._closed:
                    return
                try:
                    self._on_change(sid, rels)
                except Exception as exc:  # noqa: BLE001 - one failed rescan must not end watching
                    logger.warning("a folder change could not be read (%s)", type(exc).__name__)

    # -- stop --------------------------------------------------------------------------
    def stop(self, timeout_s: float) -> bool:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for sid in list(self._observers):
            self._drop(sid)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout_s))
            return not thread.is_alive()
        return True


__all__ = ["DEBOUNCE_S", "SourceWatcher", "is_watching"]
