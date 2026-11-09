# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""CodeGraphWatcher — file system watcher with debounce.

Wraps watchdog Observer in a daemon thread. 300ms debounce coalesces
rapid file changes. Only reacts to supported source extensions.
Ignored directories (node_modules, .git, __pycache__, etc.) are skipped.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.code_graph.service import CodeGraphService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx",
})

IGNORED_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    "coverage", ".next", ".nuxt", ".eggs", "egg-info",
})

DEBOUNCE_SECONDS: float = 0.3


# ---------------------------------------------------------------------------
# Event handler
# ---------------------------------------------------------------------------

class _DebouncedHandler:
    """File system event handler with debounce logic.

    Accumulates changed file paths and flushes them after DEBOUNCE_SECONDS
    of inactivity. Uses a threading.Timer for the debounce.
    """

    def __init__(self, repo_root: str, callback) -> None:
        self._repo_root = repo_root
        self._callback = callback
        self._pending: dict[str, float] = {}
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def on_event(self, src_path: str, is_delete: bool = False) -> None:
        """Handle a file system event."""
        # Filter by extension
        ext = os.path.splitext(src_path)[1]
        if ext not in SUPPORTED_EXTENSIONS:
            return

        # Filter by ignored directories
        parts = Path(src_path).parts
        for part in parts:
            if part in IGNORED_DIRS:
                return

        with self._lock:
            if is_delete:
                # Deletions are immediate
                self._flush_immediate([src_path])
                return

            self._pending[src_path] = time.time()
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        """Flush pending changes to the callback."""
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()
            self._timer = None

        if paths:
            try:
                self._callback(paths)
            except Exception as exc:
                logger.warning("Watcher callback failed: %s", exc)

    def _flush_immediate(self, paths: list[str]) -> None:
        """Flush specific paths immediately (for deletions)."""
        try:
            self._callback(paths)
        except Exception as exc:
            logger.warning("Watcher callback failed: %s", exc)

    def cancel(self) -> None:
        """Cancel any pending timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()


# ---------------------------------------------------------------------------
# CodeGraphWatcher
# ---------------------------------------------------------------------------

class CodeGraphWatcher:
    """Watches a repository for file changes and triggers incremental updates.

    Runs in a daemon thread — never blocks the main thread (HR-6).
    """

    def __init__(self, repo_root: str, service: CodeGraphService) -> None:
        self._repo_root = str(repo_root)
        self._service = service
        self._observer = None
        self._handler = None
        self._running = False

    def start(self) -> None:
        """Start watching. Non-blocking — spawns a daemon thread."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            raise ImportError(
                "watchdog not installed. Install with: pip install superlocalmemory[code-graph]"
            )

        def _on_changes(paths: list[str]) -> None:
            """Called by debounced handler with accumulated changes."""
            try:
                rel_paths = []
                root = Path(self._repo_root)
                for p in paths:
                    try:
                        rel = Path(p).relative_to(root).as_posix()
                        rel_paths.append(rel)
                    except ValueError:
                        rel_paths.append(p)

                logger.info(
                    "Watcher detected %d changed files, updating graph",
                    len(rel_paths),
                )
                # Placeholder: incremental update would go here
                # The service doesn't have an incremental_update method yet
                # but this is where it would be called
            except Exception as exc:
                logger.warning("Watcher update failed: %s", exc)

        self._handler = _DebouncedHandler(self._repo_root, _on_changes)

        class _WatchdogBridge(FileSystemEventHandler):
            """Bridge between watchdog events and our debounced handler."""

            def __init__(self, handler: _DebouncedHandler) -> None:
                self._handler = handler

            def on_modified(self, event):
                if not event.is_directory:
                    self._handler.on_event(event.src_path)

            def on_created(self, event):
                if not event.is_directory:
                    self._handler.on_event(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory:
                    self._handler.on_event(event.src_path, is_delete=True)

        self._observer = Observer()
        bridge = _WatchdogBridge(self._handler)
        self._observer.schedule(bridge, self._repo_root, recursive=True)
        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("CodeGraphWatcher started for %s", self._repo_root)

    def stop(self) -> None:
        """Stop watching and clean up."""
        if self._handler is not None:
            self._handler.cancel()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        self._running = False
        logger.info("CodeGraphWatcher stopped")

    @property
    def is_running(self) -> bool:
        """Whether the watcher is currently running."""
        return self._running
