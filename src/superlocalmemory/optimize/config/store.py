"""ConfigStore — reads, writes, and hot-reloads optimize.json.

SINGLE SOURCE OF TRUTH for the daemon, UI, CLI, and MCP tools.
UI and CLI write via save(); daemon reads via get() which is always current.

Hot-reload: a background watchdog thread polls optimize.json every 2 seconds.
On mtime change, it parses and validates; if valid, atomically swaps the
active config. If invalid, keeps the old config (fail-open).

Thread safety: self._lock (threading.RLock) protects self._current_config
and self._version. get() acquires a read-lock; save() acquires a write-lock.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.infra.data_root import DynamicStatePath
from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG
from superlocalmemory.optimize.config.schema import OptimizeConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = DynamicStatePath("optimize.json")
_POLL_INTERVAL_SECONDS: float = 2.0


class ConfigStore:
    """Manages optimize.json with hot-reload capability."""

    def __init__(
        self,
        config_path: Path | None = None,
        poll_interval: float = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._config_path = Path(
            config_path if config_path is not None else _DEFAULT_CONFIG_PATH
        )
        self._poll_interval_seconds = float(poll_interval)
        self._lock = threading.RLock()
        self._change_callbacks: list[Callable[[OptimizeConfig], None]] = []
        self._stop_event = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

        self._saved_by_self: bool = False
        self._current_config: OptimizeConfig = DEFAULT_OPTIMIZE_CONFIG
        self._last_mtime: float = 0.0
        self._version: int = DEFAULT_OPTIMIZE_CONFIG.config_version

        # Load initial state from disk if available.
        try:
            if self._config_path.exists():
                self._current_config = self._load_from_disk()
                try:
                    self._last_mtime = self._config_path.stat().st_mtime
                except OSError:
                    self._last_mtime = 0.0
                self._version = self._current_config.config_version
        except (json.JSONDecodeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                "ConfigStore: failed to parse %s — using defaults: %s",
                self._config_path, exc,
            )

    def get(self) -> OptimizeConfig:
        """Return the current active config (thread-safe, no I/O).

        INTERFACE-CONTRACT v2.2 §2 — canonical accessor.
        """
        with self._lock:
            return self._current_config

    def save(self, config: OptimizeConfig) -> None:
        """Write config to optimize.json with version bump.

        Fires registered change callbacks immediately after a successful write
        (outside the lock) so a UI/CLI save reaches the live proxy without
        waiting for the 2s watchdog poll. The watchdog skips this write via
        ``_saved_by_self`` so callbacks fire exactly once.
        """
        config.validate()
        with self._lock:
            new_version = self._version + 1
            new_cfg = OptimizeConfig.from_dict(
                {**config.as_dict(), "config_version": new_version}
            )
            new_cfg.validate()
            self._saved_by_self = True
            try:
                self._atomic_write(new_cfg.as_dict())
                try:
                    self._last_mtime = self._config_path.stat().st_mtime
                except OSError:
                    self._last_mtime = 0.0
                self._current_config = new_cfg
                self._version = new_version
            finally:
                self._saved_by_self = False
            callbacks = list(self._change_callbacks)
        # Fire callbacks OUTSIDE the lock — a callback may rebuild proxy hooks
        # or call back into get(); keeping them off the lock avoids contention.
        for cb in callbacks:
            try:
                cb(new_cfg)
            except Exception as exc:
                logger.warning("ConfigStore save callback error: %s", exc)

    def start_watchdog(self) -> None:
        """Start the background hot-reload watchdog thread.

        Idempotent — safe to call multiple times.
        """
        with self._lock:
            if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
                return
            self._stop_event.clear()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="slm-optimize-config-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        """Signal the watchdog thread to stop and join it (timeout=5s)."""
        self._stop_event.set()
        t = self._watchdog_thread
        if t is not None:
            t.join(timeout=5.0)
        with self._lock:
            self._watchdog_thread = None

    def version(self) -> int:
        with self._lock:
            return self._version

    def register_change_callback(
        self, callback: Callable[[OptimizeConfig], None]
    ) -> None:
        with self._lock:
            self._change_callbacks.append(callback)

    # ---- private ----

    def _load_from_disk(self) -> OptimizeConfig:
        raw = self._config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("optimize.json root must be an object")
        return OptimizeConfig.from_dict(data)

    def _atomic_write(self, data: dict[str, Any]) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._config_path.with_suffix(self._config_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._config_path)
        try:
            os.chmod(self._config_path, 0o600)
        except OSError:
            pass

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            # Sleep with cancellation
            if self._stop_event.wait(timeout=self._poll_interval_seconds):
                return
            try:
                try:
                    mtime = self._config_path.stat().st_mtime
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    logger.warning("ConfigStore watchdog stat error: %s", exc)
                    continue

                with self._lock:
                    if mtime == self._last_mtime:
                        continue
                    if self._saved_by_self:
                        self._last_mtime = mtime
                        self._saved_by_self = False
                        continue

                # Parse + validate OUTSIDE the lock to avoid contention.
                try:
                    new_config = self._load_from_disk()
                except (json.JSONDecodeError, ValueError, TypeError, OSError) as exc:
                    logger.warning(
                        "ConfigStore hot-reload PARSE ERROR — keeping previous config: %s",
                        exc,
                    )
                    continue

                with self._lock:
                    # Re-check after lock: another save may have completed.
                    if mtime == self._last_mtime:
                        continue
                    self._current_config = new_config
                    self._last_mtime = mtime
                    self._version += 1
                new_version = self._version

                # Snapshot callbacks under lock, iterate outside.
                with self._lock:
                    callbacks = list(self._change_callbacks)
                for cb in callbacks:
                    try:
                        cb(new_config)
                    except Exception as exc:
                        logger.warning("ConfigStore callback error: %s", exc)

                logger.info(
                    "ConfigStore hot-reloaded optimize.json (version %d)",
                    new_version,
                )
            except Exception as exc:
                logger.critical("ConfigStore watchdog thread crashed: %s", exc, exc_info=True)
