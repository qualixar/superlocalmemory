# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Base adapter class for all ingestion adapters.

All adapters inherit this. Enforces stateless, safe, cross-platform operation:
  - Clean shutdown via stop event + parent PID watchdog
  - Rate limiting per hour
  - Batch throttling with interruptible delays
  - Retry on 429 responses
  - Structured logging

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger("superlocalmemory.ingestion")


@dataclass
class AdapterConfig:
    daemon_port: int = 8765
    batch_size: int = 50
    batch_delay_sec: float = 5.0
    rate_limit_per_hour: int = 100


class IngestItem(NamedTuple):
    content: str
    dedup_key: str
    metadata: dict = {}


class BaseAdapter:
    """All ingestion adapters inherit this class.

    Provides: run loop, rate limiting, retry, shutdown, parent watchdog.
    Subclasses implement: fetch_items(), wait_for_next_cycle(), source_type.
    """

    source_type: str = "unknown"

    def __init__(self, config: AdapterConfig | None = None):
        self.config = config or AdapterConfig()
        self.daemon_url = f"http://127.0.0.1:{self.config.daemon_port}"
        self._items_this_hour = 0
        self._hour_start = time.time()
        self._stop_event = threading.Event()
        self._parent_pid = os.getppid()
        self._total_ingested = 0

    def run(self) -> None:
        """Main adapter loop. Subclasses don't override this."""
        self._setup_signals()
        logger.info("%s adapter started (PID %d)", self.source_type, os.getpid())

        while not self._stop_event.is_set():
            # Parent watchdog: exit if daemon died
            try:
                import psutil
                if not psutil.pid_exists(self._parent_pid):
                    logger.info("Parent daemon died, adapter exiting")
                    break
            except ImportError:
                try:
                    os.kill(self._parent_pid, 0)
                except (ProcessLookupError, PermissionError):
                    logger.info("Parent daemon died, adapter exiting")
                    break

            try:
                items = self.fetch_items()
            except Exception as exc:
                logger.warning("fetch_items failed: %s", exc)
                self._stop_event.wait(30)
                continue

            if not items:
                self.wait_for_next_cycle()
                continue

            # Process in batches
            for i in range(0, len(items), self.config.batch_size):
                if self._stop_event.is_set():
                    break
                batch = items[i:i + self.config.batch_size]
                for item in batch:
                    if self._stop_event.is_set():
                        break
                    if self._rate_limited():
                        logger.info("Rate limit reached (%d/hr), waiting",
                                    self.config.rate_limit_per_hour)
                        self._stop_event.wait(60)
                        continue
                    self._ingest(item)
                # Interruptible batch delay
                self._stop_event.wait(self.config.batch_delay_sec)

            self.wait_for_next_cycle()

        logger.info("%s adapter stopped (total ingested: %d)",
                    self.source_type, self._total_ingested)

    def stop(self) -> None:
        self._stop_event.set()

    # -- Subclass interface --

    def fetch_items(self) -> list[IngestItem]:
        """Fetch items from the source. Subclass MUST implement."""
        raise NotImplementedError

    def wait_for_next_cycle(self) -> None:
        """Wait before next fetch cycle. Default: 5 min interruptible."""
        self._stop_event.wait(300)

    # -- Internal --

    def _ingest(self, item: IngestItem) -> bool:
        """POST to daemon /ingest endpoint. Returns True on success."""
        payload = {
            "content": item.content,
            "source_type": self.source_type,
            "dedup_key": item.dedup_key,
            "metadata": item.metadata if item.metadata else {},
        }
        for attempt in range(2):
            try:
                from superlocalmemory.cli.daemon import daemon_request

                data = daemon_request("POST", "/ingest", payload)
                if data and data.get("ingested"):
                    self._items_this_hour += 1
                    self._total_ingested += 1
                    return True
                return False  # Already ingested (dedup)
            except Exception as exc:
                if "429" in str(exc) and attempt == 0:
                    logger.info("Daemon returned 429, backing off 5s")
                    self._stop_event.wait(5)
                    continue
                logger.debug("Ingest failed: %s", exc)
                return False
        return False

    def _rate_limited(self) -> bool:
        if time.time() - self._hour_start > 3600:
            self._items_this_hour = 0
            self._hour_start = time.time()
        return self._items_this_hour >= self.config.rate_limit_per_hour

    def _setup_signals(self) -> None:
        """Set up clean shutdown on SIGTERM."""
        def _handler(sig, frame):
            self.stop()
        signal.signal(signal.SIGTERM, _handler)
        if sys.platform != "win32":
            signal.signal(signal.SIGINT, _handler)
