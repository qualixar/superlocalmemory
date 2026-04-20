# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""WebhookDispatcher -- background HTTP POST delivery for memory events.

Runs on a daemon thread so webhook delivery never blocks the main event
flow.  Failed deliveries are retried with exponential back-off (up to
``MAX_RETRIES`` attempts).

Security:
    * Only ``http://`` and ``https://`` URLs are accepted.
    * Private / loopback IPs are rejected.
    * 10-second timeout per outgoing request.
"""

import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Dict, Optional

logger = logging.getLogger("superlocalmemory.webhooks")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2        # seconds: 2, 4, 8
REQUEST_TIMEOUT = 10           # seconds
MAX_QUEUE_SIZE = 1000
def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("superlocalmemory")
    except Exception:
        return "3.0.0"


VERSION = _get_version()

# stdlib HTTP -- always available
from urllib.request import Request, urlopen  # noqa: E402
from urllib.error import HTTPError, URLError  # noqa: E402


def _is_private_ip(hostname: str) -> bool:
    """Return ``True`` if *hostname* resolves to a private / loopback IP."""
    try:
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except (socket.gaierror, ValueError):
        return False


class WebhookDispatcher:
    """Background webhook delivery with retry logic.

    Thread-safe.  Enqueues deliveries and processes them on a dedicated
    daemon thread.
    """

    _instances: Dict[str, "WebhookDispatcher"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, name: str = "default") -> "WebhookDispatcher":
        """Get or create a named singleton."""
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = cls()
            return cls._instances[name]

    @classmethod
    def reset_instance(cls, name: Optional[str] = None) -> None:
        """Remove singleton(s).  Primarily for testing."""
        with cls._instances_lock:
            if name is None:
                for inst in cls._instances.values():
                    inst.close()
                cls._instances.clear()
            elif name in cls._instances:
                cls._instances[name].close()
                del cls._instances[name]

    def __init__(self) -> None:
        self._queue: Queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self._closed = False
        self._stats = {
            "dispatched": 0,
            "succeeded": 0,
            "failed": 0,
            "retries": 0,
        }
        self._stats_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="slm-webhook-worker",
            daemon=True,
        )
        self._worker.start()
        logger.info("WebhookDispatcher started")

    # ----- public API -----

    def dispatch(self, event: dict, webhook_url: str) -> None:
        """Enqueue a webhook delivery.

        Raises:
            ValueError: If *webhook_url* is invalid or private.
            RuntimeError: If the dispatcher is closed.
        """
        if self._closed:
            raise RuntimeError("WebhookDispatcher is closed")

        if not webhook_url or not (
            webhook_url.startswith("http://") or webhook_url.startswith("https://")
        ):
            raise ValueError(f"Invalid webhook URL: {webhook_url}")

        parsed = urllib.parse.urlparse(webhook_url)
        if parsed.hostname and _is_private_ip(parsed.hostname):
            raise ValueError(
                f"Webhook URL points to private/internal network: {webhook_url}"
            )

        try:
            self._queue.put_nowait(
                {
                    "event": event,
                    "url": webhook_url,
                    "attempt": 0,
                    "enqueued_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            with self._stats_lock:
                self._stats["dispatched"] += 1
        except Exception:
            logger.warning("Webhook queue full, dropping event for %s", webhook_url)

    def get_stats(self) -> dict:
        """Return delivery statistics snapshot."""
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:
        """Shut down the dispatcher, draining remaining items."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # sentinel
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        logger.info("WebhookDispatcher closed: stats=%s", self._stats)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ----- internal -----

    def _worker_loop(self) -> None:
        """Background loop: dequeue and deliver."""
        while not self._closed:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:  # shutdown sentinel
                self._queue.task_done()
                break

            self._deliver(item)
            self._queue.task_done()

    def _deliver(self, item: dict) -> None:
        """Attempt delivery with exponential-backoff retry."""
        event = item["event"]
        url = item["url"]
        attempt = item["attempt"]

        try:
            payload = json.dumps(
                {
                    "event": event,
                    "delivered_at": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt + 1,
                    "source": "superlocalmemory",
                    "version": VERSION,
                }
            ).encode("utf-8")

            req = Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"SuperLocalMemory/{VERSION}",
                    "X-SLM-Event-Type": event.get("event_type", "unknown"),
                },
                method="POST",
            )

            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                status = resp.status
                if 200 <= status < 300:
                    with self._stats_lock:
                        self._stats["succeeded"] += 1
                    logger.debug("Webhook delivered: url=%s status=%d", url, status)
                    return
                raise HTTPError(url, status, f"HTTP {status}", {}, None)

        except Exception as exc:
            logger.warning(
                "Webhook delivery failed (attempt %d/%d): url=%s error=%s",
                attempt + 1,
                MAX_RETRIES,
                url,
                exc,
            )

            if attempt + 1 < MAX_RETRIES:
                backoff = RETRY_BACKOFF_BASE ** (attempt + 1)
                time.sleep(backoff)
                with self._stats_lock:
                    self._stats["retries"] += 1
                item["attempt"] = attempt + 1
                self._deliver(item)
            else:
                with self._stats_lock:
                    self._stats["failed"] += 1
                logger.error(
                    "Webhook permanently failed after %d attempts: url=%s",
                    MAX_RETRIES,
                    url,
                )
