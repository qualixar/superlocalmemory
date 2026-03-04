#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
WebhookDispatcher — Delivers events via HTTP POST to configured webhook URLs.

Runs on a background thread so webhook delivery never blocks the main event flow.
Retries failed deliveries with exponential backoff (3 attempts).

Security:
    - Only allows http:// and https:// URLs
    - Validates URL format before dispatch
    - 10-second timeout per request
    - No private/internal IP blocking in v2.5 (added in v2.6 with trust enforcement)
"""

import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.parse
from queue import Queue, Empty
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger("superlocalmemory.webhooks")

# Configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds: 2, 4, 8
REQUEST_TIMEOUT = 10    # seconds
MAX_QUEUE_SIZE = 1000

# Optional: urllib3/requests for HTTP POST
try:
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


def _is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/internal IP address."""
    try:
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except (socket.gaierror, ValueError):
        return False  # DNS resolution failed — allow (might be valid external hostname)


class WebhookDispatcher:
    """
    Background webhook delivery with retry logic.

    Thread-safe. Enqueues webhook deliveries and processes them on a
    dedicated background thread. Failed deliveries are retried with
    exponential backoff.
    """

    _instances: Dict[str, "WebhookDispatcher"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get_instance(cls, name: str = "default") -> "WebhookDispatcher":
        """Get or create a singleton WebhookDispatcher."""
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = cls()
            return cls._instances[name]

    @classmethod
    def reset_instance(cls, name: Optional[str] = None) -> None:
        """Remove singleton(s). Used for testing."""
        with cls._instances_lock:
            if name is None:
                for inst in cls._instances.values():
                    inst.close()
                cls._instances.clear()
            elif name in cls._instances:
                cls._instances[name].close()
                del cls._instances[name]

    def __init__(self):
        self._queue: Queue = Queue(maxsize=MAX_QUEUE_SIZE)
        self._closed = False
        self._stats = {
            "dispatched": 0,
            "succeeded": 0,
            "failed": 0,
            "retries": 0,
        }
        self._stats_lock = threading.Lock()

        # Background worker thread
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="slm-webhook-worker",
            daemon=True,
        )
        self._worker.start()
        logger.info("WebhookDispatcher started")

    def dispatch(self, event: dict, webhook_url: str) -> None:
        """
        Enqueue a webhook delivery.

        Args:
            event: Event dict to send as JSON POST body
            webhook_url: URL to POST to

        Raises:
            ValueError: If webhook_url is invalid
            RuntimeError: If dispatcher is closed
        """
        if self._closed:
            raise RuntimeError("WebhookDispatcher is closed")

        if not webhook_url or not (webhook_url.startswith("http://") or webhook_url.startswith("https://")):
            raise ValueError(f"Invalid webhook URL: {webhook_url}")

        parsed = urllib.parse.urlparse(webhook_url)
        if parsed.hostname and _is_private_ip(parsed.hostname):
            raise ValueError(f"Webhook URL points to private/internal network: {webhook_url}")

        try:
            self._queue.put_nowait({
                "event": event,
                "url": webhook_url,
                "attempt": 0,
                "enqueued_at": datetime.now().isoformat(),
            })
            with self._stats_lock:
                self._stats["dispatched"] += 1
        except Exception:
            logger.warning("Webhook queue full, dropping event for %s", webhook_url)

    def _worker_loop(self):
        """Background worker: processes webhook deliveries sequentially."""
        while not self._closed:
            try:
                item = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if item is None:  # Shutdown sentinel
                self._queue.task_done()
                break

            self._deliver(item)
            self._queue.task_done()

    def _deliver(self, item: dict):
        """Attempt to deliver a webhook. Retry on failure."""
        event = item["event"]
        url = item["url"]
        attempt = item["attempt"]

        if not HTTP_AVAILABLE:
            logger.error("HTTP library not available, cannot deliver webhook to %s", url)
            with self._stats_lock:
                self._stats["failed"] += 1
            return

        try:
            payload = json.dumps({
                "event": event,
                "delivered_at": datetime.now().isoformat(),
                "attempt": attempt + 1,
                "source": "superlocalmemory",
                "version": "2.5.0",
            }).encode("utf-8")

            req = Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SuperLocalMemory/2.5.0",
                    "X-SLM-Event-Type": event.get("event_type", "unknown"),
                },
                method="POST",
            )

            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                status = resp.status
                if 200 <= status < 300:
                    with self._stats_lock:
                        self._stats["succeeded"] += 1
                    logger.debug("Webhook delivered: url=%s, status=%d", url, status)
                    return
                else:
                    raise HTTPError(url, status, f"HTTP {status}", {}, None)

        except Exception as e:
            logger.warning("Webhook delivery failed (attempt %d/%d): url=%s, error=%s",
                          attempt + 1, MAX_RETRIES, url, e)

            if attempt + 1 < MAX_RETRIES:
                # Retry with exponential backoff
                backoff = RETRY_BACKOFF_BASE ** (attempt + 1)
                time.sleep(backoff)
                with self._stats_lock:
                    self._stats["retries"] += 1
                item["attempt"] = attempt + 1
                self._deliver(item)  # Recursive retry
            else:
                with self._stats_lock:
                    self._stats["failed"] += 1
                logger.error("Webhook permanently failed after %d attempts: url=%s", MAX_RETRIES, url)

    def get_stats(self) -> dict:
        """Get webhook delivery statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:
        """Shut down the dispatcher. Drains remaining items."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # Shutdown sentinel
        if self._worker.is_alive():
            self._worker.join(timeout=5)
        logger.info("WebhookDispatcher closed: stats=%s", self._stats)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
