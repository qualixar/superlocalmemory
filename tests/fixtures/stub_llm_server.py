# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | WP-15 coverage tests

"""Ephemeral stub LLM HTTP server for backbone.py integration tests.

Listens on 127.0.0.1:0 (OS-assigned port). Handles:
- POST /v1/chat/completions  — OpenAI-compatible
- POST /api/chat             — Ollama native

Records all received requests. Thread-safe. Torn down via finalizer.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import NamedTuple


class RecordedRequest(NamedTuple):
    method: str
    path: str
    headers: dict[str, str]
    body: dict


class StubLLMServer:
    """Ephemeral local HTTP server that records requests and returns stubs.

    Usage::

        with StubLLMServer(reply_text="EXTRACTED") as stub:
            # stub.url -> "http://127.0.0.1:<port>"
            # stub.requests -> list[RecordedRequest]
    """

    def __init__(self, reply_text: str = "stub-reply") -> None:
        self._reply_text = reply_text
        self._requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> "StubLLMServer":
        self._start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop()

    # -- public API ------------------------------------------------------------

    @property
    def url(self) -> str:
        assert self._server is not None, "Server not started"
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return list(self._requests)

    def clear(self) -> None:
        with self._lock:
            self._requests.clear()

    # -- internals -------------------------------------------------------------

    def _start(self) -> None:
        stub = self  # captured by handler class

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a) -> None:
                pass  # suppress request logging during tests

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    body = {}

                recorded = RecordedRequest(
                    method="POST",
                    path=self.path,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=body,
                )
                with stub._lock:
                    stub._requests.append(recorded)

                # Build response based on path
                if self.path.rstrip("/").endswith("/api/chat"):
                    # Ollama native format
                    payload = {
                        "message": {"content": stub._reply_text},
                        "done": True,
                    }
                else:
                    # OpenAI-compatible format
                    payload = {
                        "choices": [
                            {"message": {"content": stub._reply_text}}
                        ]
                    }

                body_bytes = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body_bytes)))
                self.end_headers()
                self.wfile.write(body_bytes)

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="stub-llm-server",
        )
        self._thread.start()

    def _stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
