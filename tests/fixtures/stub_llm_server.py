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
        self._lifecycle_lock = threading.Lock()
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

        with self._lifecycle_lock:
            if self._server is not None or self._thread is not None:
                raise RuntimeError("StubLLMServer is already running")

            server = HTTPServer(("127.0.0.1", 0), _Handler)
            try:
                thread = threading.Thread(
                    target=server.serve_forever,
                    daemon=True,
                    name="stub-llm-server",
                )
                self._server = server
                self._thread = thread
                thread.start()
            except BaseException as start_error:
                # Thread construction/start can fail after the listening socket
                # has been bound. Detach state so a later _stop() remains safe.
                self._server = None
                self._thread = None
                try:
                    server.server_close()
                except BaseException as close_error:  # pragma: no cover
                    start_error.add_note(
                        "server_close also failed during startup cleanup: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                raise

    def _stop(self) -> None:
        # Claim ownership once. Concurrent or repeated stops become no-ops,
        # while the owning caller completes every cleanup stage below.
        with self._lifecycle_lock:
            server, self._server = self._server, None
            thread, self._thread = self._thread, None

        first_error: BaseException | None = None

        if server is not None and thread is not None and thread.is_alive():
            try:
                server.shutdown()
            except BaseException as exc:
                first_error = exc

        if thread is not None:
            try:
                thread.join(timeout=3)
                if thread.is_alive() and first_error is None:
                    first_error = RuntimeError(
                        "stub LLM server thread did not stop within 3 seconds"
                    )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        if server is not None:
            try:
                server.server_close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error
