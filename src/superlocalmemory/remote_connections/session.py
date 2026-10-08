"""Bounded companion request execution against the existing canonical runtime."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from superlocalmemory.remote_connections.codec import (
    MAX_RESPONSE_BYTES,
    FrameError,
    compact,
    decode_frame,
    encode_frame,
)
from superlocalmemory.remote_connections.credentials import ConnectorCredential

#: Longest a relayed call may wait for this laptop. Matches the gateway's budget;
#: a correct answer from a busy database is not a failure.
RELAY_DEADLINE_MS = 25000
#: deadlineAt is stamped by the relay's clock. Tolerate a laptop clock this far
#: behind it, but never wait longer than RELAY_DEADLINE_MS.
CLOCK_SKEW_TOLERANCE_MS = 5000


@dataclass(frozen=True)
class OriginResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class RelaySession:
    def __init__(
        self,
        credential: ConnectorCredential,
        *,
        exchange: Callable[[dict, ConnectorCredential], Awaitable[OriginResponse]],
        send: Callable[[str], Awaitable[None]],
        close: Callable[[str], None],
        clock: Callable[[], float] = time.time,
        param_headers: tuple[str, ...] = (),
    ):
        decode_frame(
            '{"v":1,"kind":"cancel","id":"validation","generation":1}', param_headers=param_headers
        )
        self._credential, self._exchange, self._send, self._close = (
            credential,
            exchange,
            send,
            close,
        )
        self._clock, self._params = clock, tuple(param_headers)
        self._generation: int | None = None
        self._stopped = False
        self._operations: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def ready(self) -> bool:
        return not self._stopped and self._generation is not None

    async def stop(self) -> None:
        self._stopped = True
        self._operations.clear()
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_idle(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _fail(self, code: str = "connector_protocol_error") -> None:
        await self.stop()
        try:
            self._close(code)
        except Exception:
            pass  # An observer cannot prevent request cancellation.

    async def receive(self, text: str) -> None:
        if self._stopped:
            return
        if self._credential.expires_at_ms <= self._clock() * 1000:
            await self._fail("authorization_required")
            return
        if self._generation is None:
            try:
                if not isinstance(text, str) or len(text) > 1024:
                    raise ValueError("invalid_ready")
                ready = json.loads(text)
                if (
                    not isinstance(ready, dict)
                    or set(ready) != {"v", "kind", "generation"}
                    or type(ready["v"]) is not int
                    or ready["v"] != 1
                    or ready["kind"] != "ready"
                    or type(ready["generation"]) is not int
                    or not self._credential.generation <= ready["generation"] <= 2**53 - 1
                    or compact(ready) != text
                ):
                    raise ValueError("invalid_ready")
                self._generation = ready["generation"]
            except (ValueError, TypeError, RecursionError):
                await self._fail()
            return
        try:
            frame = decode_frame(text, param_headers=self._params)
        except FrameError:
            await self._fail()
            return
        if frame["generation"] != self._generation or frame["kind"] == "response":
            await self._fail()
            return
        if frame["kind"] == "cancel":
            task = self._operations.pop(frame["id"], None)
            if task is not None:
                task.cancel()
            return
        duration = frame["deadlineAt"] - self._clock() * 1000
        if duration <= 0:
            await self._reply(frame, 504, "relay_timeout")
            return
        if duration > RELAY_DEADLINE_MS + CLOCK_SKEW_TOLERANCE_MS:
            await self._reply(frame, 400, "invalid_deadline")
            return
        duration = min(duration, RELAY_DEADLINE_MS)
        if frame["id"] in self._operations:
            await self._fail()
            return
        if len(self._operations) >= 8:
            await self._reply(frame, 429, "connector_busy")
            return
        task = asyncio.create_task(self._perform(frame, duration / 1000), name="slm-remote-request")
        self._operations[frame["id"]] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _current(self, identifier: str) -> bool:
        return not self._stopped and self._operations.get(identifier) is asyncio.current_task()

    async def _perform(self, frame: dict, timeout: float) -> None:
        try:
            response = await asyncio.wait_for(self._exchange(frame, self._credential), timeout)
            if not self._current(frame["id"]):
                return
            if frame["deadlineAt"] <= self._clock() * 1000:
                await self._reply(frame, 504, "origin_timeout")
                return
            if self._credential.expires_at_ms <= self._clock() * 1000:
                await self._fail("authorization_required")
                return
            if (
                not isinstance(response, OriginResponse)
                or not isinstance(response.body, bytes)
                or len(response.body) > MAX_RESPONSE_BYTES
                or (response.status in {204, 205, 304} and response.body)
            ):
                raise FrameError("invalid_origin_response")
            wire = {
                "v": 1,
                "kind": "response",
                "id": frame["id"],
                "generation": frame["generation"],
                "status": response.status,
                "headers": [list(pair) for pair in response.headers],
                "bodyBase64": base64.b64encode(response.body).decode("ascii"),
            }
            await self._send(encode_frame(wire))
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if self._current(frame["id"]):
                await self._reply(frame, 504, "origin_timeout")
        except Exception:
            if self._current(frame["id"]):
                await self._reply(frame, 502, "origin_unavailable")
        finally:
            if self._operations.get(frame["id"]) is asyncio.current_task():
                self._operations.pop(frame["id"], None)

    async def _reply(self, frame: dict, status: int, code: str) -> None:
        if self._stopped:
            return
        wire = {
            "v": 1,
            "kind": "response",
            "id": frame["id"],
            "generation": frame["generation"],
            "status": status,
            "headers": [["content-type", "application/json"]],
            "bodyBase64": base64.b64encode(compact({"error": code}).encode()).decode("ascii"),
        }
        try:
            await self._send(encode_frame(wire))
        except Exception:
            await self._fail()
