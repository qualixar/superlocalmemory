"""Outbound companion in SLM's existing Python runtime; no user Node setup."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncContextManager, Awaitable, Callable

from superlocalmemory.remote_connections.codec import MAX_FRAME_BYTES
from superlocalmemory.remote_connections.credentials import ConnectorCredential, _validate
from superlocalmemory.remote_connections.session import OriginResponse, RelaySession

ENDPOINT = "wss://connect.superlocalmemory.com/connector"
logger = logging.getLogger(__name__)


def _dial(endpoint: str, token: str, device_key: str = ""):
    from superlocalmemory.remote_connections.proof import DeviceSigner

    signer = DeviceSigner(device_key)
    proof = signer.proof("GET", "https://connect.superlocalmemory.com/connector", token=token)
    from websockets.asyncio.client import connect

    class NoRedirect(connect):
        def process_redirect(self, exc: Exception):
            return exc

    # TLS verification stays on; no proxy inheritance, redirects or compression.
    return NoRedirect(
        endpoint,
        additional_headers={"Authorization": "Bearer " + token, "DPoP": proof},
        proxy=None,
        compression=None,
        max_size=MAX_FRAME_BYTES,
        max_queue=8,
        open_timeout=10,
        close_timeout=3,
        ping_interval=None,
    )


class Companion:
    def __init__(
        self,
        *,
        enabled: bool,
        load_credential: Callable[[], Awaitable[ConnectorCredential | None]],
        exchange: Callable[[dict, ConnectorCredential], Awaitable[OriginResponse]],
        on_state: Callable[[str], None],
        dial: Callable[..., AsyncContextManager] = _dial,
        retry_ms: int = 1000,
        ready_timeout_ms: int = 10000,
        heartbeat_ms: int = 20000,
        param_headers: tuple[str, ...] = (),
        clock: Callable[[], float] = time.time,
    ):
        if type(enabled) is not bool or any(
            type(x) is not int or not 1 <= x <= 60000
            for x in (retry_ms, ready_timeout_ms, heartbeat_ms)
        ):
            raise ValueError("invalid_companion_configuration")
        self._enabled, self._load, self._exchange, self._observer, self._dial = (
            enabled,
            load_credential,
            exchange,
            on_state,
            dial,
        )
        self._retry, self._ready_timeout, self._heartbeat = retry_ms, ready_timeout_ms, heartbeat_ms
        self._params, self._clock = tuple(param_headers), clock
        self._task: asyncio.Task | None = None
        self._epoch = 0
        self._running = False

    def _publish(self, state: str) -> None:
        try:
            self._observer(state)
        except Exception:
            logger.warning("remote_companion_state_observer_failed")

    def _current(self, epoch: int) -> bool:
        return self._running and self._epoch == epoch

    async def start(self) -> None:
        if self._running:
            return
        if not self._enabled:
            self._publish("disabled")
            return
        self._running = True
        self._epoch += 1
        self._task = asyncio.create_task(self._run(self._epoch), name="slm-remote-companion")

    async def stop(self) -> None:
        self._running = False
        self._epoch += 1
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._publish("stopped")

    async def _run(self, epoch: int) -> None:
        failures = 0
        try:
            while self._current(epoch):
                try:
                    credential = await self._load()
                    if not self._current(epoch):
                        return
                    if credential is None or credential.expires_at_ms <= self._clock() * 1000:
                        self._publish("authorization_required")
                        return
                    _validate(credential)
                    if self._dial is _dial and not credential.device_private_key:
                        raise ValueError("device_key_required")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._publish("configuration_error")
                    return
                self._publish("connecting")
                try:
                    reason = await self._connection(credential, epoch)
                    if reason == "authorization_required":
                        self._publish(reason)
                        return
                    failures = 0 if reason == "connected_then_lost" else failures
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    from websockets.exceptions import InvalidStatus

                    if isinstance(error, InvalidStatus) and error.response.status_code in {
                        401,
                        403,
                    }:
                        self._publish("authorization_required")
                        return
                if not self._current(epoch):
                    return
                self._publish("reconnecting")
                await asyncio.sleep(min(60000, self._retry * 2 ** min(failures, 6)) / 1000)
                failures += 1
        finally:
            if self._epoch == epoch:
                self._running = False

    async def _connection(self, credential: ConnectorCredential, epoch: int) -> str:
        fault: list[str] = []
        connection = (
            _dial(ENDPOINT, credential.device_token, credential.device_private_key)
            if self._dial is _dial
            else self._dial(ENDPOINT, credential.device_token)
        )
        async with connection as socket:

            async def send(text: str) -> None:
                transport = getattr(socket, "transport", None)
                if transport is not None and transport.get_write_buffer_size() > MAX_FRAME_BYTES:
                    raise ValueError("connector_backpressure")
                await socket.send(text)

            session = RelaySession(
                credential,
                exchange=self._exchange,
                send=send,
                close=fault.append,
                param_headers=self._params,
                clock=self._clock,
            )
            try:
                ready = await asyncio.wait_for(socket.recv(), self._ready_timeout / 1000)
                await session.receive(ready)
                if not session.ready:
                    return fault[-1] if fault else "connector_protocol_error"
                if not self._current(epoch):
                    return "stopped"
                self._publish("transport_ready")
                waiting_pong = False
                while self._current(epoch) and session.ready:
                    if credential.expires_at_ms <= self._clock() * 1000:
                        return "authorization_required"
                    try:
                        message = await asyncio.wait_for(socket.recv(), self._heartbeat / 1000)
                    except TimeoutError:
                        if waiting_pong:
                            return "connected_then_lost"
                        await send("ping")
                        waiting_pong = True
                        continue
                    if not isinstance(message, str):
                        return "connected_then_lost"
                    if message == "pong" and waiting_pong:
                        waiting_pong = False
                        continue
                    await session.receive(message)
                return fault[-1] if fault else "connected_then_lost"
            finally:
                await session.stop()
