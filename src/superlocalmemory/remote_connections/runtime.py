"""Opt-in lifecycle in the canonical daemon; no extra engine or database writer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import replace
from typing import Callable

from superlocalmemory.remote_connections.async_state import finish_on_cancel, mutate
from superlocalmemory.remote_connections.companion import Companion
from superlocalmemory.remote_connections.credentials import ConnectorCredential, CredentialVault
from superlocalmemory.remote_connections import peer_names, peer_sync
from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
from superlocalmemory.remote_connections.grant_keys import GrantKeyStore
from superlocalmemory.remote_connections.journal import EnrollmentJournal, JournalConflict
from superlocalmemory.remote_connections.native_enrollment import (
    NativeEnrollmentStore,
    PendingEnrollment,
)
from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin
from superlocalmemory.remote_connections.renewal import (
    RENEWAL_CHECK_S,
    RENEWAL_RETRY_S,
    RENEWAL_WINDOW_MS,
    access_state,
    renewal_delay_s,
)
from superlocalmemory.remote_connections.service import RemoteConnectionService
from superlocalmemory.server.remote_keys import RemoteKeyStore

logger = logging.getLogger(__name__)
MCP_URL = "https://mcp.superlocalmemory.com/mcp"
__all__ = ["NativeConnectionRuntime", "RENEWAL_CHECK_S", "RENEWAL_WINDOW_MS", "renewal_delay_s"]


def _connected_app(value: object) -> bool:
    """Gateway rows are re-checked before the dashboard sees them."""
    if not isinstance(value, dict) or set(value) != {
        "authorization_id",
        "name",
        "client_host",
        "permissions",
        "version",
        "connected_at_ms",
        "last_used_at_ms",
    }:
        return False
    permissions = value["permissions"]
    return (
        isinstance(value["authorization_id"], str)
        and 0 < len(value["authorization_id"]) <= 256
        and isinstance(value["name"], str)
        and 0 < len(value["name"]) <= 80
        and (value["client_host"] is None or isinstance(value["client_host"], str))
        and isinstance(permissions, dict)
        and set(permissions) == {"read", "save", "session"}
        and all(type(flag) is bool for flag in permissions.values())
        and type(value["version"]) is int
        and value["version"] >= 1
        and all(
            value[key] is None or type(value[key]) is int
            for key in ("connected_at_ms", "last_used_at_ms")
        )
    )


def _connected_app_v2(value: object) -> bool:
    """A row of the version-2 list: the same row, plus mesh and media consent."""
    if not isinstance(value, dict) or not isinstance(value.get("permissions"), dict):
        return False
    permissions = value["permissions"]
    if set(permissions) != {"read", "save", "session", "mesh", "media"}:
        return False
    v1 = dict(value, permissions={k: permissions[k] for k in ("read", "save", "session")})
    return _connected_app(v1) and all(type(flag) is bool for flag in permissions.values())


#: At most one forced grant-key refresh per connection in this many seconds.
GRANT_REFRESH_INTERVAL_S = 600.0
PEER_SYNC_INTERVAL_S = peer_sync.INTERVAL_S


class NativeConnectionRuntime:
    def __init__(
        self,
        app,
        journal: EnrollmentJournal,
        store: NativeEnrollmentStore,
        *,
        current_profile: Callable[[], str],
        can_manage: Callable[[str, str], bool],
        redirect_uri: str,
    ):
        self.journal, self.store = journal, store
        self._app = app
        self._peer_sync_tasks: dict[str, asyncio.Task] = {}
        self.provider = CloudGatewayProvider(store, redirect_uri=redirect_uri)
        self.current_profile, self.can_manage = current_profile, can_manage
        self.grant_keys = GrantKeyStore(lambda: self.store.backend)
        self._grant_now: Callable[[], float] = time.time
        self._grant_asked: dict[str, float] = {}
        self._grant_tasks: dict[str, asyncio.Task] = {}
        self.origin = CanonicalMcpOrigin(
            app, grant_keys=self.grant_keys.load, on_unknown_kid=self.request_grant_refresh)
        self.keys = RemoteKeyStore()
        self._restore_task: asyncio.Task | None = None
        self._companions: dict[str, Companion] = {}
        self._states: dict[str, str] = {}
        self._verified: set[str] = set()
        self._epochs: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._verification_tasks: dict[str, asyncio.Task] = {}
        self._renewal_tasks: dict[str, asyncio.Task] = {}
        self._recovery_tasks: dict[str, asyncio.Task] = {}
        #: A transient verify failure (cold engine, network blip) is retried while the
        #: same transport stays up, so the dashboard does not stay wrongly unavailable.
        self._verify_retry_limit = 5
        self._verify_retry_base_s = 5.0
        self.service = ManagedConnectionService(
            journal,
            self.provider,
            hosts=("muse", "chatgpt", "claude_web", "claude_code_web", "composio", "other_mcp"),
            runtime=self,
        )

    def vault(self) -> CredentialVault:
        return CredentialVault(self.store.root / "connector", backend=self.store.backend)

    def _creator(self, row: PendingEnrollment) -> None:
        if self.current_profile() != row.profile:
            raise ValueError("profile_changed")
        if not self.can_manage(row.owner, row.profile):
            raise ValueError("creator_permission_changed")

    async def _current(self, row: PendingEnrollment, version: int | None = None):
        self._creator(row)
        record = await asyncio.to_thread(
            self.journal.get, row.owner, row.profile, row.connection_id
        )
        if record.state != "pending" or (version is not None and record.version != version):
            raise ValueError("enrollment_cancelled")
        return record

    async def callback(self, state: str, code: str) -> str:
        row = await asyncio.to_thread(self.store.by_state, state)
        if row is None:
            raise ValueError("invalid_callback_state")
        self._creator(row)
        lock = self._locks.setdefault(row.connection_id, asyncio.Lock())
        async with lock:
            try:
                row = await asyncio.to_thread(self.store.by_state, state)
                if row is None:
                    raise ValueError("invalid_callback_state")
                record = await self._current(row)
                row = await self.provider.exchange(row, code)
                await self._current(row, record.version)
                delivered = await self.provider.provision(row)
                await self._current(row, record.version)
                if not row.origin_key:
                    name = "web-" + row.connection_id
                    # An interrupted secure-store write can leave an unusable key.
                    # Revoke only this connection's own key before recreating it.
                    for key in await asyncio.to_thread(self.keys.list):
                        if key.name == name and key.active:
                            await mutate(self.keys.revoke, key.key_id)
                    _, secret = await mutate(
                        self.keys.add,
                        name,
                        "write"
                        if (
                            record.intent["permissions"]["write"]
                            or record.intent["permissions"]["session"]
                        )
                        else "read",
                        profile=row.profile,
                    )
                    row = replace(row, origin_key=secret)
                    await mutate(self.store.save, row)
                credential = ConnectorCredential(
                    row.installation_id,
                    row.owner,
                    row.profile,
                    row.connection_id,
                    delivered["generation"],
                    delivered["expires_at_ms"],
                    delivered["device_token"],
                    row.origin_key,
                    row.private_key,
                )
                await self._current(row, record.version)
                await mutate(self.vault().save, credential)
                row = replace(row, expires_at_ms=credential.expires_at_ms, completed=True)
                await mutate(self.store.save, row)
                await self._current(row, record.version)
                await self.start(row)
                return row.connection_id
            except asyncio.CancelledError:
                await finish_on_cancel(self._abandon_callback(row))
                raise

    async def _abandon_callback(self, row: PendingEnrollment) -> None:
        current = await asyncio.to_thread(
            self.journal.get, row.owner, row.profile, row.connection_id
        )
        if current.state == "pending":
            await mutate(
                self.journal.cancel, row.owner, row.profile, row.connection_id, current.version
            )
        if await self._cancel(row.owner, row.profile, row.connection_id):
            await mutate(self.journal.clear_cleanup, row.owner, row.profile, row.connection_id)

    async def start(self, row: PendingEnrollment) -> None:
        existing = self._companions.get(row.connection_id)
        if existing is not None:
            await self._current(row)
            if not existing._running:
                await existing.start()
            return
        await self._current(row)

        async def load():
            try:
                await self._current(row)
                return await asyncio.to_thread(
                    self.vault().load,
                    row.installation_id,
                    row.owner,
                    row.profile,
                    row.connection_id,
                )
            except (ValueError, JournalConflict):
                return None

        async def exchange(frame, credential):
            await self._current(row)
            response = await self.origin(frame, credential)
            await self._current(row)
            return response

        def state(value: str):
            self._epochs[row.connection_id] = self._epochs.get(row.connection_id, 0) + 1
            self._states[row.connection_id] = value
            if value == "authorization_required":
                self._start_recovery(row)
            if value != "transport_ready":
                self._verified.discard(row.connection_id)
            else:
                previous = self._verification_tasks.get(row.connection_id)
                if previous:
                    previous.cancel()
                self._verification_tasks[row.connection_id] = asyncio.create_task(
                    self.verify(row, self._epochs[row.connection_id])
                )

        companion = Companion(enabled=True, load_credential=load, exchange=exchange, on_state=state)
        self._companions[row.connection_id] = companion
        await companion.start()
        self._schedule_renewal(row)
        self._schedule_grant_work(row, force=False)
        self._schedule_peer_sync(row)

    async def renew_credential(self, row: PendingEnrollment, *, recovering: bool = False) -> str:
        """Replace this connection's laptop credential.

        The link is paused first, so it never dials with a credential the gateway
        has just retired. Outcomes: renewed, not_due, unavailable, authorization_required.
        When recovering from a refusal, the link reconnects only with a newer
        credential, so a refused credential is never retried in a loop."""
        lock = self._locks.setdefault(row.connection_id, asyncio.Lock())
        async with lock:
            await self._current(row)
            identity = (row.installation_id, row.owner, row.profile, row.connection_id)
            held = await asyncio.to_thread(self.vault().generation, *identity)
            latest = await asyncio.to_thread(self.store.by_connection, row.connection_id)
            if held is None or latest is None or not latest.completed:
                self._states[row.connection_id] = "authorization_required"
                return "authorization_required"
            companion = self._companions.get(row.connection_id)
            if companion is not None:
                await companion.stop()
            outcome = await self._replace_credential(latest, held)
            if outcome == "renewed" or (outcome != "authorization_required" and not recovering):
                if companion is not None:
                    await companion.start()
            else:
                self._states[row.connection_id] = "authorization_required"
            return outcome

    async def _replace_credential(self, latest: PendingEnrollment, held: int) -> str:
        try:
            latest = await self.provider.exchange(latest, "")
            try:
                delivered = await self.provider.renew(latest, held)
            except ValueError as error:
                if not error.args or error.args[0] != "renewal_conflict":
                    raise
                # Another renewal already won, or none is due: take what the gateway holds.
                delivered = await self.provider.provision(latest)
                if delivered["generation"] <= held:
                    return "not_due"
        except ValueError as error:
            if error.args and error.args[0] == "connection_unavailable":
                return "authorization_required"
            return "unavailable"
        credential = ConnectorCredential(
            latest.installation_id,
            latest.owner,
            latest.profile,
            latest.connection_id,
            delivered["generation"],
            delivered["expires_at_ms"],
            delivered["device_token"],
            latest.origin_key,
            latest.private_key,
        )
        await mutate(self.vault().save, credential)
        await mutate(self.store.save, replace(latest, expires_at_ms=credential.expires_at_ms))
        return "renewed"

    async def recover(self, row: PendingEnrollment) -> str:
        """The gateway refused the held credential: it may hold a newer one (a renewal
        that finished there but was not saved here) or the held one expired."""
        return await self.renew_credential(row, recovering=True)

    def _start_recovery(self, row: PendingEnrollment) -> None:
        running = self._recovery_tasks.get(row.connection_id)
        if running is None or running.done():
            self._recovery_tasks[row.connection_id] = asyncio.create_task(
                self._quietly(self.recover(row)), name="slm-remote-recovery"
            )

    def _schedule_renewal(self, row: PendingEnrollment) -> None:
        running = self._renewal_tasks.get(row.connection_id)
        if running is None or running.done():
            self._renewal_tasks[row.connection_id] = asyncio.create_task(
                self._quietly(self._renew_when_due(row)), name="slm-remote-renewal"
            )

    async def _renew_when_due(self, row: PendingEnrollment) -> None:
        identity = (row.installation_id, row.owner, row.profile, row.connection_id)
        while True:
            held = await asyncio.to_thread(self.vault().load, *identity)
            if held is None:
                return  # missing or expired: the link's refusal triggers recovery
            delay = renewal_delay_s(held.expires_at_ms, time.time() * 1000)
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            outcome = await self.renew_credential(row)
            if outcome == "authorization_required":
                return
            if outcome != "renewed":
                pause = RENEWAL_RETRY_S if outcome == "unavailable" else RENEWAL_CHECK_S
                await asyncio.sleep(pause)

    @staticmethod
    async def _quietly(work) -> None:
        """A cancelled or removed connection ends its background renewal silently."""
        try:
            await work
        except (ValueError, JournalConflict):
            return

    async def verify(
        self, row: PendingEnrollment, epoch: int | None = None, attempt: int = 0
    ) -> None:
        captured = self._epochs.get(row.connection_id, 0) if epoch is None else epoch
        try:
            await self._current(row)
            latest = await asyncio.to_thread(self.store.by_connection, row.connection_id)
            if latest is None:
                return
            latest = await self.provider.exchange(latest, "")
            result = await self.provider.verify(latest)
            await self._current(latest)
            if (
                self._epochs.get(row.connection_id, 0) == captured
                # The epoch moves on every transport change, so an unchanged epoch
                # means the same socket is still up even after a failed attempt.
                and self._states.get(row.connection_id)
                in {"transport_ready", "verification_unavailable"}
                and result.get("verified") is True
                and result.get("connection_id") == row.connection_id
            ):
                self._verified.add(row.connection_id)
                self._states[row.connection_id] = "ready_for_client"
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._epochs.get(row.connection_id, 0) == captured:
                self._verified.discard(row.connection_id)
                self._states[row.connection_id] = "verification_unavailable"
                if attempt < self._verify_retry_limit:
                    delay = self._verify_retry_base_s * 2**attempt
                    self._verification_tasks[row.connection_id] = asyncio.create_task(
                        self._verify_later(row, captured, attempt + 1, delay)
                    )

    async def _verify_later(
        self, row: PendingEnrollment, epoch: int, attempt: int, delay: float
    ) -> None:
        await asyncio.sleep(delay)
        if self._epochs.get(row.connection_id, 0) == epoch:
            await self.verify(row, epoch, attempt)

    async def _owned_link(self, owner: str, profile: str, connection_id: str):
        """:meth:`_owned_link_unlocked` under the connection's lock, like every other
        exchange of the owner token."""
        lock = self._locks.setdefault(connection_id, asyncio.Lock())
        async with lock:
            return await self._owned_link_unlocked(owner, profile, connection_id)

    async def _owned_link_unlocked(self, owner: str, profile: str, connection_id: str):
        """The completed laptop link for a connection this owner/profile holds, with a
        fresh owner token. Anything else is reported as not_found (no existence leak)."""
        try:
            record = await asyncio.to_thread(self.journal.get, owner, profile, connection_id)
        except (JournalConflict, ValueError):
            raise ValueError("not_found") from None
        if record.state != "pending":
            raise ValueError("not_found")
        row = await asyncio.to_thread(self.store.by_connection, connection_id)
        if row is None or not row.completed:
            raise ValueError("not_found")
        return await self.provider.exchange(row, "")

    async def list_apps(self, owner: str, profile: str, connection_id: str) -> dict:
        started = peer_sync.started_at()
        latest = await self._owned_link(owner, profile, connection_id)
        value = await self.provider.list_apps(latest)
        apps = value.get("apps") if isinstance(value, dict) else None
        rows = apps if isinstance(apps, list) else []
        listed = [app for app in rows if _connected_app(app)]
        await self._sync_peers_quietly(connection_id, listed, apps, started)
        return {"connection_id": connection_id, "apps": listed}

    async def _apps_v2(self, owner: str, profile: str, connection_id: str):
        """``(readable rows, raw answer)`` of the version-2 connected-apps list."""
        latest = await self._owned_link(owner, profile, connection_id)
        value = await self.provider.list_apps_v2(latest)
        apps = value.get("apps") if isinstance(value, dict) else None
        rows = apps if isinstance(apps, list) else []
        return [app for app in rows if _connected_app_v2(app)], apps

    async def list_apps_v2(self, owner: str, profile: str, connection_id: str) -> dict:
        """Connected apps including mesh/media consent; feeds the peer-name cache."""
        listed, _ = await self._apps_v2(owner, profile, connection_id)
        return {"connection_id": connection_id, "apps": listed}

    async def refresh_peer_names(self, owner: str, profile: str, connection_id: str) -> None:
        """Re-read the app list: refresh the names and retire the peers of revoked apps."""
        started = peer_sync.started_at()
        listed, apps = await self._apps_v2(owner, profile, connection_id)
        await self._sync_peers(connection_id, listed, apps, started)

    async def _sync_peers(self, connection_id: str, listed: list, apps: object,
                          started: str) -> None:
        broker = getattr(getattr(self._app, "state", None), "mesh_broker", None)
        await asyncio.to_thread(
            peer_sync.apply, connection_id, listed,
            complete=peer_sync.is_complete(apps, listed), broker=broker, started=started)

    async def _sync_peers_quietly(self, connection_id: str, listed: list, apps: object,
                                  started: str) -> None:
        """Viewing the apps keeps the mesh in step; a failure here never fails the view."""
        try:
            await self._sync_peers(connection_id, listed, apps, started)
        except Exception:
            logger.debug("peer sync after viewing apps failed", exc_info=True)

    def _schedule_peer_sync(self, row: PendingEnrollment) -> None:
        running = self._peer_sync_tasks.get(row.connection_id)
        if running is None or running.done():
            self._peer_sync_tasks[row.connection_id] = asyncio.create_task(
                self._quietly(self._sync_peers_while_running(row)), name="slm-remote-peer-sync")

    async def _sync_peers_while_running(self, row: PendingEnrollment) -> None:
        """Every ten minutes while the connection runs; ends when it is cancelled."""
        while True:
            try:
                await self.refresh_peer_names(row.owner, row.profile, row.connection_id)
            except asyncio.CancelledError:
                raise
            except JournalConflict:
                return
            except Exception as error:
                if isinstance(error, ValueError) and error.args[:1] == ("not_found",):
                    return  # the connection is gone
                if isinstance(error, ValueError) and error.args[:1] == ("connection_unavailable",):
                    await self._sync_peers_quietly(
                        row.connection_id, [], [], peer_sync.started_at())
                    return  # the gateway no longer knows it: no app can call
                logger.debug("peer sync unavailable for %s", row.connection_id[:6])
            await asyncio.sleep(PEER_SYNC_INTERVAL_S)

    async def ensure_grant_key(self, row: PendingEnrollment, *, force: bool = False) -> None:
        """Fetch a grant key when none is held (or ``force``). Failure is silent:
        without a key there is simply no grant, which is today's behaviour."""
        try:
            if not force and self.grant_keys.load(row.connection_id).current is not None:
                return
            lock = self._locks.setdefault(row.connection_id, asyncio.Lock())
            async with lock:
                latest = await self.provider.exchange(
                    await asyncio.to_thread(self.store.by_connection, row.connection_id) or row,
                    "")
                fetched = await self.provider.grant_key(latest)
            await asyncio.to_thread(
                self.grant_keys.store_new, row.connection_id, fetched["version"], fetched["key"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("grant key unavailable for %s", row.connection_id[:6])

    async def rotate_grant_key(self, owner: str, profile: str, connection_id: str) -> int:
        """Owner action: replace this connection's grant key. Returns the new version."""
        async with self._locks.setdefault(connection_id, asyncio.Lock()):
            latest = await self._owned_link_unlocked(owner, profile, connection_id)
            fetched = await self.provider.grant_key(latest)
            await asyncio.to_thread(
                self.grant_keys.store_new, connection_id, fetched["version"], fetched["key"])
        return fetched["version"]

    def request_grant_refresh(self, connection_id: str) -> bool:
        """Ask for a fresh grant key in the background; at most once per 10 minutes
        per connection. Returns whether a refresh was started."""
        now = self._grant_now()
        asked = self._grant_asked.get(connection_id)
        if asked is not None and now - asked < GRANT_REFRESH_INTERVAL_S:
            return False
        row = self.store.by_connection(connection_id)
        if row is None or not row.completed:
            return False
        self._grant_asked[connection_id] = now
        self._schedule_grant_work(row, force=True)
        return True

    def _schedule_grant_work(self, row: PendingEnrollment, *, force: bool) -> None:
        running = self._grant_tasks.get(row.connection_id)
        if running is not None and not running.done():
            return
        self._grant_tasks[row.connection_id] = asyncio.create_task(
            self._quietly(self.ensure_grant_key(row, force=force)), name="slm-remote-grant-key")

    async def revoke_app(
        self,
        owner: str,
        profile: str,
        connection_id: str,
        authorization_id: str,
        expected_version: int,
    ) -> dict:
        latest = await self._owned_link(owner, profile, connection_id)
        value = await self.provider.revoke_app(latest, authorization_id, expected_version)
        if not isinstance(value, dict) or value.get("revoked") is not True:
            raise ValueError("apps_unavailable")
        return {"revoked": True}

    async def resume(self, owner: str, profile: str) -> None:
        for record in await asyncio.to_thread(self.journal.list, owner, profile):
            if record.state != "pending":
                continue
            existing = self._companions.get(record.connection_id)
            if existing is not None and existing._running:
                continue
            row = await asyncio.to_thread(self.store.by_connection, record.connection_id)
            if row and row.completed:
                try:
                    await self.start(row)
                except (ValueError, JournalConflict):
                    self._states[row.connection_id] = "authorization_required"

    async def restore(self) -> None:
        """Resume prior opt-ins in the active profile without delaying local startup."""
        try:
            profile = self.current_profile()
            owners = await asyncio.to_thread(self.journal.pending_owners, profile)
        except Exception:
            import logging

            logging.getLogger(__name__).warning("remote_connections_restore_unavailable")
            return
        for owner in owners:
            if not self.can_manage(owner, profile) or self.current_profile() != profile:
                continue
            try:
                await self.resume(owner, profile)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A locked keychain/provider cannot interrupt local memory service.
                import logging

                logging.getLogger(__name__).warning("remote_connections_restore_unavailable")

    def schedule_restore(self) -> None:
        if self._restore_task is None or self._restore_task.done():
            self._restore_task = asyncio.create_task(self.restore(), name="slm-remote-restore")

    async def cancel(self, owner: str, profile: str, connection: str) -> bool:
        lock = self._locks.setdefault(connection, asyncio.Lock())
        async with lock:
            return await self._cancel(owner, profile, connection)

    async def _cancel(self, owner: str, profile: str, connection: str) -> bool:
        await asyncio.to_thread(self.journal.get, owner, profile, connection)
        companion = self._companions.pop(connection, None)
        task = self._verification_tasks.pop(connection, None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if companion:
            await companion.stop()
        self._epochs[connection] = self._epochs.get(connection, 0) + 1
        self._verified.discard(connection)
        self._states[connection] = "stopped"
        row = await asyncio.to_thread(self.store.by_connection, connection, for_cleanup=True)
        for key in await asyncio.to_thread(self.keys.list):
            if key.name == "web-" + connection and key.active:
                await mutate(self.keys.revoke, key.key_id)
        if row is None:
            await self._sync_peers_quietly(connection, [], [], peer_sync.started_at())
            return False
        if (row.owner, row.profile) != (owner, profile):
            raise ValueError("connection_binding_mismatch")
        await asyncio.to_thread(
            self.vault().revoke, row.installation_id, owner, profile, connection
        )
        await asyncio.to_thread(self.grant_keys.forget, connection)
        peer_names.set_names(connection, {})
        syncing = self._peer_sync_tasks.pop(connection, None)
        if syncing is not None:
            syncing.cancel()
        await self._sync_peers_quietly(connection, [], [], peer_sync.started_at())
        try:
            if row.access_token:
                row = await self.provider.exchange(row, "")
                reply = await self.provider.revoke(row)
                confirmed = reply.get("revoked") is True
            else:
                reply = await self.provider.cancel_bootstrap(row)
                confirmed = reply.get("cancelled") is True
            if confirmed:
                await asyncio.to_thread(self.store.cancel, connection)
                return True
        except Exception:
            try:
                reply = await self.provider.cancel_bootstrap(row)
                if reply.get("cancelled") is True:
                    await asyncio.to_thread(self.store.cancel, connection)
                    return True
            except Exception:
                return False
        return False

    async def stop(self) -> None:
        if self._restore_task is not None:
            self._restore_task.cancel()
            await asyncio.gather(self._restore_task, return_exceptions=True)
            self._restore_task = None
        for task in self._verification_tasks.values():
            task.cancel()
        await asyncio.gather(*self._verification_tasks.values(), return_exceptions=True)
        self._verification_tasks.clear()
        background = [*self._renewal_tasks.values(), *self._recovery_tasks.values(),
                      *self._grant_tasks.values(), *self._peer_sync_tasks.values()]
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        self._renewal_tasks.clear()
        self._recovery_tasks.clear()
        self._grant_tasks.clear()
        self._peer_sync_tasks.clear()
        for companion in self._companions.values():
            await companion.stop()
        self._companions.clear()


class ManagedConnectionService(RemoteConnectionService):
    def __init__(self, *args, runtime: NativeConnectionRuntime, **kwargs):
        super().__init__(*args, **kwargs)
        self.runtime = runtime
        self._restart_locks: dict[tuple[str, str], asyncio.Lock] = {}

    def status(self, owner: str, profile: str) -> dict:
        result = super().status(owner, profile)
        for connection in result["connections"]:
            identifier = connection["connection_id"]
            was_pending = connection["state"] == "pending"
            if connection["state"] == "pending" and identifier in self.runtime._verified:
                connection.update(
                    state="ready_for_client",
                    verified=True,
                    mcp_url=MCP_URL,
                    account_provider="github",
                )
            elif connection["state"] == "pending" and identifier in self.runtime._states:
                connection["transport_state"] = self.runtime._states[identifier]
            if was_pending:
                row = self.runtime.store.by_connection(identifier, for_cleanup=True)
                if row and not row.completed:
                    connection["authorization_expires_at_ms"] = row.expires_at_ms
                    connection["sign_in_state"] = (
                        "expired" if row.expires_at_ms <= time.time() * 1000 else "required"
                    )
                elif row:
                    connection["access_expires_at_ms"] = row.expires_at_ms
                    connection["access_state"] = access_state(
                        row.expires_at_ms,
                        time.time() * 1000,
                        self.runtime._states.get(identifier),
                    )
        return result

    async def restart(self, owner: str, profile: str, identifier: str, version: int) -> dict:
        """Revoke first, then retry the same consent with a durable idempotent key."""
        lock = self._restart_locks.setdefault((owner, profile), asyncio.Lock())
        async with lock:
            if self.runtime.current_profile() != profile or not self.runtime.can_manage(
                owner, profile
            ):
                raise JournalConflict("profile_changed")
            row = await asyncio.to_thread(self.journal.get, owner, profile, identifier)
            if identifier in self.runtime._verified:
                raise JournalConflict("connection_already_verified")
            if row.state == "pending" and row.version != version:
                raise JournalConflict("version_conflict")
            if row.state == "cancelled" and row.version not in {version, version + 1, version + 2}:
                raise JournalConflict("version_conflict")
            if row.state not in {"pending", "cancelled"}:
                raise JournalConflict("connection_not_pending")
            if row.state == "pending" or row.cleanup_pending:
                result = await self.cancel(owner, profile, identifier, row.version)
                if result["cleanup_pending"]:
                    raise JournalConflict("cleanup_pending")
            if self.runtime.current_profile() != profile or not self.runtime.can_manage(
                owner, profile
            ):
                raise JournalConflict("profile_changed")
            # Derive the same new intent after lost HTTP replies/process restarts.
            # This key conveys no authority; local owner/profile checks still apply.
            material = json.dumps(
                ["restart-v1", row.installation_id, owner, profile, identifier]
            )
            key = hashlib.sha256(material.encode()).hexdigest()[:32]
            return await self.initiate(owner, profile, key, row.intent)

    async def cancel(self, owner: str, profile: str, identifier: str, version: int) -> dict:
        await super().cancel(owner, profile, identifier, version)
        if await self.runtime.cancel(owner, profile, identifier):
            await asyncio.to_thread(self.journal.clear_cleanup, owner, profile, identifier)
        row = await asyncio.to_thread(self.journal.get, owner, profile, identifier)
        return self.public(row)


def install_runtime(application, *, root=None) -> NativeConnectionRuntime:
    """Advertise feature availability; keyring and network remain dormant."""
    from superlocalmemory.access.rbac import Permission
    from superlocalmemory.infra.data_root import state_path
    from superlocalmemory.server.profile_runtime import get_profile_runtime

    root = root or state_path("web-connections")
    descriptor = application.state.daemon_descriptor
    port = descriptor.port

    def profile():
        return get_profile_runtime(application.state).snapshot.profile_id

    def allowed(owner, chosen):
        if owner == "owner":
            return True
        rbac = getattr(application.state, "rbac", None)
        return bool(rbac is not None and rbac.has_permission(owner, chosen, Permission.MANAGE))

    runtime = NativeConnectionRuntime(
        application,
        EnrollmentJournal(root / "journal"),
        NativeEnrollmentStore(root / "secure"),
        current_profile=profile,
        can_manage=allowed,
        redirect_uri=f"http://127.0.0.1:{port}/api/v3/connections/callback",
    )
    application.state.remote_connection_runtime = runtime
    application.state.remote_connections = runtime.service
    return runtime
