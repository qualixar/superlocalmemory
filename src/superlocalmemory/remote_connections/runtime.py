"""Opt-in lifecycle in the canonical daemon; no extra engine or database writer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import replace
from typing import Callable

from superlocalmemory.remote_connections.async_state import finish_on_cancel, mutate
from superlocalmemory.remote_connections.companion import Companion
from superlocalmemory.remote_connections.credentials import ConnectorCredential, CredentialVault
from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
from superlocalmemory.remote_connections.journal import EnrollmentJournal, JournalConflict
from superlocalmemory.remote_connections.native_enrollment import (
    NativeEnrollmentStore,
    PendingEnrollment,
)
from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin
from superlocalmemory.remote_connections.service import RemoteConnectionService
from superlocalmemory.server.remote_keys import RemoteKeyStore

MCP_URL = "https://mcp.superlocalmemory.com/mcp"


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
        self.provider = CloudGatewayProvider(store, redirect_uri=redirect_uri)
        self.current_profile, self.can_manage = current_profile, can_manage
        self.origin = CanonicalMcpOrigin(app)
        self.keys = RemoteKeyStore()
        self._restore_task: asyncio.Task | None = None
        self._companions: dict[str, Companion] = {}
        self._states: dict[str, str] = {}
        self._verified: set[str] = set()
        self._epochs: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._verification_tasks: dict[str, asyncio.Task] = {}
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

    async def verify(self, row: PendingEnrollment, epoch: int | None = None) -> None:
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
                and self._states.get(row.connection_id) == "transport_ready"
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
            return False
        if (row.owner, row.profile) != (owner, profile):
            raise ValueError("connection_binding_mismatch")
        await asyncio.to_thread(
            self.vault().revoke, row.installation_id, owner, profile, connection
        )
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
            if connection["state"] == "pending" and identifier in self.runtime._verified:
                connection.update(
                    state="ready_for_client",
                    verified=True,
                    mcp_url=MCP_URL,
                    account_provider="github",
                )
            elif connection["state"] == "pending" and identifier in self.runtime._states:
                connection["transport_state"] = self.runtime._states[identifier]
            if connection["state"] == "pending":
                row = self.runtime.store.by_connection(identifier, for_cleanup=True)
                if row and not row.completed:
                    connection["authorization_expires_at_ms"] = row.expires_at_ms
                    connection["sign_in_state"] = (
                        "expired" if row.expires_at_ms <= time.time() * 1000 else "required"
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
