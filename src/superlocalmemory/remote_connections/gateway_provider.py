"""First-party cloud enrollment, activated only by dashboard opt-in."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import replace
from typing import Awaitable, Callable
from urllib.parse import urlencode

from superlocalmemory.remote_connections.async_state import mutate
from superlocalmemory.remote_connections.native_enrollment import (
    NativeEnrollmentStore,
    PendingEnrollment,
)
from superlocalmemory.remote_connections.proof import DeviceSigner
from superlocalmemory.remote_connections.service import GatewayReceipt, validate_sign_in

AUTH = "https://auth.superlocalmemory.com"
OWNER_RESOURCE = AUTH + "/owner"


class _GatewayAnswer(Exception):
    """A non-success gateway status, mapped to a fixed code before leaving _request."""


class CloudGatewayProvider:
    def __init__(
        self,
        store: NativeEnrollmentStore,
        *,
        redirect_uri: str,
        http: Callable[..., Awaitable[dict]] | None = None,
    ):
        self.store, self.redirect_uri = store, redirect_uri
        self._http = http or self._request
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    @staticmethod
    async def _request(path: str, **kwargs) -> dict:
        import httpx

        if path not in {
            "/oauth/register",
            "/bootstrap",
            "/oauth/token",
            "/owner/connections",
            "/owner/revoke",
            "/owner/verify",
            "/owner/apps",
            "/owner/apps/revoke",
            "/bootstrap/cancel",
        }:
            raise ValueError("invalid_gateway_endpoint")
        try:
            async with httpx.AsyncClient(
                trust_env=False, follow_redirects=False, timeout=10
            ) as client:
                async with client.stream("POST", AUTH + path, **kwargs) as response:
                    if not response.is_success:
                        distinct = CloudGatewayProvider._removal_error(path, response.status_code)
                        raise _GatewayAnswer(distinct or "unavailable")
                    chunks, length = [], 0
                    async for chunk in response.aiter_bytes():
                        length += len(chunk)
                        if length > 65536:
                            raise ValueError("too_large")
                        chunks.append(chunk)
                    value = json.loads(b"".join(chunks).decode("utf-8"))
                    if not isinstance(value, dict):
                        raise ValueError("invalid")
                    return value
        except _GatewayAnswer as answer:
            if answer.args[0] in {"not_found", "version_conflict"}:
                raise ValueError(answer.args[0]) from None
            raise ValueError("remote_gateway_unavailable") from None
        except Exception:
            raise ValueError("remote_gateway_unavailable") from None

    async def enroll(
        self, *, installation_id: str, connection_id: str, owner: str, profile: str, intent: dict
    ) -> GatewayReceipt:
        lock = self._locks.setdefault((installation_id, owner, profile), asyncio.Lock())
        async with lock:
            return await self._enroll(
                installation_id=installation_id,
                connection_id=connection_id,
                owner=owner,
                profile=profile,
                intent=intent,
            )

    async def _enroll(
        self, *, installation_id: str, connection_id: str, owner: str, profile: str, intent: dict
    ) -> GatewayReceipt:
        encoded = json.dumps(intent, sort_keys=True, separators=(",", ":"))
        row = await asyncio.to_thread(self.store.by_connection, connection_id)
        if row is None:
            desktop = await asyncio.to_thread(
                self.store.profile_client, installation_id, owner, profile
            )
            if desktop and desktop["redirect_uri"] != self.redirect_uri:
                raise ValueError("desktop_callback_changed")
            signer = DeviceSigner(desktop["private_key"]) if desktop else DeviceSigner.generate()
            row = PendingEnrollment(
                installation_id=installation_id,
                owner=owner,
                profile=profile,
                connection_id=connection_id,
                redirect_uri=self.redirect_uri,
                state=secrets.token_urlsafe(32),
                verifier=secrets.token_urlsafe(48),
                private_key=signer.private_pem,
                intent_json=encoded,
                expires_at_ms=int(time.time() * 1000) + 15 * 60 * 1000,
                client_id=desktop["client_id"] if desktop else "",
            )
            await mutate(self.store.save, row)
        if (row.installation_id, row.owner, row.profile, row.intent_json) != (
            installation_id,
            owner,
            profile,
            encoded,
        ) or row.completed:
            raise ValueError("enrollment_binding_conflict")
        if not row.client_id:
            recovered = await asyncio.to_thread(
                self.store.profile_client, installation_id, owner, profile
            )
            if recovered:
                if (
                    recovered["private_key"] != row.private_key
                    or recovered["redirect_uri"] != row.redirect_uri
                ):
                    raise ValueError("desktop_binding_conflict")
                row = replace(row, client_id=recovered["client_id"])
                await mutate(self.store.save, row)
        if not row.client_id:
            registered = await self._http(
                "/oauth/register",
                json={
                    "client_name": "SuperLocalMemory Desktop",
                    "redirect_uris": [row.redirect_uri],
                    "token_endpoint_auth_method": "none",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                },
            )
            client = registered.get("client_id")
            if not isinstance(client, str) or not client or len(client) > 2048:
                raise ValueError("invalid_client_registration")
            row = replace(row, client_id=client)
            await mutate(self.store.bind_profile_client, row)
            await mutate(self.store.save, row)
        from superlocalmemory.remote_connections.proof import _base64url

        authorization_url = (
            AUTH
            + "/authorize?"
            + urlencode(
                {
                    "response_type": "code",
                    "client_id": row.client_id,
                    "redirect_uri": row.redirect_uri,
                    "scope": "slm:connect",
                    "resource": OWNER_RESOURCE,
                    "state": row.state,
                    "code_challenge": _base64url(hashlib.sha256(row.verifier.encode()).digest()),
                    "code_challenge_method": "S256",
                }
            )
        )
        receipt = await self._http(
            "/bootstrap",
            json={
                "connectionId": connection_id,
                "installationId": installation_id,
                "profileId": profile,
                "host": intent["host"],
                "permissions": intent["permissions"],
                "deviceJwk": DeviceSigner(row.private_key).public_jwk,
                "expiresAtMs": row.expires_at_ms,
                "authorizationUrl": authorization_url,
            },
        )
        if receipt.get("connection_id") != connection_id:
            raise ValueError("invalid_gateway_receipt")
        url = validate_sign_in(receipt.get("authorize_url"), connection_id)
        return GatewayReceipt(reference=connection_id, authorization_url=url)

    async def exchange(self, row: PendingEnrollment, code: str) -> PendingEnrollment:
        if not row.access_token:
            if not isinstance(code, str) or not code or len(code) > 8192:
                raise ValueError("invalid_authorization_code")
            issued = await self._http(
                "/oauth/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": row.client_id,
                    "redirect_uri": row.redirect_uri,
                    "code_verifier": row.verifier,
                    "resource": OWNER_RESOURCE,
                },
            )
            row = self._tokens(row, issued)
            await mutate(self.store.save, row)
        elif row.access_expires_ms <= time.time() * 1000 + 30000:
            issued = await self._http(
                "/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": row.client_id,
                    "refresh_token": row.refresh_token,
                    "resource": OWNER_RESOURCE,
                },
            )
            row = self._tokens(row, issued)
            await mutate(self.store.save, row)
        return row

    @staticmethod
    def _tokens(row: PendingEnrollment, issued: dict) -> PendingEnrollment:
        access, refresh, seconds = (
            issued.get("access_token"),
            issued.get("refresh_token"),
            issued.get("expires_in"),
        )
        if (
            not isinstance(access, str)
            or not access
            or len(access) > 8192
            or not isinstance(refresh, str)
            or not refresh
            or len(refresh) > 8192
            or type(seconds) is not int
            or not 0 < seconds <= 3600
            or issued.get("scope") != "slm:connect"
        ):
            raise ValueError("invalid_native_token")
        return replace(
            row,
            access_token=access,
            refresh_token=refresh,
            access_expires_ms=int(time.time() * 1000) + seconds * 1000,
        )

    async def provision(self, row: PendingEnrollment) -> dict:
        proof = DeviceSigner(row.private_key).proof(
            "POST", AUTH + "/owner/connections", token=row.access_token
        )
        value = await self._http(
            "/owner/connections",
            headers={"Authorization": "Bearer " + row.access_token, "DPoP": proof},
        )
        if (
            value.get("connection_id") != row.connection_id
            or value.get("profile_id") != row.profile
            or not isinstance(value.get("device_token"), str)
            or len(value["device_token"]) != 64
            or type(value.get("generation")) is not int
            or value["generation"] < 1
            or type(value.get("expires_at_ms")) is not int
            or value["expires_at_ms"] <= time.time() * 1000
        ):
            raise ValueError("invalid_device_delivery")
        return value

    async def revoke(self, row: PendingEnrollment) -> dict:
        proof = DeviceSigner(row.private_key).proof(
            "POST", AUTH + "/owner/revoke", token=row.access_token
        )
        return await self._http(
            "/owner/revoke", headers={"Authorization": "Bearer " + row.access_token, "DPoP": proof}
        )

    async def verify(self, row: PendingEnrollment) -> dict:
        proof = DeviceSigner(row.private_key).proof(
            "POST", AUTH + "/owner/verify", token=row.access_token
        )
        return await self._http(
            "/owner/verify", headers={"Authorization": "Bearer " + row.access_token, "DPoP": proof}
        )

    @staticmethod
    def _removal_error(path: str, status: int) -> str | None:
        """Only app removal distinguishes a stale list and an already-removed app."""
        if path != "/owner/apps/revoke":
            return None
        return {404: "not_found", 409: "version_conflict"}.get(status)

    async def list_apps(self, row: PendingEnrollment) -> dict:
        proof = DeviceSigner(row.private_key).proof(
            "POST", AUTH + "/owner/apps", token=row.access_token
        )
        return await self._http(
            "/owner/apps", headers={"Authorization": "Bearer " + row.access_token, "DPoP": proof}
        )

    async def revoke_app(
        self, row: PendingEnrollment, authorization_id: str, expected_version: int
    ) -> dict:
        proof = DeviceSigner(row.private_key).proof(
            "POST", AUTH + "/owner/apps/revoke", token=row.access_token
        )
        return await self._http(
            "/owner/apps/revoke",
            headers={"Authorization": "Bearer " + row.access_token, "DPoP": proof},
            json={"authorization_id": authorization_id, "expected_version": expected_version},
        )

    async def cancel_bootstrap(self, row: PendingEnrollment) -> dict:
        return await self._http(
            "/bootstrap/cancel", json={"connection_id": row.connection_id, "verifier": row.verifier}
        )
