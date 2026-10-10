"""Existing-dashboard enrollment endpoints; off without a configured service."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from superlocalmemory.remote_connections.journal import JournalConflict
from superlocalmemory.remote_connections.service import RemoteConnectionService
from superlocalmemory.server.loopback import is_loopback
from superlocalmemory.server.rbac_enforce import require_manage
from superlocalmemory.server.write_identity import require_write_actor

router = APIRouter(prefix="/api/v3/connections", tags=["connections"])
logger = logging.getLogger(__name__)


class Permissions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    read: StrictBool
    write: StrictBool
    correction: StrictBool
    session: StrictBool

    @model_validator(mode="after")
    def validate_scope(self):
        if not self.read or (self.correction and not self.write):
            raise ValueError("invalid_consent")
        return self


class ConnectionIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    host: Literal["muse", "chatgpt", "claude_web", "claude_code_web", "composio", "other_mcp"]
    profile_id: str
    remote_opt_in: StrictBool
    permissions: Permissions

    @field_validator("remote_opt_in")
    @classmethod
    def explicit_opt_in(cls, value: bool) -> bool:
        if not value:
            raise ValueError("remote_opt_in_required")
        return value


def _context(request: Request, *, mutation: bool = False) -> tuple[str, str]:
    peer = request.client.host if request.client else ""
    if not is_loopback(peer) or not is_loopback(request.url.hostname or ""):
        raise HTTPException(403, "local_dashboard_required")
    origin = request.headers.get("Origin")
    if origin and origin != str(request.url).split(request.url.path)[0]:
        raise HTTPException(403, "same_origin_required")
    if request.headers.get("Sec-Fetch-Site") in {"cross-site", "same-site"}:
        raise HTTPException(403, "same_origin_required")
    if mutation:
        require_write_actor(
            request,
            getattr(request.app.state, "daemon_descriptor", None),
            actor_kind="remote-enrollment",
        )
    from superlocalmemory.server.profile_runtime import current_request_profile, get_profile_runtime

    profile = (
        current_request_profile() or get_profile_runtime(request.app.state).snapshot.profile_id
    )
    principal = require_manage(request, profile=profile)
    return str(principal["user_id"]), profile


def _service(request: Request) -> RemoteConnectionService | None:
    service = getattr(request.app.state, "remote_connections", None)
    if service is not None and not isinstance(service, RemoteConnectionService):
        raise HTTPException(503, "connection_service_unavailable")
    return service


class CancellationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str
    expected_version: StrictInt = Field(ge=0)


@router.post("/{connection_id}/cancel")
async def cancel_connection(request: Request, connection_id: str, intent: CancellationIntent):
    owner, profile = _context(request, mutation=True)
    if intent.profile_id != profile:
        raise HTTPException(409, "profile_changed")
    service = _service(request)
    if service is None:
        raise HTTPException(503, "connection_service_unavailable")
    try:
        return await service.cancel(owner, profile, connection_id, intent.expected_version)
    except JournalConflict as exc:
        code = str(exc)
        raise HTTPException(404 if code == "not_found" else 409, code) from None
    except ValueError:
        raise HTTPException(400, "invalid_enrollment_request") from None
    except Exception:
        logger.error("remote_connection_cancellation_unavailable")
        raise HTTPException(503, "connection_service_unavailable") from None


class RestartIntent(CancellationIntent):
    remote_opt_in: StrictBool

    @model_validator(mode="after")
    def explicit_restart(self):
        if self.remote_opt_in is not True:
            raise ValueError("remote opt-in required")
        return self


@router.post("/{connection_id}/restart")
async def restart_connection(request: Request, connection_id: str, intent: RestartIntent):
    owner, profile = _context(request, mutation=True)
    if intent.profile_id != profile:
        raise HTTPException(409, "profile_changed")
    service = _service(request)
    if service is None or not hasattr(service, "restart"):
        raise HTTPException(503, "connection_service_unavailable")
    try:
        return await service.restart(owner, profile, connection_id, intent.expected_version)
    except JournalConflict as exc:
        code = str(exc)
        raise HTTPException(404 if code == "not_found" else 409, code) from None
    except ValueError:
        raise HTTPException(400, "invalid_enrollment_request") from None
    except Exception:
        logger.error("remote_connection_restart_unavailable")
        raise HTTPException(503, "connection_service_unavailable") from None


def _apps_runtime(request: Request):
    runtime = getattr(request.app.state, "remote_connection_runtime", None)
    if runtime is None or not hasattr(runtime, "list_apps"):
        raise HTTPException(503, "connection_service_unavailable")
    return runtime


def _apps_error(code: str) -> HTTPException:
    # Fixed codes only; provider or gateway text never reaches the dashboard.
    if code == "not_found":
        return HTTPException(404, "not_found")
    if code == "version_conflict":
        return HTTPException(409, "version_conflict")
    return HTTPException(503, "apps_unavailable")


@router.get("/{connection_id}/apps")
async def connected_apps(request: Request, connection_id: str):
    """Apps that currently hold access to this computer's memory (Connected apps)."""
    owner, profile = _context(request)
    runtime = _apps_runtime(request)
    try:
        return await runtime.list_apps(owner, profile, connection_id)
    except ValueError as exc:
        raise _apps_error(str(exc)) from None
    except Exception:
        logger.error("remote_connected_apps_unavailable")
        raise HTTPException(503, "apps_unavailable") from None


class AppRemovalIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str
    expected_version: StrictInt = Field(ge=1)


@router.post("/{connection_id}/apps/{authorization_id}/revoke")
async def remove_connected_app(
    request: Request, connection_id: str, authorization_id: str, intent: AppRemovalIntent
):
    owner, profile = _context(request, mutation=True)
    if intent.profile_id != profile:
        raise HTTPException(409, "profile_changed")
    runtime = _apps_runtime(request)
    try:
        return await runtime.revoke_app(
            owner, profile, connection_id, authorization_id, intent.expected_version
        )
    except ValueError as exc:
        raise _apps_error(str(exc)) from None
    except Exception:
        logger.error("remote_connected_app_removal_unavailable")
        raise HTTPException(503, "apps_unavailable") from None


class AbilityIntent(BaseModel):
    """The second yes: let every app on this connection use bots (mesh) or pictures (media).

    The first yes is the owner's approval of each app at sign-in; both are needed."""
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_id: str
    ability: Literal["mesh", "media"]
    allow: StrictBool


def _abilities_runtime(request: Request):
    runtime = _apps_runtime(request)
    if not hasattr(runtime, "key_abilities"):
        raise HTTPException(503, "connection_service_unavailable")
    return runtime


@router.get("/{connection_id}/abilities")
async def connection_abilities(request: Request, connection_id: str):
    """Whether apps on this connection may message your other bots and save pictures."""
    owner, profile = _context(request)
    runtime = _abilities_runtime(request)
    try:
        return await runtime.key_abilities(owner, profile, connection_id)
    except ValueError as exc:
        raise _apps_error(str(exc)) from None
    except Exception:
        logger.error("remote_connection_abilities_unavailable")
        raise HTTPException(503, "apps_unavailable") from None


@router.post("/{connection_id}/abilities")
async def set_connection_ability(request: Request, connection_id: str, intent: AbilityIntent):
    owner, profile = _context(request, mutation=True)
    if intent.profile_id != profile:
        raise HTTPException(409, "profile_changed")
    runtime = _abilities_runtime(request)
    try:
        return await runtime.set_key_ability(owner, profile, connection_id, intent.ability, intent.allow)
    except ValueError as exc:
        raise _apps_error(str(exc)) from None
    except Exception:
        logger.error("remote_connection_ability_change_unavailable")
        raise HTTPException(503, "apps_unavailable") from None


@router.get("/status")
async def connection_status(request: Request):
    owner, profile = _context(request)
    service = _service(request)
    if service is None:
        return {
            "available": False,
            "installation_id": "",
            "current_profile": profile,
            "hosts": [],
            "connections": [],
        }
    try:
        runtime = getattr(request.app.state, "remote_connection_runtime", None)
        if runtime is not None:
            await runtime.resume(owner, profile)
        return await asyncio.to_thread(service.status, owner, profile)
    except Exception:
        logger.error("remote_connection_status_unavailable")
        raise HTTPException(503, "connection_service_unavailable") from None


@router.post("/initiate")
async def initiate_connection(request: Request, intent: ConnectionIntent):
    owner, profile = _context(request, mutation=True)
    if intent.profile_id != profile:
        raise HTTPException(409, "profile_changed")
    service = _service(request)
    if service is None:
        raise HTTPException(503, "connection_service_unavailable")
    key = request.headers.get("Idempotency-Key", "")
    try:
        return await service.initiate(owner, profile, key, intent.model_dump())
    except JournalConflict as exc:
        code = str(exc)
        status = 403 if code == "host_unavailable" else 429 if code == "capacity_exhausted" else 409
        raise HTTPException(status, code) from None
    except ValueError:
        raise HTTPException(400, "invalid_enrollment_request") from None
    except Exception:
        logger.error("remote_connection_enrollment_unavailable")
        raise HTTPException(503, "connection_service_unavailable") from None


@router.get("/callback")
async def connection_callback(request: Request):
    """OAuth return: the secret state delegates the already authorized local intent."""
    query = request.query_params
    states, codes = query.getlist("state"), query.getlist("code")
    # Uvicorn constructs its access-log URL from this scope after response start.
    # Never put OAuth codes or state values into that access log.
    request.scope["query_string"] = b""
    peer = request.client.host if request.client else ""
    if not is_loopback(peer) or not is_loopback(request.url.hostname or ""):
        raise HTTPException(403, "local_callback_required")
    runtime = getattr(request.app.state, "remote_connection_runtime", None)
    if runtime is None:
        raise HTTPException(503, "connection_service_unavailable")
    if len(states) != 1 or len(codes) != 1:
        raise HTTPException(400, "invalid_callback")
    try:
        await runtime.callback(states[0], codes[0])
    except ValueError:
        return HTMLResponse(
            "<!doctype html><title>SuperLocalMemory</title>"
            "<h1>Web connection could not be enabled</h1>"
            "<p>Return to your SLM dashboard, refresh the connection status and try again.</p>",
            status_code=400,
            headers={"Cache-Control": "no-store"},
        )
    except Exception:
        logger.error("remote_connection_callback_unavailable")
        raise HTTPException(503, "connection_service_unavailable") from None
    # Do not leave the one-use OAuth code in a refreshable success-page URL.
    return RedirectResponse("/#apps-pane", status_code=303, headers={"Cache-Control": "no-store"})
