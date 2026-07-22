# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Profile Routes
 - AGPL-3.0-or-later

Routes: /api/profiles, /api/profiles/{name}/switch,
        /api/profiles/create, DELETE /api/profiles/{name}

SQLite is the single source of truth for profiles. profiles.json
is kept in sync as a cache for backward compatibility.
"""
import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request

from superlocalmemory.server.route_mutations import authorize_route_mutation

from .helpers import (
    get_db_connection, validate_profile_name,
    ProfileSwitch, DB_PATH,
    sync_profiles, ensure_profile_in_db, ensure_profile_in_json,
    delete_profile_from_db,
    _load_profiles_json, _save_profiles_json,
)
from superlocalmemory.server.profile_runtime import (
    commit_daemon_profile_switch,
    get_profile_runtime,
    TransitionDrainTimeout,
)

logger = logging.getLogger("superlocalmemory.routes.profiles")
router = APIRouter()


def _internal_error(detail: str = "Internal server error") -> HTTPException:
    """SEC-H-02: log full traceback server-side; return a generic message to the client."""
    logger.exception("profiles route error")
    return HTTPException(status_code=500, detail=detail)

# WebSocket manager reference (set by ui_server.py at startup)
ws_manager = None


def _get_memory_count(profile: str) -> int:
    """Get memory count for a profile."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT COUNT(*) FROM atomic_facts WHERE profile_id = ?", (profile,),
            )
            count = cursor.fetchone()[0]
        except Exception:
            cursor.execute(
                "SELECT COUNT(*) FROM memories WHERE profile = ?", (profile,),
            )
            count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


@router.get("/api/profiles")
async def list_profiles(request: Request):
    """List available memory profiles (synced from SQLite + profiles.json)."""
    try:
        merged = sync_profiles()
        active = get_profile_runtime(request.app.state).snapshot.profile_id

        profiles = []
        for p in merged:
            # profile_id is the canonical key (PK, FK target, used by engine)
            pid = p.get('profile_id', p.get('name', ''))
            count = _get_memory_count(pid)
            profiles.append({
                "name": pid,
                "description": p.get('description', ''),
                "memory_count": count,
                "created_at": p.get('created_at', ''),
                "last_used": p.get('last_used', ''),
                "is_active": pid == active,
            })

        return {
            "profiles": profiles,
            "active_profile": active,
            "total_profiles": len(profiles),
        }

    except Exception:
        raise _internal_error("Profile list error")


@router.post("/api/profiles/{name}/switch")
async def switch_profile(name: str, request: Request):
    """Switch active memory profile (persists to both config stores)."""
    try:
        if not validate_profile_name(name):
            raise HTTPException(status_code=400, detail="Invalid profile name.")

        merged = sync_profiles()
        merged_ids = {p.get('profile_id', p.get('name', '')) for p in merged}

        if name not in merged_ids:
            available = ', '.join(sorted(merged_ids))
            raise HTTPException(
                status_code=404,
                detail=f"Profile '{name}' not found. Available: {available}",
            )

        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-profile-switch",
            profile_id=name,
        )
        # RBAC (C4): a logged-in user may only activate a workspace they belong
        # to — this is the read boundary. Switching sets the active tenant for
        # subsequent reads, so a non-member must not be able to enter it. The
        # machine owner (no user session) switches freely.
        from superlocalmemory.server.rbac_enforce import resolve_principal
        _principal = resolve_principal(request)
        if _principal.get("kind") == "user":
            _rbac = getattr(request.app.state, "rbac", None)
            if _rbac is not None and _rbac.get_role(_principal["user_id"], name) is None:
                raise HTTPException(
                    status_code=403,
                    detail="You are not a member of this workspace.",
                )
        runtime = get_profile_runtime(request.app.state)
        previous = runtime.snapshot.profile_id
        try:
            snapshot = await asyncio.to_thread(
                runtime.transition,
                name,
                lambda prior, target: commit_daemon_profile_switch(
                    request.app.state,
                    prior,
                    target,
                ),
            )
        except TransitionDrainTimeout as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc),
            )

        count = _get_memory_count(name)

        if ws_manager:
            await ws_manager.broadcast({
                "type": "profile_switched", "profile": name,
                "previous": previous, "memory_count": count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        authorization.complete()
        return {
            "success": True, "active_profile": name,
            "previous_profile": previous, "memory_count": count,
            "generation": snapshot.generation,
            "message": f"Switched to profile '{name}' ({count} memories).",
        }

    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Profile switch error")


@router.post("/api/profiles/create")
async def create_profile(body: ProfileSwitch, request: Request):
    """Create a new memory profile (writes to BOTH SQLite and profiles.json)."""
    try:
        name = body.profile_name
        if not validate_profile_name(name):
            raise HTTPException(status_code=400, detail="Invalid profile name")

        # Check both stores for duplicates
        merged = sync_profiles()
        merged_ids = {p.get('profile_id', p.get('name', '')) for p in merged}
        if name in merged_ids:
            raise HTTPException(status_code=409, detail=f"Profile '{name}' already exists")

        authorization = authorize_route_mutation(
            request,
            operation="update",
            source_agent_id="http-profile-create",
        )
        # RBAC (C3): creating a tenant is an administrative action.
        from superlocalmemory.access.rbac import Permission as _Perm
        from superlocalmemory.server.rbac_enforce import require_manage as _rbac_manage
        _principal = _rbac_manage(request)
        # Write to BOTH stores atomically
        desc = f'Memory profile: {name}'
        ensure_profile_in_db(name, desc)
        ensure_profile_in_json(name, desc)

        # A logged-in user who creates a workspace becomes its admin so they
        # can manage it immediately (profile_id == name here). The machine
        # owner needs no membership row (implicit root).
        if _principal.get("kind") == "user":
            rbac = getattr(request.app.state, "rbac", None)
            if rbac is not None:
                try:
                    rbac.set_membership(name, _principal["user_id"], "admin",
                                        added_by=_principal["username"])
                except Exception:
                    pass

        authorization.complete()
        return {"success": True, "profile": name, "message": f"Profile '{name}' created"}

    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Profile create error")


@router.delete("/api/profiles/{name}")
async def delete_profile(name: str, request: Request):
    """Delete a profile. Moves its memories to 'default'."""
    try:
        if name == 'default':
            raise HTTPException(status_code=400, detail="Cannot delete 'default' profile")

        merged = sync_profiles()
        merged_ids = {p.get('profile_id', p.get('name', '')) for p in merged}
        if name not in merged_ids:
            raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

        runtime = get_profile_runtime(request.app.state)
        if runtime.snapshot.profile_id == name:
            raise HTTPException(status_code=400, detail="Cannot delete active profile.")

        json_config = _load_profiles_json()

        authorization = authorize_route_mutation(
            request,
            operation="delete",
            source_agent_id="http-profile-delete",
            profile_id=name,
        )
        # RBAC (C3): deleting a tenant is administrative. Check MANAGE on the
        # profile being deleted (not the active one).
        from superlocalmemory.server.rbac_enforce import require_manage as _rbac_manage
        _rbac_manage(request, profile=name)
        # Move data to default before deleting (bypasses CASCADE)
        conn = get_db_connection()
        cursor = conn.cursor()
        moved = 0
        try:
            cursor.execute(
                "UPDATE atomic_facts SET profile_id = 'default' WHERE profile_id = ?",
                (name,),
            )
            moved = cursor.rowcount
        except Exception:
            pass
        try:
            cursor.execute(
                "UPDATE memories SET profile_id = 'default' WHERE profile_id = ?",
                (name,),
            )
            moved += cursor.rowcount
        except Exception:
            pass
        conn.commit()
        conn.close()

        # Delete from BOTH stores
        delete_profile_from_db(name)

        profiles = json_config.get('profiles', {})
        profiles.pop(name, None)
        json_config['profiles'] = profiles
        _save_profiles_json(json_config)

        authorization.complete()
        return {
            "success": True,
            "message": f"Profile '{name}' deleted. {moved} memories moved to 'default'.",
        }

    except HTTPException:
        raise
    except Exception:
        raise _internal_error("Profile delete error")
