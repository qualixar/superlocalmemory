# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Profile Routes
 - MIT License

Routes: /api/profiles, /api/profiles/{name}/switch,
        /api/profiles/create, DELETE /api/profiles/{name}

SQLite is the single source of truth for profiles. profiles.json
is kept in sync as a cache for backward compatibility.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from .helpers import (
    get_db_connection, validate_profile_name,
    ProfileSwitch, DB_PATH,
    sync_profiles, ensure_profile_in_db, ensure_profile_in_json,
    set_active_profile_everywhere, delete_profile_from_db,
    _load_profiles_json, _save_profiles_json,
)

logger = logging.getLogger("superlocalmemory.routes.profiles")
router = APIRouter()

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
async def list_profiles():
    """List available memory profiles (synced from SQLite + profiles.json)."""
    try:
        merged = sync_profiles()
        json_config = _load_profiles_json()
        active = json_config.get('active_profile', 'default')

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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile list error: {str(e)}")


@router.post("/api/profiles/{name}/switch")
async def switch_profile(name: str):
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

        previous = _load_profiles_json().get('active_profile', 'default')
        set_active_profile_everywhere(name)

        # Update last_used in profiles.json
        json_config = _load_profiles_json()
        if name in json_config.get('profiles', {}):
            json_config['profiles'][name]['last_used'] = datetime.now().isoformat()
            _save_profiles_json(json_config)

        count = _get_memory_count(name)

        if ws_manager:
            await ws_manager.broadcast({
                "type": "profile_switched", "profile": name,
                "previous": previous, "memory_count": count,
                "timestamp": datetime.now().isoformat(),
            })

        return {
            "success": True, "active_profile": name,
            "previous_profile": previous, "memory_count": count,
            "message": f"Switched to profile '{name}' ({count} memories).",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile switch error: {str(e)}")


@router.post("/api/profiles/create")
async def create_profile(body: ProfileSwitch):
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

        # Write to BOTH stores atomically
        desc = f'Memory profile: {name}'
        ensure_profile_in_db(name, desc)
        ensure_profile_in_json(name, desc)

        return {"success": True, "profile": name, "message": f"Profile '{name}' created"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile create error: {str(e)}")


@router.delete("/api/profiles/{name}")
async def delete_profile(name: str):
    """Delete a profile. Moves its memories to 'default'."""
    try:
        if name == 'default':
            raise HTTPException(status_code=400, detail="Cannot delete 'default' profile")

        merged = sync_profiles()
        merged_ids = {p.get('profile_id', p.get('name', '')) for p in merged}
        if name not in merged_ids:
            raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")

        json_config = _load_profiles_json()
        if json_config.get('active_profile') == name:
            raise HTTPException(status_code=400, detail="Cannot delete active profile.")

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

        return {
            "success": True,
            "message": f"Profile '{name}' deleted. {moved} memories moved to 'default'.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile delete error: {str(e)}")
