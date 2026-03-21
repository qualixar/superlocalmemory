# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""SuperLocalMemory V3 - Route Helpers
 - MIT License

Shared utilities for all route modules: DB connection, dict factory,
profile helper, validation, Pydantic models, config paths.
"""
import re
import json
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Version detection (shared — avoids circular import between ui.py ↔ v3_api.py)
# ---------------------------------------------------------------------------

def _get_version() -> str:
    """Read version from package.json / pyproject.toml / importlib."""
    try:
        import json as _json
        pkg_root = Path(__file__).resolve().parent.parent.parent.parent
        pkg_json = pkg_root / "package.json"
        if pkg_json.exists():
            with open(pkg_json) as f:
                v = _json.load(f).get("version", "")
                if v:
                    return v
    except Exception:
        pass
    try:
        import tomllib
        toml_path = Path(__file__).resolve().parent.parent.parent.parent / "pyproject.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                return tomllib.load(f)["project"]["version"]
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("superlocalmemory")
    except Exception:
        pass
    return "unknown"


SLM_VERSION = _get_version()

# V3 paths (migrated from ~/.claude-memory to ~/.superlocalmemory)
MEMORY_DIR = Path.home() / ".superlocalmemory"
DB_PATH = MEMORY_DIR / "memory.db"
UI_DIR = Path(__file__).parent.parent / "ui"
PROFILES_DIR = MEMORY_DIR / "profiles"


def get_engine_lazy(app_state):
    """Get or lazily initialize the V3 engine. Returns engine or None."""
    engine = getattr(app_state, "engine", None)
    if engine is not None:
        return engine
    if getattr(app_state, "_engine_init_attempted", False):
        return None
    try:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        config = SLMConfig.load()
        engine = MemoryEngine(config)
        engine.initialize()
        app_state.engine = engine
        app_state._engine_init_attempted = True
        return engine
    except Exception:
        app_state._engine_init_attempted = True
        return None


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Memory database not found. Run 'slm init' to initialize."
        )
    return sqlite3.connect(str(DB_PATH))


def dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict:
    """Convert SQLite row to dictionary."""
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row))


def get_active_profile() -> str:
    """Read the active profile from profiles.json. Falls back to 'default'."""
    config_file = MEMORY_DIR / "profiles.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                pconfig = json.load(f)
            return pconfig.get('active_profile', 'default')
        except (json.JSONDecodeError, IOError):
            pass
    return 'default'


def validate_profile_name(name: str) -> bool:
    """Validate profile name (alphanumeric, underscore, hyphen only)."""
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))


# ============================================================================
# Profile Sync — SQLite as single source of truth
# ============================================================================


def ensure_profile_in_db(name: str, description: str = "") -> None:
    """Ensure a profile row exists in SQLite (idempotent)."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name, description) "
            "VALUES (?, ?, ?)",
            (name, name, description or f"Memory profile: {name}"),
        )
        conn.commit()
    finally:
        conn.close()


def ensure_profile_in_json(name: str, description: str = "") -> None:
    """Ensure a profile entry exists in profiles.json (idempotent)."""
    from datetime import datetime
    config_file = MEMORY_DIR / "profiles.json"
    config = _load_profiles_json()
    profiles = config.get('profiles', {})
    if name not in profiles:
        profiles[name] = {
            'name': name,
            'description': description or f'Memory profile: {name}',
            'created_at': datetime.now().isoformat(),
            'last_used': None,
        }
        config['profiles'] = profiles
        _save_profiles_json(config)


def sync_profiles() -> list[dict]:
    """Reconcile SQLite and profiles.json. Returns merged profile list.

    SQLite is the source of truth. Uses ``profile_id`` (not ``name``)
    as the canonical key because profile_id is the PK referenced by
    every FK in the database.
    """
    db_profiles = _get_db_profiles()
    json_config = _load_profiles_json()
    json_profiles = json_config.get('profiles', {})

    # profile_id is the canonical key (PK in SQLite, FK target everywhere)
    db_ids = {p['profile_id'] for p in db_profiles}
    json_keys = set(json_profiles.keys())

    changed = False

    # JSON-only → add to SQLite (fixes Dashboard-created profiles)
    for key in json_keys - db_ids:
        ensure_profile_in_db(key, json_profiles[key].get('description', ''))

    # SQLite-only → add to profiles.json (fixes CLI-created profiles)
    for pid in db_ids - json_keys:
        db_entry = next(p for p in db_profiles if p['profile_id'] == pid)
        json_profiles[pid] = {
            'name': pid,
            'description': db_entry.get('description', ''),
            'created_at': db_entry.get('created_at', ''),
            'last_used': db_entry.get('last_used'),
        }
        changed = True

    if changed:
        json_config['profiles'] = json_profiles
        _save_profiles_json(json_config)

    # Return merged list from SQLite (now authoritative)
    return _get_db_profiles()


def set_active_profile_everywhere(name: str) -> None:
    """Persist the active profile to BOTH profiles.json and config.json."""
    # profiles.json
    config = _load_profiles_json()
    config['active_profile'] = name
    _save_profiles_json(config)

    # config.json (read by Engine/MCP on startup)
    config_path = MEMORY_DIR / "config.json"
    cfg = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    cfg['active_profile'] = name
    config_path.write_text(json.dumps(cfg, indent=2))


def delete_profile_from_db(name: str) -> None:
    """Delete a profile row from SQLite. ON DELETE CASCADE handles child rows."""
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("DELETE FROM profiles WHERE profile_id = ?", (name,))
        conn.commit()
    finally:
        conn.close()


def _get_db_profiles() -> list[dict]:
    """Read all profiles from SQLite."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT profile_id, name, description, created_at, last_used "
            "FROM profiles ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _load_profiles_json() -> dict:
    """Load profiles.json config (Dashboard dict format)."""
    config_file = MEMORY_DIR / "profiles.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            # Handle ProfileManager array format → convert to dict format
            if isinstance(data.get('profiles'), list):
                converted = {}
                for p in data['profiles']:
                    n = p.get('name', '')
                    if n:
                        converted[n] = p
                data['profiles'] = converted
                if 'active' in data and 'active_profile' not in data:
                    data['active_profile'] = data.pop('active')
            return data
        except (json.JSONDecodeError, IOError):
            pass
    return {
        'profiles': {'default': {'name': 'default', 'description': 'Default memory profile'}},
        'active_profile': 'default',
    }


def _save_profiles_json(config: dict) -> None:
    """Save profiles.json config."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    config_file = MEMORY_DIR / "profiles.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================================================
# Pydantic Models (shared across routes)
# ============================================================================

class SearchRequest(BaseModel):
    """Advanced search request model."""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    category: Optional[str] = None
    project_name: Optional[str] = None
    cluster_id: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class ProfileSwitch(BaseModel):
    """Profile switching request."""
    profile_name: str = Field(..., min_length=1, max_length=50)


class BackupConfigRequest(BaseModel):
    """Backup configuration update request."""
    interval_hours: Optional[int] = Field(None, ge=1, le=8760)
    max_backups: Optional[int] = Field(None, ge=1, le=100)
    enabled: Optional[bool] = None
