"""A small app with the real role engine: an owner (no login), an admin, a member and a viewer."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.access.rbac import RbacEngine
from superlocalmemory.server.routes import features, media, sources
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.migrations import M024_rbac_users_roles as m024

LOCAL = ("127.0.0.1", 50000)
PW = "pw-123456789"


class _Hooks:
    def run_pre(self, name, payload):
        return None


class Roles:
    """``headers(role)`` gives the session header of a user holding that role on every profile here."""

    def __init__(self, tmp_path: Path, *, with_users: bool = True) -> None:
        db_path = tmp_path / "memory.db"
        self.db = DatabaseManager(db_path)
        self.db.initialize(schema)
        conn = sqlite3.connect(str(db_path))
        m024.apply(conn)
        conn.commit()
        conn.close()
        self.db = DatabaseManager(db_path)
        for pid in ("p2",):
            self.db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)", (pid, pid))
        self.rbac = RbacEngine(str(db_path))
        self._sessions: dict[str, str] = {}
        if with_users:
            for role in ("admin", "member", "viewer"):
                user = self.rbac.create_user(role, PW, role)
                for pid in ("default", "p2"):
                    self.rbac.set_membership(pid, user["user_id"], role)
                self._sessions[role] = self.rbac.create_session(user["user_id"])
            outsider = self.rbac.create_user("outsider", PW, "outsider")
            self._sessions["outsider"] = self.rbac.create_session(outsider["user_id"])
        app = FastAPI()
        app.state.engine = SimpleNamespace(
            _profile_id="default", _config=SimpleNamespace(pii_redaction=False), _db=self.db, _hooks=_Hooks())
        app.state.rbac = self.rbac
        app.state.canonical_remember_runtime = object()

        @app.middleware("http")
        async def _actor(request, call_next):
            request.state.authenticated_actor = "authenticated:test"
            return await call_next(request)

        for module in (media, sources, features):
            app.include_router(module.router)
        self.client = TestClient(app, client=LOCAL)

    def headers(self, role: str) -> dict[str, str]:
        return {"X-SLM-User-Session": self._sessions[role]} if role in self._sessions else {}
