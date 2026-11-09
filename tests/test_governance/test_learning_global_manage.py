"""Whole learning.db operations require installation-wide management."""
# ruff: noqa: F811
import sqlite3

import pytest
from starlette.exceptions import HTTPException
from starlette.requests import Request

from tests.test_governance.test_role_surface_matrix_rest import (
    _company,
    _global_admin,
)


@pytest.mark.parametrize("operation", ["backup", "migrate-legacy"])
def test_learning_store_operations_require_global_manage(matrix, monkeypatch, tmp_path, operation):
    from superlocalmemory.learning import legacy_migration
    from superlocalmemory.server.routes import learning

    db = tmp_path / "learning.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE signals (profile_id TEXT)")
        conn.executemany("INSERT INTO signals VALUES (?)", [("default",), ("sales",)])
    monkeypatch.setattr(learning, "LEARNING_DB", db)
    monkeypatch.setattr(learning, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(learning, "LEARNING_AVAILABLE", True)
    migrated = []
    monkeypatch.setattr(legacy_migration, "migrate_legacy_feedback",
                        lambda path: migrated.append(path) or {"copied": 2})
    app, client, who, ids = matrix
    _company(app)
    _global_admin(app, who, ids)
    path = "/api/learning/" + operation
    handler = learning.learning_backup if operation == "backup" else learning.learning_migrate_legacy
    # Exercise handler authority independently of the middleware floor.
    request = Request({"type": "http", "method": "POST", "path": path,
                       "app": app, "client": ("127.0.0.1", 9911),
                       "headers": [(key.lower().encode(), value.encode())
                                   for key, value in who["admin_b"].items()]})
    with pytest.raises(HTTPException) as denied:
        handler(request)
    assert denied.value.status_code == 403
    assert client.post(path, headers=who["admin_b"]).status_code == 403
    assert not migrated and not list(tmp_path.glob("learning.db.backup_*"))
    allowed = client.post(path, headers=who["root_admin"])
    assert allowed.status_code == 200 and allowed.json()["success"] is True
    if operation == "backup":
        assert len(list(tmp_path.glob("learning.db.backup_*"))) == 1
    else:
        assert migrated == [db]
