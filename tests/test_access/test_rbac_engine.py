# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""RbacEngine — users, sessions, memberships, and role enforcement."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.access.rbac import (
    Permission, RbacEngine, RbacError, Role, permissions_for_role,
)
from superlocalmemory.storage.migrations import M024_rbac_users_roles as m024


@pytest.fixture
def engine(tmp_path):
    db = tmp_path / "rbac.db"
    conn = sqlite3.connect(str(db))
    m024.apply(conn)
    assert m024.verify(conn) is True
    conn.close()
    return RbacEngine(str(db))


# -- role matrix ----------------------------------------------------------

def test_role_permission_matrix():
    assert permissions_for_role(Role.ADMIN) == frozenset(Permission)
    assert Permission.MANAGE not in permissions_for_role(Role.MEMBER)
    assert Permission.WRITE in permissions_for_role(Role.MEMBER)
    assert permissions_for_role(Role.VIEWER) == frozenset({Permission.READ})


# -- users + passwords ----------------------------------------------------

def test_create_and_verify_user(engine):
    assert engine.user_count() == 0
    u = engine.create_user("alice", "s3cret-password", display_name="Alice")
    assert engine.user_count() == 1
    assert engine.verify_credentials("alice", "s3cret-password")["user_id"] == u["user_id"]
    assert engine.verify_credentials("alice", "wrong") is None


def test_password_not_stored_plaintext(engine, tmp_path):
    engine.create_user("bob", "hunter2-hunter2")
    conn = sqlite3.connect(engine._db_path)
    stored = conn.execute("SELECT password_hash FROM rbac_users WHERE username='bob'").fetchone()[0]
    conn.close()
    assert "hunter2" not in stored
    assert stored.startswith("scrypt$")


def test_short_password_rejected(engine):
    with pytest.raises(RbacError):
        engine.create_user("carol", "short")


def test_duplicate_username_rejected(engine):
    engine.create_user("dave", "password-1234")
    with pytest.raises(RbacError):
        engine.create_user("dave", "password-5678")


def test_disabled_user_cannot_authenticate(engine):
    u = engine.create_user("erin", "password-1234")
    engine.set_status(u["user_id"], "disabled")
    assert engine.verify_credentials("erin", "password-1234") is None


# -- sessions -------------------------------------------------------------

def test_session_lifecycle(engine):
    u = engine.create_user("frank", "password-1234")
    token = engine.create_session(u["user_id"])
    resolved = engine.resolve_session(token)
    assert resolved["user_id"] == u["user_id"]
    engine.revoke_session(token)
    assert engine.resolve_session(token) is None


def test_expired_session_rejected(engine):
    u = engine.create_user("grace", "password-1234")
    token = engine.create_session(u["user_id"], ttl_hours=-1)  # already expired
    assert engine.resolve_session(token) is None


def test_password_change_invalidates_sessions(engine):
    u = engine.create_user("heidi", "password-1234")
    token = engine.create_session(u["user_id"])
    engine.set_password(u["user_id"], "new-password-5678")
    assert engine.resolve_session(token) is None


def test_only_token_hash_is_stored(engine):
    u = engine.create_user("ivan", "password-1234")
    token = engine.create_session(u["user_id"])
    conn = sqlite3.connect(engine._db_path)
    rows = [r[0] for r in conn.execute("SELECT token_hash FROM rbac_sessions")]
    conn.close()
    assert token not in rows  # raw token never persisted


# -- memberships + authorization -----------------------------------------

def test_membership_role_enforcement(engine):
    admin = engine.create_user("admin1", "password-1234")
    member = engine.create_user("member1", "password-1234")
    viewer = engine.create_user("viewer1", "password-1234")

    engine.set_membership("acme", admin["user_id"], "admin")
    engine.set_membership("acme", member["user_id"], "member")
    engine.set_membership("acme", viewer["user_id"], "viewer")

    # admin: everything
    for perm in Permission:
        assert engine.has_permission(admin["user_id"], "acme", perm)
    # member: read/write/share but not delete/manage
    assert engine.has_permission(member["user_id"], "acme", Permission.WRITE)
    assert not engine.has_permission(member["user_id"], "acme", Permission.DELETE)
    assert not engine.has_permission(member["user_id"], "acme", Permission.MANAGE)
    # viewer: read only
    assert engine.has_permission(viewer["user_id"], "acme", Permission.READ)
    assert not engine.has_permission(viewer["user_id"], "acme", Permission.WRITE)


def test_no_membership_is_deny_by_default(engine):
    u = engine.create_user("stranger", "password-1234")
    # No membership on 'acme' → no permission at all, even read.
    assert not engine.has_permission(u["user_id"], "acme", Permission.READ)


def test_membership_isolated_per_profile(engine):
    u = engine.create_user("multi", "password-1234")
    engine.set_membership("acme", u["user_id"], "admin")
    engine.set_membership("globex", u["user_id"], "viewer")
    assert engine.has_permission(u["user_id"], "acme", Permission.DELETE)
    assert not engine.has_permission(u["user_id"], "globex", Permission.DELETE)
    assert engine.has_permission(u["user_id"], "globex", Permission.READ)


def test_invalid_role_rejected(engine):
    u = engine.create_user("x", "password-1234")
    with pytest.raises(RbacError):
        engine.set_membership("acme", u["user_id"], "superuser")


def test_role_change_and_removal(engine):
    u = engine.create_user("y", "password-1234")
    engine.set_membership("acme", u["user_id"], "viewer")
    assert engine.get_role(u["user_id"], "acme") == Role.VIEWER
    engine.set_membership("acme", u["user_id"], "admin")  # upsert
    assert engine.get_role(u["user_id"], "acme") == Role.ADMIN
    engine.remove_membership("acme", u["user_id"])
    assert engine.get_role(u["user_id"], "acme") is None


def test_delete_user_cascades(engine):
    u = engine.create_user("z", "password-1234")
    engine.set_membership("acme", u["user_id"], "admin")
    token = engine.create_session(u["user_id"])
    engine.delete_user(u["user_id"])
    assert engine.get_user(u["user_id"]) is None
    assert engine.get_role(u["user_id"], "acme") is None
    assert engine.resolve_session(token) is None
