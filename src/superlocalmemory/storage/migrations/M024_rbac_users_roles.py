# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — RBAC / teams (C3)

"""M024 — user identity + role-based access control tables (memory.db).

SLM's tenancy boundary is the profile (individual / team / company). Until now
the daemon had no *user* identity — access was machine-level (install token /
daemon capability). For team and company deployments an org needs real users,
each with a role on a profile, managed from the dashboard.

Adopts the Cognee reference model (Tenant -> User -> Role):
  * rbac_users        — a person/agent identity with a hashed password.
  * rbac_memberships  — (profile_id, user_id) -> role, the tenant/role grant.
  * rbac_sessions     — dashboard login sessions (hashed bearer token + expiry).

RBAC is additive: with zero users the daemon stays single-operator (the machine
operator over loopback is the implicit owner). Once users + memberships exist,
logged-in users are enforced against their role. Brand-new independent tables,
so this is a normal (pre-engine-init) migration with an idempotent apply().

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import sqlite3

NAME = "M024_rbac_users_roles"
DB_TARGET = "memory"

DDL = """
CREATE TABLE IF NOT EXISTS rbac_users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT DEFAULT '',
    password_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    created_by TEXT DEFAULT 'owner'
);
CREATE TABLE IF NOT EXISTS rbac_memberships (
    profile_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    added_at TEXT NOT NULL,
    added_by TEXT DEFAULT 'owner',
    PRIMARY KEY (profile_id, user_id)
);
CREATE TABLE IF NOT EXISTS rbac_sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rbac_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rbac_memberships_user
    ON rbac_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_rbac_sessions_user
    ON rbac_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_rbac_sessions_expiry
    ON rbac_sessions(expires_at);
"""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def apply(conn: sqlite3.Connection) -> None:
    """Create the RBAC tables. Idempotent (CREATE TABLE IF NOT EXISTS)."""
    conn.executescript(DDL)


def verify(conn: sqlite3.Connection) -> bool:
    """Applied once all RBAC tables exist."""
    return all(
        _table_exists(conn, t)
        for t in ("rbac_users", "rbac_memberships", "rbac_sessions",
                  "rbac_settings")
    )
