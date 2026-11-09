# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Turning on per-user access must mean the same thing however the write arrives.

THE DEFECT

"Company mode" was two independent settings that had never been joined up: a
``deployment`` value in a config file, which the admission gate read, and a
``require_login`` policy in the workspace's own settings, which the dashboard
toggle writes and which the HTTP routes read. Turning company mode on from the
dashboard therefore changed what HTTP allowed and changed nothing anywhere else.

Measured on a copy of a real store, with the policy on, two users configured,
and the viewer's role granting READ only: a write over the agent transport
resolved to ``local-operator`` holding role ``owner`` and succeeded, while the
identical write over HTTP returned 401. Nothing was missing a role check. The
role check ran, against an actor built on the belief that the workspace was
still single-user.

So these tests are about one property: **either switch means a login is
required, on every transport**, and an unidentified caller fails closed.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from superlocalmemory.core.actor_context import ActorRole
from superlocalmemory.core.admission import (
    OperationKind,
    _company_mode_active,
    _session_principal,
    admits,
)


class _Deployment:
    def __init__(self, enterprise: bool) -> None:
        self.is_enterprise = enterprise


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A store with the role tables, and the data root pointed at it."""
    from superlocalmemory.infra.data_root import canonical_data_root
    from superlocalmemory.storage import schema

    # Ask where the data root actually resolves rather than assuming: the suite
    # already pins it per test via SLM_DATA_DIR, which outranks the other
    # aliases, so writing the store anywhere else leaves the code under test
    # reading an empty directory and every assertion below passing or failing
    # for the wrong reason.
    root = canonical_data_root()
    root.mkdir(parents=True, exist_ok=True)
    db = root / "memory.db"
    connection = sqlite3.connect(db)
    schema.create_all_tables(connection)
    # The role tables arrive with migrations, not with the base schema, so a
    # store built from create_all_tables alone has no rbac_users and every call
    # below would fail on a missing table rather than on a policy decision.
    from superlocalmemory.storage.migrations import (
        M024_rbac_users_roles as _rbac_tables,
    )
    from superlocalmemory.storage.migrations import (
        M026_rbac_memberships_fk as _rbac_fk,
    )
    for statement in _rbac_tables.DDL.split(";"):
        if statement.strip():
            connection.execute(statement)
    for statement in getattr(_rbac_fk, "DDL", "").split(";"):
        if statement.strip():
            try:
                connection.execute(statement)
            except sqlite3.Error:
                pass
    connection.commit()
    connection.close()

    monkeypatch.delenv("SLM_USER_SESSION", raising=False)

    from superlocalmemory.access.rbac import RbacEngine

    rbac = RbacEngine(str(db))
    admin = rbac.create_user("admin-user", "not-a-real-secret", "Admin")
    viewer = rbac.create_user("viewer-user", "not-a-real-secret", "Viewer")
    outsider = rbac.create_user("outsider", "not-a-real-secret", "Outsider")
    rbac.set_membership("default", admin["user_id"], "admin")
    rbac.set_membership("default", viewer["user_id"], "viewer")
    return {
        "rbac": rbac, "admin": admin, "viewer": viewer, "outsider": outsider,
    }


@admits(OperationKind.REMEMBER)
async def _write(content: str = "a memory") -> dict:
    return {"success": True, "wrote": content}


def _attempt() -> dict:
    return asyncio.run(_write())


class TestEitherSwitchMeansCompanyMode:
    def test_the_dashboard_toggle_alone_turns_it_on(self, workspace) -> None:
        """This is the half that was not being read."""
        assert _company_mode_active(_Deployment(False)) is False

        workspace["rbac"].set_require_login(True)

        assert _company_mode_active(_Deployment(False)) is True

    def test_the_config_file_alone_turns_it_on(self, workspace) -> None:
        assert workspace["rbac"].require_login() is False
        assert _company_mode_active(_Deployment(True)) is True

    def test_an_unreadable_policy_is_treated_as_requiring_a_login(
        self, workspace, monkeypatch,
    ) -> None:
        """A policy we cannot read is not permission to skip it."""
        import superlocalmemory.core.admission as admission

        class _Broken:
            def require_login(self):
                raise sqlite3.OperationalError("policy table is locked")

        monkeypatch.setattr(admission, "_rbac_engine", lambda: _Broken())

        assert _company_mode_active(_Deployment(False)) is True

    def test_a_workspace_with_no_role_store_is_personal(
        self, workspace, monkeypatch,
    ) -> None:
        """No roles configured means no roles to enforce, not a locked door."""
        import superlocalmemory.core.admission as admission

        monkeypatch.setattr(admission, "_rbac_engine", lambda: None)

        assert _company_mode_active(_Deployment(False)) is False


class TestPersonalModeGainsNoFriction:
    """The single-operator case must be exactly as it was."""

    def test_a_write_with_no_session_is_allowed(self, workspace) -> None:
        assert workspace["rbac"].require_login() is False
        assert _attempt()["success"] is True

    def test_a_stray_session_token_changes_nothing_without_accounts(
        self, workspace, monkeypatch,
    ) -> None:
        """No team accounts at all: a token cannot name anyone, so it is ignored."""
        import os

        import superlocalmemory.core.admission as admission

        monkeypatch.setattr(admission, "_accounts_exist", lambda: False)
        os.environ["SLM_USER_SESSION"] = "whatever"
        try:
            assert _attempt()["success"] is True
        finally:
            os.environ.pop("SLM_USER_SESSION", None)

    def test_an_ended_session_is_refused_not_promoted_to_owner(self, workspace) -> None:
        """OPS-24 on this transport: with accounts and login optional, a session
        that names nobody (signed out, expired, forged) is refused. Before
        4.1.25 it was ignored and the caller wrote as the machine owner."""
        import os

        os.environ["SLM_USER_SESSION"] = "whatever"
        try:
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)
        assert result["success"] is False
        assert result["reason"] == "authentication_required"

    def test_a_viewer_session_is_held_to_viewer_when_login_is_optional(
        self, workspace,
    ) -> None:
        import os

        os.environ["SLM_USER_SESSION"] = workspace["rbac"].create_session(
            workspace["viewer"]["user_id"])
        try:
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)
        assert result["success"] is False
        assert result["reason"] == "insufficient_roles"

    def test_no_session_is_still_the_owner_when_login_is_optional(self, workspace) -> None:
        assert _attempt()["success"] is True


class TestCompanyModeOnTheAgentTransport:
    """The four cases, matching what HTTP already does."""

    @pytest.fixture(autouse=True)
    def _company(self, workspace):
        workspace["rbac"].set_require_login(True)
        return workspace

    def test_an_unidentified_write_is_refused(self, workspace) -> None:
        """This is the write that used to succeed as the machine owner."""
        result = _attempt()

        assert result["success"] is False
        assert result["reason"] == "authentication_required"

    def test_a_token_that_resolves_to_nobody_is_refused(self, workspace) -> None:
        """Fails closed: an expired or forged token is not a principal."""
        import os

        os.environ["SLM_USER_SESSION"] = "not-a-real-token"
        try:
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)

        assert result["success"] is False
        assert result["reason"] == "authentication_required"

    def test_a_reader_cannot_write(self, workspace) -> None:
        import os

        rbac = workspace["rbac"]
        os.environ["SLM_USER_SESSION"] = rbac.create_session(
            workspace["viewer"]["user_id"],
        )
        try:
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)

        assert result["success"] is False
        assert result["reason"] == "insufficient_roles"

    def test_someone_with_the_role_can_write(self, workspace) -> None:
        import os

        rbac = workspace["rbac"]
        os.environ["SLM_USER_SESSION"] = rbac.create_session(
            workspace["admin"]["user_id"],
        )
        try:
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)

        assert result["success"] is True

    def test_a_member_of_no_workspace_is_not_treated_as_a_member(
        self, workspace,
    ) -> None:
        """A valid login with no membership here must not inherit write access.

        Defaulting an unknown membership to MEMBER would hand every
        authenticated user write access to every workspace on the machine, which
        is the opposite of what per-workspace roles are for.
        """
        import os

        rbac = workspace["rbac"]
        os.environ["SLM_USER_SESSION"] = rbac.create_session(
            workspace["outsider"]["user_id"],
        )
        try:
            principal, _token, roles = _session_principal("default")
            result = _attempt()
        finally:
            os.environ.pop("SLM_USER_SESSION", None)

        assert principal == workspace["outsider"]["user_id"], (
            "the caller is identified — they simply have no role here"
        )
        assert roles == frozenset({ActorRole.ANONYMOUS})
        assert result["success"] is False


class TestTheGuardCoversEveryWriteTool:
    """One decorator, so the fix is not per-tool — but prove the coverage."""

    def test_every_gated_tool_goes_through_the_same_gate(self) -> None:
        """The registry fills at decoration time, so the tools must be registered.

        Reading the set without registering would find it empty and the test
        would pass by asserting nothing about anything.
        """
        from unittest.mock import MagicMock

        from superlocalmemory.core.admission import _GATED_MCP_TOOLS
        from superlocalmemory.mcp.tools_core import register_core_tools

        server = MagicMock()
        server.tool.return_value = lambda fn: fn
        register_core_tools(server, lambda: None)

        expected = {
            "remember", "delete_memory", "update_memory",
            "build_graph", "switch_profile", "correct_pattern",
            "review_correction",
        }
        missing = expected - _GATED_MCP_TOOLS
        assert not missing, (
            f"these write tools are not behind the admission gate: {missing}"
        )
