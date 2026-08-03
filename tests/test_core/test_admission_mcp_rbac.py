# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 — MCP RBAC admission tests.

Spec from LLD: test_admission_mcp_rbac
  - MCP remember/forget as VIEWER -> denied (insufficient_roles)
  - OWNER/ADMIN/MEMBER -> allowed
  - unauthenticated -> authentication_required
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
from superlocalmemory.core.admission import AdmissionDenied, admit
from superlocalmemory.core.operation_request import OperationKind


def _mcp_actor(role: ActorRole, principal: str = "user-42") -> ActorContext:
    return ActorContext(
        principal_id=principal if role != ActorRole.ANONYMOUS else "",
        roles=frozenset({role}),
        transport=Transport.MCP,
        client_host="127.0.0.1",
    )


# ---------------------------------------------------------------------------
# REMEMBER — multi-role matrix
# ---------------------------------------------------------------------------

def test_mcp_remember_viewer_denied():
    actor = _mcp_actor(ActorRole.VIEWER)
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.REMEMBER, actor, mode="company")
    assert exc_info.value.decision.reason == "insufficient_roles"


def test_mcp_remember_owner_allowed():
    actor = _mcp_actor(ActorRole.OWNER)
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed


def test_mcp_remember_admin_allowed():
    actor = _mcp_actor(ActorRole.ADMIN)
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed


def test_mcp_remember_member_allowed():
    actor = _mcp_actor(ActorRole.MEMBER)
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed


def test_mcp_remember_unauthenticated_denied():
    actor = _mcp_actor(ActorRole.ANONYMOUS, principal="")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.REMEMBER, actor, mode="company")
    assert exc_info.value.decision.reason == "authentication_required"


# ---------------------------------------------------------------------------
# FORGET — role restrictions (OWNER/ADMIN only)
# ---------------------------------------------------------------------------

def test_mcp_forget_viewer_denied():
    actor = _mcp_actor(ActorRole.VIEWER)
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.FORGET, actor, mode="local")
    assert exc_info.value.decision.reason == "insufficient_roles"


def test_mcp_forget_member_denied():
    actor = _mcp_actor(ActorRole.MEMBER)
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.FORGET, actor, mode="local")
    assert exc_info.value.decision.reason == "insufficient_roles"


def test_mcp_forget_owner_allowed():
    actor = _mcp_actor(ActorRole.OWNER)
    decision = admit(OperationKind.FORGET, actor, mode="local")
    assert decision.allowed


def test_mcp_forget_admin_allowed():
    actor = _mcp_actor(ActorRole.ADMIN)
    decision = admit(OperationKind.FORGET, actor, mode="local")
    assert decision.allowed


# ---------------------------------------------------------------------------
# RECALL — all roles including VIEWER
# ---------------------------------------------------------------------------

def test_mcp_recall_viewer_allowed():
    actor = _mcp_actor(ActorRole.VIEWER)
    decision = admit(OperationKind.RECALL, actor, mode="local")
    assert decision.allowed


def test_mcp_recall_anonymous_denied():
    actor = _mcp_actor(ActorRole.ANONYMOUS, principal="")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.RECALL, actor, mode="company")
    assert exc_info.value.decision.reason == "authentication_required"
