# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 admission gateway — core unit tests.

Covers:
  - resolve_actor: personal mode → OWNER (frictionless)
  - resolve_actor: enterprise mode + no principal → ANONYMOUS
  - resolve_actor: enterprise mode + principal → supplied roles
  - admit: OWNER + REMEMBER → allowed
  - admit: ANONYMOUS + REMEMBER → AdmissionDenied(authentication_required)
  - admit: VIEWER + FORGET → AdmissionDenied(insufficient_roles)
  - AdmissionDenied carries the PolicyDecision
  - frictionless regression: personal+MCP/CLI/INTERNAL OWNER can do all
    operations permitted by the existing transport policies
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport, make_internal_owner_context
from superlocalmemory.core.admission import AdmissionDenied, admit, resolve_actor
from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
from superlocalmemory.core.operation_request import OperationKind


# ---------------------------------------------------------------------------
# resolve_actor
# ---------------------------------------------------------------------------

def test_resolve_actor_personal_mcp_returns_owner():
    actor = resolve_actor(Transport.MCP, tier="personal")
    assert ActorRole.OWNER in actor.roles
    assert actor.transport == Transport.MCP
    assert actor.is_authenticated


def test_resolve_actor_personal_cli_returns_owner():
    actor = resolve_actor(Transport.CLI, tier="personal")
    assert ActorRole.OWNER in actor.roles
    assert actor.transport == Transport.CLI


def test_resolve_actor_personal_internal_returns_owner():
    actor = resolve_actor(Transport.INTERNAL, tier="personal")
    assert ActorRole.OWNER in actor.roles
    assert actor.transport == Transport.INTERNAL


def test_resolve_actor_enterprise_no_principal_returns_anonymous():
    actor = resolve_actor(Transport.MCP, tier="enterprise", mode="company")
    assert ActorRole.ANONYMOUS in actor.roles
    assert not actor.is_authenticated


def test_resolve_actor_enterprise_with_principal_returns_member_default():
    actor = resolve_actor(Transport.MCP, tier="enterprise", mode="company", principal="user-42")
    assert actor.principal_id == "user-42"
    assert ActorRole.ANONYMOUS not in actor.roles
    assert ActorRole.MEMBER in actor.roles


def test_resolve_actor_enterprise_with_explicit_admin_roles():
    actor = resolve_actor(
        Transport.MCP,
        tier="enterprise",
        mode="company",
        principal="admin-1",
        roles=frozenset({ActorRole.ADMIN}),
    )
    assert ActorRole.ADMIN in actor.roles
    assert actor.principal_id == "admin-1"


def test_resolve_actor_company_mode_string_is_enterprise():
    actor = resolve_actor(Transport.CLI, mode="company")
    assert ActorRole.ANONYMOUS in actor.roles


def test_resolve_actor_default_is_personal_owner():
    actor = resolve_actor(Transport.MCP)
    assert ActorRole.OWNER in actor.roles


# ---------------------------------------------------------------------------
# admit
# ---------------------------------------------------------------------------

def test_admit_owner_remember_local_allowed():
    actor = resolve_actor(Transport.MCP, tier="personal")
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed
    assert decision.reason == "allow"


def test_admit_owner_forget_local_allowed():
    actor = resolve_actor(Transport.CLI, tier="personal")
    decision = admit(OperationKind.FORGET, actor, mode="local")
    assert decision.allowed


def test_admit_anonymous_remember_raises_authentication_required():
    actor = resolve_actor(Transport.MCP, tier="enterprise", mode="company")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.REMEMBER, actor, mode="company")
    assert exc_info.value.decision.reason == "authentication_required"


def test_admit_viewer_forget_raises_insufficient_roles():
    actor = ActorContext(
        principal_id="viewer-1",
        roles=frozenset({ActorRole.VIEWER}),
        transport=Transport.MCP,
    )
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.FORGET, actor, mode="local")
    assert exc_info.value.decision.reason == "insufficient_roles"


def test_admission_denied_has_decision():
    actor = resolve_actor(Transport.MCP, tier="enterprise", mode="company")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.REMEMBER, actor, mode="company")
    exc = exc_info.value
    assert exc.decision is not None
    assert not exc.decision.allowed
    assert exc.decision.reason == "authentication_required"


def test_admit_member_remember_local_allowed():
    actor = ActorContext(
        principal_id="member-1",
        roles=frozenset({ActorRole.MEMBER}),
        transport=Transport.MCP,
    )
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed


# ---------------------------------------------------------------------------
# Frictionless regression: personal OWNER can do all transport-appropriate ops
# ---------------------------------------------------------------------------

def test_personal_mcp_owner_remember_frictionless():
    actor = resolve_actor(Transport.MCP, tier="personal")
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed


def test_personal_mcp_owner_recall_frictionless():
    actor = resolve_actor(Transport.MCP, tier="personal")
    decision = admit(OperationKind.RECALL, actor, mode="local")
    assert decision.allowed


def test_personal_mcp_owner_forget_frictionless():
    actor = resolve_actor(Transport.MCP, tier="personal")
    decision = admit(OperationKind.FORGET, actor, mode="local")
    assert decision.allowed


def test_personal_cli_owner_consolidate_frictionless():
    actor = resolve_actor(Transport.CLI, tier="personal")
    decision = admit(OperationKind.CONSOLIDATE, actor, mode="local")
    assert decision.allowed


def test_personal_internal_owner_remember_frictionless():
    actor = make_internal_owner_context(principal_id="local-operator")
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed
