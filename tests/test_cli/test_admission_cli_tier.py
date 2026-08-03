# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 — CLI tier admission tests.

Spec from LLD: test_admission_cli_tier
  - CLI mutation, enterprise tier, no login -> denied
  - personal tier loopback -> allowed (frictionless)
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.actor_context import ActorRole, Transport
from superlocalmemory.core.admission import AdmissionDenied, admit, resolve_actor
from superlocalmemory.core.operation_request import OperationKind


def test_cli_personal_remember_frictionless():
    actor = resolve_actor(Transport.CLI, tier="personal", mode="local")
    decision = admit(OperationKind.REMEMBER, actor, mode="local")
    assert decision.allowed
    assert ActorRole.OWNER in actor.roles


def test_cli_personal_forget_frictionless():
    actor = resolve_actor(Transport.CLI, tier="personal")
    decision = admit(OperationKind.FORGET, actor, mode="local")
    assert decision.allowed


def test_cli_personal_consolidate_frictionless():
    actor = resolve_actor(Transport.CLI, tier="personal")
    decision = admit(OperationKind.CONSOLIDATE, actor, mode="local")
    assert decision.allowed


def test_cli_enterprise_no_login_denied():
    actor = resolve_actor(Transport.CLI, tier="enterprise", mode="company")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.REMEMBER, actor, mode="company")
    assert exc_info.value.decision.reason == "authentication_required"
    assert ActorRole.ANONYMOUS in actor.roles


def test_cli_enterprise_no_login_consolidate_denied():
    actor = resolve_actor(Transport.CLI, tier="enterprise", mode="company")
    with pytest.raises(AdmissionDenied) as exc_info:
        admit(OperationKind.CONSOLIDATE, actor, mode="company")
    assert exc_info.value.decision.reason == "authentication_required"


def test_cli_enterprise_with_admin_remember_allowed():
    actor = resolve_actor(
        Transport.CLI,
        tier="enterprise",
        mode="company",
        principal="admin-user",
        roles=frozenset({ActorRole.ADMIN}),
    )
    decision = admit(OperationKind.REMEMBER, actor, mode="company")
    assert decision.allowed


def test_cli_enterprise_with_owner_schema_migrate_allowed():
    actor = resolve_actor(
        Transport.CLI,
        tier="enterprise",
        mode="company",
        principal="sysadmin",
        roles=frozenset({ActorRole.OWNER}),
    )
    decision = admit(OperationKind.SCHEMA_MIGRATE, actor, mode="company")
    assert decision.allowed
