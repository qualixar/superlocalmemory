# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 — registry coverage self-check tests.

Spec from LLD: test_registry_coverage
  - Every mutating OperationKind has a policy + enforcement site (self-check)
  - EVOLVE_SKILL exists as an OperationKind
  - coverage() shows every hero kind has a policy
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
from superlocalmemory.core.operation_request import OperationKind


def test_all_operation_kinds_have_policy():
    """Every OperationKind value must have an entry in the default registry."""
    missing = []
    for kind in OperationKind:
        policy = _DEFAULT_REGISTRY.get_policy(kind)
        if policy is None:
            missing.append(kind.name)
    assert not missing, f"OperationKind values missing policy: {missing}"


def test_evolve_skill_kind_exists():
    assert hasattr(OperationKind, "EVOLVE_SKILL")
    assert OperationKind.EVOLVE_SKILL.value == "evolve_skill"


def test_evolve_skill_has_policy():
    policy = _DEFAULT_REGISTRY.get_policy(OperationKind.EVOLVE_SKILL)
    assert policy is not None
    assert policy.kind == OperationKind.EVOLVE_SKILL


def test_evolve_skill_owner_admin_only():
    from superlocalmemory.core.actor_context import ActorRole
    policy = _DEFAULT_REGISTRY.get_policy(OperationKind.EVOLVE_SKILL)
    assert ActorRole.OWNER in policy.required_roles
    assert ActorRole.ADMIN in policy.required_roles
    assert ActorRole.MEMBER not in policy.required_roles


def test_coverage_method_exists():
    assert callable(getattr(_DEFAULT_REGISTRY, "coverage", None))


def test_coverage_returns_dict():
    cov = _DEFAULT_REGISTRY.coverage()
    assert isinstance(cov, dict)


def test_coverage_all_kinds_present():
    cov = _DEFAULT_REGISTRY.coverage()
    for kind in OperationKind:
        assert kind.value in cov, f"{kind.value} missing from coverage()"


def test_coverage_has_has_policy_field():
    cov = _DEFAULT_REGISTRY.coverage()
    for kind in OperationKind:
        entry = cov[kind.value]
        assert "has_policy" in entry, f"{kind.value} missing has_policy"
        assert entry["has_policy"] is True, f"{kind.value} has_policy is False"


def test_coverage_evolve_skill_is_covered():
    cov = _DEFAULT_REGISTRY.coverage()
    assert "evolve_skill" in cov
    assert cov["evolve_skill"]["has_policy"] is True
