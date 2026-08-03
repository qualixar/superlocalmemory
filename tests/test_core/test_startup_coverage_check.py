# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Phase 1 — startup coverage self-check unit tests.

Tests the coverage_self_check() function that unified_daemon calls at startup.
"""

from __future__ import annotations

import pytest


def test_startup_coverage_check_passes_personal_mode():
    """coverage_self_check() in personal mode warns but does not raise."""
    from superlocalmemory.core.admission import coverage_self_check
    from superlocalmemory.core.config import DEPLOYMENT_PERSONAL

    coverage_self_check(DEPLOYMENT_PERSONAL)


def test_startup_coverage_check_all_kinds_covered():
    """All OperationKind values in the default registry have policies."""
    from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
    from superlocalmemory.core.operation_request import OperationKind

    cov = _DEFAULT_REGISTRY.coverage()
    uncovered = [
        k.value for k in OperationKind
        if not cov.get(k.value, {}).get("has_policy", False)
    ]
    assert not uncovered, f"OperationKinds without policies: {uncovered}"


def test_startup_coverage_check_enterprise_mode_raises_on_missing_policy(monkeypatch):
    """coverage_self_check() in enterprise mode raises RuntimeError if a kind lacks a policy."""
    from superlocalmemory.core.admission import coverage_self_check
    from superlocalmemory.core.config import DEPLOYMENT_ENTERPRISE
    from superlocalmemory.core.operation_policy_registry import OperationPolicyRegistry
    from superlocalmemory.core.operation_request import OperationKind

    gapped_registry = OperationPolicyRegistry({})

    with pytest.raises(RuntimeError, match="policy coverage gap"):
        coverage_self_check(DEPLOYMENT_ENTERPRISE, registry=gapped_registry)
