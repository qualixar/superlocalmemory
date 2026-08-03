# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for the V4 Phase 4 admission / policy layer.

Coverage targets:
  - ActorContext properties (is_local, is_authenticated)
  - OperationRequest defaults and idempotency_key fallback
  - PolicyDecision construction
  - OperationPolicyRegistry:
      * default allow for REMEMBER + OWNER + local (happy path MUST NOT regress)
      * default allow for REMEMBER + MEMBER + HTTP (multi-user local)
      * unknown kind string: fail-open in local mode
      * unknown kind string: fail-closed in company mode
      * explicit deny overrides policy
      * payload annotation (no rejection)
      * insufficient roles → deny
      * unauthenticated actor → deny
      * restricted transport → deny
      * get_policy() returns known / None for unknown
      * all OperationKind values have a default policy
      * evaluate() is CPU-only — no file, network, or DB I/O
      * evaluate() is reentrant (thread-safe reads)
      * singleton _DEFAULT_REGISTRY is shared, not copied per call
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from superlocalmemory.core.actor_context import (
    ActorContext,
    ActorRole,
    Transport,
    make_http_actor_context,
    make_internal_owner_context,
)
from superlocalmemory.core.operation_policy import OperationPolicy
from superlocalmemory.core.operation_policy_registry import (
    _DEFAULT_REGISTRY,
    OperationPolicyRegistry,
    PolicyDecision,
)
from superlocalmemory.core.operation_request import (
    OperationKind,
    OperationRequest,
    OperationStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _owner_local_actor(profile_id: str = "test-profile") -> ActorContext:
    """Standard ActorContext for the internal/single-user path."""
    return ActorContext(
        principal_id="local-operator",
        roles=frozenset({ActorRole.OWNER}),
        active_profile_id=profile_id,
        transport=Transport.INTERNAL,
        client_host="",
    )


def _member_http_actor(profile_id: str = "test-profile") -> ActorContext:
    return ActorContext(
        principal_id="user-42",
        roles=frozenset({ActorRole.MEMBER}),
        active_profile_id=profile_id,
        transport=Transport.HTTP,
        client_host="127.0.0.1",
    )


def _anon_actor() -> ActorContext:
    return ActorContext(
        principal_id="",
        roles=frozenset({ActorRole.ANONYMOUS}),
        transport=Transport.HTTP,
        client_host="10.0.0.1",
    )


# ---------------------------------------------------------------------------
# ActorContext
# ---------------------------------------------------------------------------


class TestActorContext:
    def test_is_local_empty_host(self):
        actor = ActorContext(client_host="")
        assert actor.is_local is True

    def test_is_local_loopback_v4(self):
        actor = ActorContext(client_host="127.0.0.1")
        assert actor.is_local is True

    def test_is_local_loopback_v6(self):
        actor = ActorContext(client_host="::1")
        assert actor.is_local is True

    def test_is_local_localhost(self):
        actor = ActorContext(client_host="localhost")
        assert actor.is_local is True

    def test_is_not_local_remote(self):
        actor = ActorContext(client_host="10.0.0.5")
        assert actor.is_local is False

    def test_is_authenticated_owner(self):
        actor = ActorContext(
            principal_id="user-1",
            roles=frozenset({ActorRole.OWNER}),
        )
        assert actor.is_authenticated is True

    def test_is_not_authenticated_empty_principal(self):
        actor = ActorContext(principal_id="")
        assert actor.is_authenticated is False

    def test_is_not_authenticated_anonymous_role(self):
        actor = ActorContext(
            principal_id="anon",
            roles=frozenset({ActorRole.ANONYMOUS}),
        )
        assert actor.is_authenticated is False

    def test_frozen(self):
        actor = ActorContext(principal_id="x")
        with pytest.raises((AttributeError, TypeError)):
            actor.principal_id = "y"  # type: ignore[misc]

    def test_default_roles_owner(self):
        actor = ActorContext()
        assert ActorRole.OWNER in actor.roles

    def test_allowed_profiles_empty_is_unrestricted(self):
        actor = ActorContext()
        # Empty frozenset means "all profiles allowed" — verify the default
        assert len(actor.allowed_profiles) == 0

    def test_make_internal_owner_context(self):
        actor = make_internal_owner_context(principal_id="daemon", profile_id="p1")
        assert actor.transport == Transport.INTERNAL
        assert actor.is_local is True
        assert actor.is_authenticated is True
        assert ActorRole.OWNER in actor.roles

    def test_make_http_actor_context(self):
        actor = make_http_actor_context(
            principal_id="u1",
            profile_id="p1",
            client_host="127.0.0.1",
        )
        assert actor.transport == Transport.HTTP
        assert actor.is_authenticated is True


# ---------------------------------------------------------------------------
# OperationRequest
# ---------------------------------------------------------------------------


class TestOperationRequest:
    def test_operation_id_auto_generated(self):
        req = OperationRequest()
        assert len(req.operation_id) == 32  # UUID hex

    def test_idempotency_key_defaults_to_operation_id(self):
        req = OperationRequest()
        assert req.idempotency_key == req.operation_id

    def test_idempotency_key_explicit(self):
        req = OperationRequest(idempotency_key="my-key")
        assert req.idempotency_key == "my-key"

    def test_default_kind_remember(self):
        req = OperationRequest()
        assert req.kind == OperationKind.REMEMBER

    def test_frozen(self):
        req = OperationRequest()
        with pytest.raises((AttributeError, TypeError)):
            req.kind = OperationKind.RECALL  # type: ignore[misc]

    def test_created_at_is_utc_iso(self):
        req = OperationRequest()
        assert "+" in req.created_at or req.created_at.endswith("Z") or "T" in req.created_at

    def test_operation_kind_members(self):
        # Verify all expected members exist — regression guard.
        # Phase 1 addition: "evolve_skill" (PHASE1_LLD.md mandate).
        expected = {
            "remember", "recall", "forget", "archive", "restore", "correct",
            "erase", "consolidate", "backup", "restore_backup", "mesh_send",
            "mesh_lock", "provider_test", "mode_change", "profile_switch",
            "schema_migrate", "vector_migrate", "evolve_skill",
        }
        actual = {k.value for k in OperationKind}
        assert expected == actual

    def test_operation_status_members(self):
        expected = {
            "accepted", "committed", "projecting", "complete",
            "degraded", "failed", "rolled_back",
        }
        assert {s.value for s in OperationStatus} == expected


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------


class TestPolicyDecision:
    def test_allowed_true(self):
        d = PolicyDecision(allowed=True, reason="allow")
        assert d.allowed is True
        assert d.reason == "allow"
        assert d.annotations == {}

    def test_allowed_false(self):
        d = PolicyDecision(allowed=False, reason="insufficient_roles")
        assert d.allowed is False

    def test_annotations(self):
        d = PolicyDecision(allowed=False, reason="x", annotations={"k": "v"})
        assert d.annotations["k"] == "v"

    def test_frozen(self):
        d = PolicyDecision(allowed=True, reason="allow")
        with pytest.raises((AttributeError, TypeError)):
            d.allowed = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — happy-path (MUST NOT regress)
# ---------------------------------------------------------------------------


class TestRegistryHappyPath:
    """The single most important test class.

    Every test here represents a currently-working ingestion path that MUST
    continue to be ALLOWED after Phase 4 is wired up. A failure here is a P0.
    """

    def test_remember_owner_internal_local(self):
        """The canonical in-process Python-API path must always be allowed."""
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER,
            _owner_local_actor(),
            "local",
        )
        assert decision.allowed, f"P0 regression: {decision.reason} {decision.annotations}"
        assert decision.reason == "allow"

    def test_remember_owner_http_local(self):
        actor = ActorContext(
            principal_id="owner",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, actor, "local",
        )
        assert decision.allowed, f"P0 regression: {decision.reason}"

    def test_remember_member_http_local(self):
        """MEMBER role (multi-user local) must also be admitted for REMEMBER."""
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, _member_http_actor(), "local",
        )
        assert decision.allowed, f"P0 regression: {decision.reason}"

    def test_remember_owner_mcp_local(self):
        actor = ActorContext(
            principal_id="mcp-user",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.MCP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(OperationKind.REMEMBER, actor, "local")
        assert decision.allowed, f"P0 regression: {decision.reason}"

    def test_remember_admin_company_mode(self):
        """ADMIN role in company mode must be admitted for REMEMBER."""
        actor = ActorContext(
            principal_id="admin-1",
            roles=frozenset({ActorRole.ADMIN}),
            transport=Transport.HTTP,
            client_host="10.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(OperationKind.REMEMBER, actor, "company")
        assert decision.allowed, f"regression: {decision.reason}"

    def test_default_mode_is_local(self):
        """Omitting mode must default to local (fail-open)."""
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, _owner_local_actor()
        )  # no mode argument
        assert decision.allowed, f"P0 regression: {decision.reason}"


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — unknown kind
# ---------------------------------------------------------------------------


class TestUnknownKind:
    def test_unknown_string_local_mode_allows(self):
        """Unknown kind in local/single-user mode → fail-open (allow + audit)."""
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate("completely_unknown_v99", actor, "local")
        assert decision.allowed is True
        assert decision.annotations.get("audit") is True
        assert "unknown_kind_allow_local" in decision.reason

    def test_unknown_string_single_user_mode_allows(self):
        """'single-user' is a synonym for 'local'."""
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate("future_op", actor, "single-user")
        assert decision.allowed is True

    def test_unknown_string_company_mode_denies(self):
        """Unknown kind in company mode → fail-closed (deny)."""
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate("future_op", actor, "company")
        assert decision.allowed is False
        assert "unknown_kind_deny_company" in decision.reason

    def test_unknown_string_remote_mode_denies(self):
        """'remote' is a synonym for 'company'."""
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate("future_op", actor, "remote")
        assert decision.allowed is False

    def test_unknown_string_annotations_carry_kind_raw(self):
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate("weird:op/v2", actor, "local")
        assert decision.annotations.get("kind_raw") == "weird:op/v2"


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — explicit deny
# ---------------------------------------------------------------------------


class TestExplicitDeny:
    def test_explicit_deny_overrides_policy(self):
        """register_deny() must deny regardless of actor role or mode."""
        registry = _DEFAULT_REGISTRY.register_deny(OperationKind.ERASE)
        actor = _owner_local_actor()
        decision = registry.evaluate(OperationKind.ERASE, actor, "local")
        assert decision.allowed is False
        assert decision.reason == "explicit_deny"

    def test_explicit_deny_does_not_affect_other_kinds(self):
        """Denying ERASE must not affect REMEMBER."""
        registry = _DEFAULT_REGISTRY.register_deny(OperationKind.ERASE)
        actor = _owner_local_actor()
        decision = registry.evaluate(OperationKind.REMEMBER, actor, "local")
        assert decision.allowed is True

    def test_register_deny_returns_new_registry(self):
        """register_deny() is an immutable-update — original is unchanged."""
        original = OperationPolicyRegistry.default()
        updated = original.register_deny(OperationKind.ERASE)
        assert updated is not original
        # Original must still allow ERASE for OWNER
        actor = _owner_local_actor()
        assert original.evaluate(OperationKind.ERASE, actor, "local").allowed is True
        assert updated.evaluate(OperationKind.ERASE, actor, "local").allowed is False

    def test_deny_list_denies_even_in_local_mode(self):
        """Explicit deny always wins — even in fail-open local mode."""
        registry = _DEFAULT_REGISTRY.register_deny(OperationKind.REMEMBER)
        actor = _owner_local_actor()
        decision = registry.evaluate(OperationKind.REMEMBER, actor, "local")
        assert decision.allowed is False


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — role / auth / transport denials
# ---------------------------------------------------------------------------


class TestPolicyViolations:
    def test_insufficient_roles_deny(self):
        """VIEWER cannot write (REMEMBER requires at least MEMBER)."""
        actor = ActorContext(
            principal_id="viewer-1",
            roles=frozenset({ActorRole.VIEWER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(OperationKind.REMEMBER, actor, "local")
        assert decision.allowed is False
        assert decision.reason == "insufficient_roles"

    def test_unauthenticated_actor_deny(self):
        """Anonymous / unauthenticated actor is denied on authenticated operations."""
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, _anon_actor(), "local",
        )
        assert decision.allowed is False
        assert decision.reason == "authentication_required"

    def test_schema_migrate_denied_over_http(self):
        """SCHEMA_MIGRATE is restricted to CLI/INTERNAL — HTTP must be denied."""
        actor = ActorContext(
            principal_id="operator",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.SCHEMA_MIGRATE, actor, "local",
        )
        assert decision.allowed is False
        assert decision.reason == "transport_not_allowed"

    def test_schema_migrate_allowed_internal(self):
        actor = ActorContext(
            principal_id="daemon",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.INTERNAL,
            client_host="",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.SCHEMA_MIGRATE, actor, "local",
        )
        assert decision.allowed is True

    def test_erase_denied_to_admin(self):
        """ERASE requires OWNER — ADMIN must be denied."""
        actor = ActorContext(
            principal_id="admin-1",
            roles=frozenset({ActorRole.ADMIN}),
            transport=Transport.HTTP,
            client_host="127.0.0.1",
        )
        decision = _DEFAULT_REGISTRY.evaluate(OperationKind.ERASE, actor, "local")
        assert decision.allowed is False
        assert decision.reason == "insufficient_roles"

    def test_mode_change_denied_over_mesh(self):
        """MODE_CHANGE is restricted to CLI/INTERNAL/DASHBOARD — MESH is denied."""
        actor = ActorContext(
            principal_id="operator",
            roles=frozenset({ActorRole.OWNER}),
            transport=Transport.MESH,
            client_host="10.0.0.2",
        )
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.MODE_CHANGE, actor, "local",
        )
        assert decision.allowed is False
        assert decision.reason == "transport_not_allowed"


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — payload annotation
# ---------------------------------------------------------------------------


class TestPayloadAnnotation:
    def test_oversized_payload_is_annotated_not_denied(self):
        """Payload > max_payload_bytes annotates but NEVER rejects (double-rejection ban)."""
        actor = _owner_local_actor()
        oversized = 2 * 1024 * 1024  # 2 MiB > 1 MiB default
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, actor, "local", payload_bytes=oversized,
        )
        assert decision.allowed is True  # NOT denied
        assert decision.annotations.get("payload_oversized") is True
        assert decision.annotations["payload_bytes"] == oversized

    def test_normal_payload_has_no_annotation(self):
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, actor, "local", payload_bytes=512,
        )
        assert decision.allowed is True
        assert "payload_oversized" not in decision.annotations

    def test_zero_payload_bytes_skips_annotation(self):
        actor = _owner_local_actor()
        decision = _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, actor, "local", payload_bytes=0,
        )
        assert "payload_oversized" not in decision.annotations


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — get_policy
# ---------------------------------------------------------------------------


class TestGetPolicy:
    def test_get_policy_returns_known_kind(self):
        policy = _DEFAULT_REGISTRY.get_policy(OperationKind.REMEMBER)
        assert policy is not None
        assert policy.kind == OperationKind.REMEMBER

    def test_get_policy_unknown_kind_returns_none(self):
        # Build a minimal registry with only REMEMBER registered.
        sparse = OperationPolicyRegistry(
            {OperationKind.REMEMBER: OperationPolicy(kind=OperationKind.REMEMBER)}
        )
        result = sparse.get_policy(OperationKind.RECALL)
        assert result is None

    def test_all_known_kinds_have_default_policy(self):
        """Every OperationKind member must have a policy in the default registry."""
        missing = []
        for kind in OperationKind:
            if _DEFAULT_REGISTRY.get_policy(kind) is None:
                missing.append(kind.value)
        assert not missing, f"Missing default policies for: {missing}"

    def test_get_policy_transport_arg_accepted(self):
        """get_policy() accepts transport without error (reserved for future use)."""
        policy = _DEFAULT_REGISTRY.get_policy(OperationKind.REMEMBER, Transport.HTTP)
        assert policy is not None


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — immutability and structure
# ---------------------------------------------------------------------------


class TestRegistryImmutability:
    def test_singleton_is_not_mutated_by_evaluate(self):
        """evaluate() must not mutate _DEFAULT_REGISTRY's internal state."""
        actor = _owner_local_actor()
        before = repr(_DEFAULT_REGISTRY)
        for _ in range(100):
            _DEFAULT_REGISTRY.evaluate(OperationKind.REMEMBER, actor, "local")
        after = repr(_DEFAULT_REGISTRY)
        assert before == after

    def test_with_policy_returns_new_instance(self):
        original = OperationPolicyRegistry.default()
        updated = original.with_policy(
            OperationPolicy(
                kind=OperationKind.RECALL,
                required_roles=frozenset({ActorRole.OWNER}),
            )
        )
        assert updated is not original

    def test_internal_mapping_is_read_only(self):
        """The internal _policies mapping must reject mutation attempts."""
        import types as _types

        assert isinstance(
            _DEFAULT_REGISTRY._policies, _types.MappingProxyType
        ), "Internal policies must be MappingProxyType for immutability"
        with pytest.raises(TypeError):
            _DEFAULT_REGISTRY._policies[OperationKind.REMEMBER] = None  # type: ignore


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — I/O purity
# ---------------------------------------------------------------------------


class TestEvaluatePurity:
    def test_evaluate_is_cpu_only_no_file_io(self):
        """evaluate() must not touch the filesystem. Patching open() verifies this."""
        actor = _owner_local_actor()
        with patch(
            "builtins.open",
            side_effect=AssertionError("open() called inside evaluate() — violates I/O purity"),
        ):
            decision = _DEFAULT_REGISTRY.evaluate(
                OperationKind.REMEMBER, actor, "local"
            )
        assert decision.allowed is True

    def test_evaluate_is_cpu_only_no_sqlite(self):
        """evaluate() must not touch SQLite."""
        import sqlite3

        actor = _owner_local_actor()
        with patch.object(
            sqlite3,
            "connect",
            side_effect=AssertionError("sqlite3.connect() called inside evaluate()"),
        ):
            decision = _DEFAULT_REGISTRY.evaluate(
                OperationKind.REMEMBER, actor, "local"
            )
        assert decision.allowed is True

    def test_evaluate_runs_in_microseconds(self):
        """1 000 sequential evaluate() calls must complete well under 1 second."""
        actor = _owner_local_actor()
        start = time.monotonic()
        for _ in range(1000):
            _DEFAULT_REGISTRY.evaluate(OperationKind.REMEMBER, actor, "local")
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, (
            f"evaluate() is too slow for a pure CPU function: {elapsed:.3f}s for 1000 calls"
        )

    def test_evaluate_is_thread_safe(self):
        """evaluate() must be safe to call concurrently from multiple threads."""
        actor = _owner_local_actor()
        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _worker():
            try:
                for _ in range(200):
                    d = _DEFAULT_REGISTRY.evaluate(
                        OperationKind.REMEMBER, actor, "local"
                    )
                    with lock:
                        results.append(d.allowed)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety violations: {errors}"
        # 8 threads × 200 calls = 1600 results, all True
        assert all(results), "Some evaluate() calls returned unexpected deny"
        assert len(results) == 1600


# ---------------------------------------------------------------------------
# OperationPolicyRegistry — with_policy (custom override)
# ---------------------------------------------------------------------------


class TestWithPolicy:
    def test_custom_policy_override(self):
        """with_policy() lets callers tighten a policy (e.g. restrict REMEMBER to OWNER)."""
        strict_policy = OperationPolicy(
            kind=OperationKind.REMEMBER,
            required_roles=frozenset({ActorRole.OWNER}),
            required_authentication=True,
        )
        strict_registry = _DEFAULT_REGISTRY.with_policy(strict_policy)

        member_actor = _member_http_actor()
        owner_actor = _owner_local_actor()

        # MEMBER is denied because the custom policy requires OWNER.
        assert strict_registry.evaluate(
            OperationKind.REMEMBER, member_actor, "local"
        ).allowed is False

        # OWNER is still admitted.
        assert strict_registry.evaluate(
            OperationKind.REMEMBER, owner_actor, "local"
        ).allowed is True

    def test_with_policy_does_not_mutate_default_registry(self):
        """Modifying a derived registry must not affect _DEFAULT_REGISTRY."""
        _DEFAULT_REGISTRY.with_policy(
            OperationPolicy(
                kind=OperationKind.REMEMBER,
                required_roles=frozenset(),  # empty → deny all
            )
        )
        # Default registry must still allow REMEMBER for OWNER.
        actor = _owner_local_actor()
        assert _DEFAULT_REGISTRY.evaluate(
            OperationKind.REMEMBER, actor, "local"
        ).allowed is True
