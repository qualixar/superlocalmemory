"""Experiment 8 — OperationPolicyRegistry: access-control decisions.

Spine exercised:
  OperationPolicyRegistry (core/operation_policy_registry.py, line 72)
  _DEFAULT_REGISTRY singleton (module-level, constructed once at import time)
  ActorContext.is_authenticated predicate (actor_context.py, line 108)

This is a pure CPU test — no I/O, no DB, no threads. Each trial runs five
sub-assertions against _DEFAULT_REGISTRY.evaluate():

  A. KNOWN KIND / AUTHORIZED ROLES — reason check
     REMEMBER + OWNER, ADMIN, MEMBER (each in a separate ActorContext,
     principal_id="alice", transport=HTTP) → allowed=True AND reason=="allow".
     Both the allow decision AND the exact reason string are asserted; a
     policy that admits but returns a wrong reason code would fail here.

  B. UNKNOWN KIND / LOCAL MODE (fail-open)
     A raw string that cannot map to any OperationKind enum value,
     mode="local" → allowed=True, reason="unknown_kind_allow_local".

  C. UNKNOWN KIND / COMPANY MODE (fail-closed)
     Same unknown kind string, mode="company" →
     allowed=False, reason="unknown_kind_deny_company".

  D. UNAUTHENTICATED ACTOR / KNOWN KIND
     ActorContext(principal_id="", roles={OWNER}, transport=HTTP).
     is_authenticated=False (principal_id is empty).
     REMEMBER policy has required_authentication=True → authentication
     check fires BEFORE role check → reason="authentication_required"
     (NOT "insufficient_roles").

  E. VIEWER → REMEMBER → deny with insufficient_roles
     ActorContext(principal_id="alice", roles={VIEWER}, transport=HTTP).
     VIEWER is not in REMEMBER's required_roles ({OWNER, ADMIN, MEMBER}).
     Expected: allowed=False, reason="insufficient_roles".
     (VIEWER may read via RECALL — which includes VIEWER in required_roles —
     but may NOT write via REMEMBER.)

A trial holds only when all five sub-assertions pass.
"""

from __future__ import annotations

from pathlib import Path

from _harness import TrialOutcome, run_trials

_UNKNOWN_KIND = "NONEXISTENT_OP_XYZ_7b3f9a"


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


def _trial(index: int) -> TrialOutcome:
    from superlocalmemory.core.actor_context import ActorContext, ActorRole, Transport
    from superlocalmemory.core.operation_policy_registry import _DEFAULT_REGISTRY
    from superlocalmemory.core.operation_request import OperationKind

    reg = _DEFAULT_REGISTRY

    # ----------------------------------------------------------------
    # Sub-assertion A: REMEMBER allowed for OWNER, ADMIN, MEMBER
    # AND reason == "allow" for each (not just any truthy allowed).
    # ----------------------------------------------------------------
    auth_roles = [ActorRole.OWNER, ActorRole.ADMIN, ActorRole.MEMBER]
    auth_results = []
    for role in auth_roles:
        actor = ActorContext(
            principal_id="alice",
            roles=frozenset({role}),
            transport=Transport.HTTP,
        )
        decision = reg.evaluate(OperationKind.REMEMBER, actor, "local")
        auth_results.append((role.value, decision.allowed, decision.reason))

    roles_ok = all(
        allowed and reason == "allow"
        for _, allowed, reason in auth_results
    )
    roles_detail: dict = {}
    if not roles_ok:
        roles_detail = {
            r: {"allowed": a, "reason": reason}
            for r, a, reason in auth_results
            if not a or reason != "allow"
        }

    # ----------------------------------------------------------------
    # Sub-assertion B: unknown kind + local mode → fail-open
    # ----------------------------------------------------------------
    anon_actor = ActorContext(
        principal_id="alice",
        roles=frozenset({ActorRole.OWNER}),
        transport=Transport.HTTP,
    )
    local_decision = reg.evaluate(_UNKNOWN_KIND, anon_actor, "local")
    local_ok = (
        local_decision.allowed
        and local_decision.reason == "unknown_kind_allow_local"
    )
    local_detail: dict = {}
    if not local_ok:
        local_detail = {
            "allowed": local_decision.allowed,
            "reason": local_decision.reason,
        }

    # ----------------------------------------------------------------
    # Sub-assertion C: unknown kind + company mode → fail-closed
    # ----------------------------------------------------------------
    company_decision = reg.evaluate(_UNKNOWN_KIND, anon_actor, "company")
    company_ok = (
        not company_decision.allowed
        and company_decision.reason == "unknown_kind_deny_company"
    )
    company_detail: dict = {}
    if not company_ok:
        company_detail = {
            "allowed": company_decision.allowed,
            "reason": company_decision.reason,
        }

    # ----------------------------------------------------------------
    # Sub-assertion D: empty principal_id → authentication_required
    # (not insufficient_roles — auth check fires before role check)
    # ----------------------------------------------------------------
    unauthed = ActorContext(
        principal_id="",
        roles=frozenset({ActorRole.OWNER}),  # role would pass if checked
        transport=Transport.HTTP,
    )
    unauthed_decision = reg.evaluate(OperationKind.REMEMBER, unauthed, "local")
    unauthed_ok = (
        not unauthed_decision.allowed
        and unauthed_decision.reason == "authentication_required"
    )
    unauthed_detail: dict = {}
    if not unauthed_ok:
        unauthed_detail = {
            "allowed": unauthed_decision.allowed,
            "reason": unauthed_decision.reason,
            "expected_reason": "authentication_required",
        }

    # ----------------------------------------------------------------
    # Sub-assertion E: VIEWER → REMEMBER → insufficient_roles
    # VIEWER is not in REMEMBER's required_roles ({OWNER, ADMIN, MEMBER}).
    # An authenticated VIEWER must be denied for role, not for auth.
    # ----------------------------------------------------------------
    viewer = ActorContext(
        principal_id="alice",
        roles=frozenset({ActorRole.VIEWER}),
        transport=Transport.HTTP,
    )
    viewer_decision = reg.evaluate(OperationKind.REMEMBER, viewer, "local")
    viewer_ok = (
        not viewer_decision.allowed
        and viewer_decision.reason == "insufficient_roles"
    )
    viewer_detail: dict = {}
    if not viewer_ok:
        viewer_detail = {
            "allowed": viewer_decision.allowed,
            "reason": viewer_decision.reason,
            "expected_reason": "insufficient_roles",
        }

    held = roles_ok and local_ok and company_ok and unauthed_ok and viewer_ok
    detail: dict = {"index": index}
    if not held:
        detail.update(
            roles=roles_detail,
            local_unknown=local_detail,
            company_unknown=company_detail,
            unauthed=unauthed_detail,
            viewer_remember=viewer_detail,
        )
    return TrialOutcome(index=index, held=held, detail=detail)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(n_trials: int = 200, seed: int = 0) -> object:
    return run_trials(
        name="exp8_policy_registry",
        guarantee=(
            "_DEFAULT_REGISTRY.evaluate(): REMEMBER allowed with reason=='allow' "
            "for OWNER/ADMIN/MEMBER; VIEWER→REMEMBER denied with "
            "insufficient_roles; unknown kind fail-open in local mode "
            "(unknown_kind_allow_local), fail-closed in company mode "
            "(unknown_kind_deny_company); unauthenticated actor denied with "
            "authentication_required (auth check before role check)"
        ),
        metric_name="policy-correct rate",
        n_trials=n_trials,
        trial_fn=_trial,
        method=(
            "Pure CPU — no I/O. Imports _DEFAULT_REGISTRY singleton "
            "(operation_policy_registry.py, constructed once at import time). "
            "Five sub-assertions: authorized roles (OWNER/ADMIN/MEMBER) → "
            "allow with reason=='allow'; VIEWER→REMEMBER → insufficient_roles; "
            "unknown kind+local → unknown_kind_allow_local; "
            "unknown kind+company → unknown_kind_deny_company; "
            "empty principal_id → authentication_required (auth before role)."
        ),
    )


if __name__ == "__main__":
    from _harness import write_result

    result = run()
    print(write_result(result, Path(__file__).parent / "results"))
    print(f"{result.name}: {result.held}/{result.trials} ({result.metric_value:.4f})")
