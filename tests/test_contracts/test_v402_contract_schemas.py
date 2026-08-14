"""Contract-freeze gates for the SLM 4.0.2 Brain release.

These schemas are consumed by separate brain, host, and bounded-loops lanes.
Keep their identity and authority boundaries explicit before implementation.
"""
from __future__ import annotations

import json
from importlib import resources

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from superlocalmemory.contracts.v402 import (
    ContractValidationError,
    validate_agent_experience,
    validate_cognitive_turn,
    validate_integration_certificate,
)


def _schema(name: str) -> dict:
    path = resources.files("superlocalmemory.contracts").joinpath("schemas", name)
    return json.loads(path.read_text(encoding="utf-8"))


def test_agent_experience_contract_separates_route_identity_and_authority() -> None:
    schema = _schema("agent-experience-v1.schema.json")

    assert schema["$id"].endswith("agent-experience-v1.schema.json")
    required = set(schema["required"])
    assert {
        "experience_id", "profile_id", "route", "verification", "terminal_status",
    } <= required
    route = schema["$defs"]["route_identity"]
    assert {"harness", "provider", "model", "effort", "machine"} <= set(route["required"])
    verification = schema["$defs"]["verification"]
    assert "producer_self_report" not in verification["properties"]["authority"]["enum"]
    assert "deterministic_gate" in verification["properties"]["authority"]["enum"]
    assert schema["properties"]["producer_claim"]["enum"] == [
        "success", "failure", "partial", "unknown",
    ]

    with pytest.raises(ContractValidationError, match="occurred_at"):
        validate_agent_experience({
            "experience_id": "x", "profile_id": "default", "occurred_at": "not-a-date",
            "task_class": "code", "project_scope": "opaque", "route": {
                "harness": "codex", "provider": "openai", "model": "model",
                "effort": "high", "machine": "opaque",
            },
            "verification": {"authority": "deterministic_gate", "evidence_digest": "0" * 64},
            "producer_claim": "success",
            "terminal_status": "succeeded",
        })

    validate_agent_experience({
        "experience_id": "experience", "profile_id": "default",
        "occurred_at": "2026-08-15T00:00:00+00:00", "task_class": "code",
        "project_scope": "opaque", "route": {
            "harness": "codex", "provider": "openai", "model": "model",
            "effort": "high", "machine": "opaque",
        },
        "verification": {"authority": "deterministic_gate", "evidence_digest": "0" * 64},
        "producer_claim": "success", "terminal_status": "succeeded",
    })


def test_cognitive_turn_contract_keeps_memory_use_bounded_to_one_fact_decision() -> None:
    schema = _schema("cognitive-turn-receipt-v1.schema.json")

    assert schema["$id"].endswith("cognitive-turn-receipt-v1.schema.json")
    required = set(schema["required"])
    assert {
        "receipt_id", "task_id", "project_scope", "query_digest",
        "profile_id", "fact_decisions", "state",
    } <= required
    assert schema["properties"]["state"]["enum"] == [
        "open", "finalized", "abandoned", "reconciled",
    ]
    assert schema["properties"]["query_digest"]["pattern"] == "^[a-f0-9]{64}$"
    fact_decisions = schema["properties"]["fact_decisions"]
    assert fact_decisions["type"] == "object"
    assert fact_decisions["additionalProperties"] is False
    assert fact_decisions["patternProperties"][".+"]["enum"] == [
        "considered", "used", "rejected", "corrected",
    ]

    legacy_conflicting_decisions = {
        "receipt_id": "receipt", "task_id": "task", "profile_id": "default",
        "project_scope": "opaque", "query_digest": "0" * 64,
        "considered_facts": [
            {"fact_id": "fact-a", "decision": "used"},
            {"fact_id": "fact-a", "decision": "rejected"},
        ],
        "state": "open",
    }
    errors = list(Draft202012Validator(schema).iter_errors(legacy_conflicting_decisions))
    assert any(error.validator == "required" for error in errors)
    assert any(error.validator == "additionalProperties" for error in errors)

    with pytest.raises(ContractValidationError, match="fact_decisions"):
        validate_cognitive_turn({
            "receipt_id": "receipt", "task_id": "task", "profile_id": "default",
            "project_scope": "opaque", "query_digest": "0" * 64,
            "fact_decisions": {"": "used"}, "state": "open",
        })

    validate_cognitive_turn({
        "receipt_id": "receipt", "task_id": "task", "profile_id": "default",
        "project_scope": "opaque", "query_digest": "0" * 64,
        "fact_decisions": {"fact-a": "used"}, "state": "open",
    })

    finalized_without_evidence = {
        "receipt_id": "receipt", "task_id": "task", "profile_id": "default",
        "project_scope": "opaque", "query_digest": "0" * 64,
        "fact_decisions": {"fact-a": "used"}, "state": "finalized",
    }
    errors = list(Draft202012Validator(schema).iter_errors(finalized_without_evidence))
    assert any(error.validator == "required" for error in errors)


def test_host_integration_contract_requires_end_to_end_lifecycle_evidence() -> None:
    schema = _schema("agent-integration-contract-v2.schema.json")

    assert schema["$id"].endswith("agent-integration-contract-v2.schema.json")
    stages = schema["$defs"]["certification_stage"]["enum"]
    assert stages == ["STATIC", "CONFIG_PROVEN", "LOCAL_RUNTIME", "END_TO_END", "RELEASE_GATED"]
    required = schema["properties"]["lifecycle"]["items"]["enum"]
    assert required == [
        "INSTALL", "CONFIG_PARSE", "PROCESS_START", "MCP_DISCOVERY", "SESSION_OPEN",
        "TARGETED_RECALL", "CONTEXT_INJECTION", "MEMORY_WRITE", "OUTCOME_CAPTURE",
        "RECONNECT", "SESSION_CLOSE", "UNINSTALL", "CONFIG_PRESERVATION", "SECRET_BOUNDARY",
    ]
    assert schema["properties"]["lifecycle"]["minItems"] == len(required)
    assert schema["properties"]["lifecycle"]["maxItems"] == len(required)
    assert schema["properties"]["lifecycle"]["uniqueItems"] is True

    invalid = {
        "schema_version": "v2",
        "host": "codex",
        "stage": "END_TO_END",
        "lifecycle": ["INSTALL"] * len(required),
        "artifact_digest": "0" * 64,
    }
    errors = list(Draft202012Validator(schema).iter_errors(invalid))
    assert any(error.validator == "uniqueItems" for error in errors)

    with pytest.raises(ContractValidationError, match="evidence"):
        validate_integration_certificate({
            "schema_version": "v2", "host": "codex", "stage": "RELEASE_GATED",
            "lifecycle": required, "artifact_digest": "0" * 64,
        })

    validate_integration_certificate({
        "schema_version": "v2", "host": "codex", "stage": "RELEASE_GATED",
        "lifecycle": required, "artifact_digest": "0" * 64,
        "evidence": [{"kind": "test_run", "digest": "1" * 64, "reference": "test-run"}],
    })
