"""Pure contract tests for the bounded-loops MCP evidence bridge."""

from __future__ import annotations

import pytest

from superlocalmemory.integrations.bounded_loops_mcp import (
    BridgeUnavailable,
    bridge_payload,
    observe_terminal_runs,
    supports_bridge,
)


def _evidence() -> dict:
    return {
        "contract": "bounded-loops.dev/slm-bridge/v1",
        "workspace_id": "sha256:" + "a" * 64,
        "run_id": "sandbox-demo-run",
        "run_ref": "nightly-1",
        "organization_id": "demo-org",
        "project_id": "demo-project",
        "outcome": "SUCCEEDED",
        "run_state": "SUCCEEDED",
        "demonstration": True,
        "eligible_for_learning": False,
        "terminal_at": "2026-08-15T15:53:42Z",
        "graph_digest": "sha256:" + "b" * 64,
        "plan_digest": "sha256:" + "c" * 64,
        "policy_digest": "sha256:" + "d" * 64,
        "receipt": {
            "sequence": 9,
            "head_digest": "sha256:" + "e" * 64,
            "trust": "local_hash_chain_only",
        },
        "nodes": [
            {
                "node_id": "probe",
                "state": "SUCCEEDED",
                "gate_passed": None,
                "attempts": 2,
                "artifact_digests": [],
            }
        ],
    }


def test_bridge_is_selected_by_contract_not_engine_version() -> None:
    assert supports_bridge(
        {
            "evidence_contracts": [
                {
                    "id": "bounded-loops.dev/slm-bridge/v1",
                    "tool": "bl_graph_evidence",
                    "operation": "observe_terminal_run",
                }
            ]
        }
    )
    assert not supports_bridge({"engine": {"version": "99.0.0"}})


def test_bridge_payload_is_observation_only_and_preserves_run_truth() -> None:
    payload = bridge_payload(_evidence(), profile_id="alpha")
    assert payload["profile_id"] == "alpha"
    assert payload["run_state"] == "SUCCEEDED"
    assert payload["eligible_for_learning"] is False
    assert payload["nodes"][0]["gate_passed"] is None
    assert "organization_id" not in payload


def test_unknown_contract_is_refused_without_a_compatibility_fallback() -> None:
    bad = _evidence()
    bad["contract"] = "bounded-loops.dev/slm-bridge/v2"
    with pytest.raises(BridgeUnavailable, match="unsupported"):
        bridge_payload(bad, profile_id="alpha")


@pytest.mark.asyncio
async def test_poller_fetches_only_terminal_run_refs_and_skips_normal_unavailability() -> None:
    calls: list[tuple[str, dict]] = []

    async def call(name: str, arguments: dict) -> dict:
        calls.append((name, arguments))
        if name == "bl_capabilities":
            return {
                "status": "ok",
                "capabilities": {
                    "evidence_contracts": [
                        {
                            "id": "bounded-loops.dev/slm-bridge/v1",
                            "tool": "bl_graph_evidence",
                            "operation": "observe_terminal_run",
                        }
                    ]
                },
            }
        if name == "bl_graph_terminal_runs":
            return {
                "status": "ok",
                "contract": "bounded-loops.dev/slm-bridge/v1",
                "runs": [
                    {
                        "run_ref": "nightly-1",
                        "run_id": "sandbox-demo-run",
                        "run_state": "SUCCEEDED",
                        "terminal_at": "2026-08-15T15:53:42Z",
                    },
                    {
                        "run_ref": "gone",
                        "run_id": "gone",
                        "run_state": "FAILED",
                        "terminal_at": "2026-08-15T15:53:41Z",
                    },
                ],
            }
        if arguments["run_ref"] == "gone":
            return {
                "status": "unavailable",
                "contract": "bounded-loops.dev/slm-bridge/v1",
                "reason": "no such run",
            }
        return {"status": "ok", "evidence": _evidence()}

    observed = await observe_terminal_runs(call, profile_id="alpha")
    assert [item["run_ref"] for item in observed] == ["nightly-1"]
    assert calls == [
        ("bl_capabilities", {}),
        ("bl_graph_terminal_runs", {"limit": 100}),
        ("bl_graph_evidence", {"run_ref": "nightly-1"}),
        ("bl_graph_evidence", {"run_ref": "gone"}),
    ]
