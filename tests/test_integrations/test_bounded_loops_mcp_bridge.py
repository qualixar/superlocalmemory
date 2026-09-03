"""Pure contract tests for the bounded-loops MCP evidence bridge."""

from __future__ import annotations

from pathlib import Path

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


def test_producer_identity_binds_the_console_launcher_to_its_package_source(
    tmp_path: Path,
) -> None:
    """Changing bounded_loops source changes the v2 producer identity."""
    from superlocalmemory.integrations.bounded_loops_mcp import _producer_identity

    venv = tmp_path / "venv"
    launcher = venv / "bin" / "bounded-loops-mcp"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(f"#!{venv / 'bin' / 'python'}\n")
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    source = venv / "lib" / "python3.13" / "site-packages" / "bounded_loops" / "mcp_server.py"
    source.parent.mkdir(parents=True)
    source.write_text("def main(): return 0\n")

    original = _producer_identity(launcher)
    source.write_text("def main(): return 1\n")
    changed = _producer_identity(launcher)

    assert original.startswith("sha256:")
    assert original != changed


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


@pytest.mark.asyncio
async def test_poller_enforces_its_own_terminal_run_bound() -> None:
    evidence_calls = 0

    async def call(name: str, arguments: dict) -> dict:
        nonlocal evidence_calls
        if name == "bl_capabilities":
            return {"status": "ok", "capabilities": {"evidence_contracts": [{
                "id": "bounded-loops.dev/slm-bridge/v1",
                "tool": "bl_graph_evidence",
                "operation": "observe_terminal_run",
            }]}}
        if name == "bl_graph_terminal_runs":
            return {
                "status": "ok", "contract": "bounded-loops.dev/slm-bridge/v1",
                "runs": [{"run_ref": f"run-{i}"} for i in range(101)],
            }
        evidence_calls += 1
        result = _evidence()
        result["run_ref"] = arguments["run_ref"]
        return {"status": "ok", "evidence": result}

    observed = await observe_terminal_runs(call, profile_id="alpha")
    assert len(observed) == 100
    assert evidence_calls == 100
