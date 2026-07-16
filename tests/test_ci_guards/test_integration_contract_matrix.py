"""Executable truth gate for the named SLM client/integration matrix.

The matrix is deliberately evidence-level aware: generating a valid client
configuration is not the same claim as launching that external client.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.integration_compatibility import (
    build_evidence,
    load_manifest,
    validate_local_contracts,
)
from superlocalmemory.hooks.portable_kit import IDE_MATRIX


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "ide" / "integration-contracts.json"
EVIDENCE = ROOT / "ide" / "integration-compatibility-evidence.json"

REQUIRED_CONTRACTS = {
    "config_generation_parse",
    "start_command_shape",
    "memory_surface",
    "reconnect",
    "context_injection",
    "uninstall_ownership",
}
ALLOWED_STATUS = {
    "proven_local",
    "static_only",
    "manual_only",
    "not_applicable",
    "not_proven",
}


def test_manifest_covers_every_portable_client_and_config_artifact() -> None:
    manifest = load_manifest(MANIFEST)
    clients = {item["id"]: item for item in manifest["clients"]}

    assert set(IDE_MATRIX) <= set(clients)
    assert len(clients) == len(manifest["clients"]), "client ids must be unique"

    mapped = {
        path
        for client in clients.values()
        for path in client.get("artifacts", [])
        if path.startswith("ide/configs/")
    }
    present = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "ide" / "configs").iterdir()
        if path.is_file()
    }
    assert mapped == present


def test_each_client_has_explicit_evidence_level_and_complete_contract_shape() -> None:
    manifest = load_manifest(MANIFEST)
    for client in manifest["clients"]:
        assert set(client["contracts"]) == REQUIRED_CONTRACTS
        for contract in client["contracts"].values():
            assert contract["status"] in ALLOWED_STATUS
            assert contract["evidence"], (
                f"{client['id']} must explain or prove every contract status"
            )
        if client["external_host_runtime_proven"]:
            assert client["test_level"] == "external_host_runtime"
        else:
            assert client["test_level"] != "external_host_runtime"


def test_local_contract_runner_executes_every_claimed_local_check(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    result = validate_local_contracts(ROOT, manifest, work_dir=tmp_path)

    assert result["failures"] == []
    assert result["checks_run"] > 0
    claimed = {
        (client["id"], name)
        for client in manifest["clients"]
        for name, contract in client["contracts"].items()
        if contract["status"] == "proven_local"
    }
    assert claimed == {
        (check["client_id"], check["contract"])
        for check in result["checks"]
        if check["status"] == "passed"
    }


def test_evidence_counts_do_not_conflate_local_contracts_with_host_runtime(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST)
    evidence = build_evidence(ROOT, manifest, work_dir=tmp_path)
    summary = evidence["summary"]

    assert evidence["schema"] == "superlocalmemory.integration-compatibility/v1"
    assert summary["named_clients"] == len(manifest["clients"])
    assert summary["external_host_runtime_proven"] == sum(
        bool(client["external_host_runtime_proven"])
        for client in manifest["clients"]
    )
    assert summary["external_host_runtime_proven"] < summary["named_clients"]
    assert summary["locally_contract_tested"] == sum(
        any(c["status"] == "proven_local" for c in client["contracts"].values())
        for client in manifest["clients"]
    )

    # Evidence must be deterministic and safe to check into release artifacts.
    assert evidence == build_evidence(ROOT, manifest, work_dir=tmp_path / "again")
    json.dumps(evidence, sort_keys=True)


def test_evidence_is_canonical_when_manifest_client_order_changes(
    tmp_path: Path,
) -> None:
    """Checked-in release evidence must not vary with input iteration order."""
    manifest = load_manifest(MANIFEST)
    reordered = deepcopy(manifest)
    reordered["clients"].reverse()

    assert build_evidence(manifest=manifest, repo=ROOT, work_dir=tmp_path) == (
        build_evidence(repo=ROOT, manifest=reordered, work_dir=tmp_path / "reordered")
    )


def test_checked_in_compatibility_evidence_matches_executable_matrix(
    tmp_path: Path,
) -> None:
    expected = build_evidence(ROOT, load_manifest(MANIFEST), work_dir=tmp_path)
    assert json.loads(EVIDENCE.read_text(encoding="utf-8")) == expected
