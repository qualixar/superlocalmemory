"""Public-CLI boundary tests for bounded-loops v0.5.1 graph receipts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from superlocalmemory.integrations import bounded_loops_v051 as adapter
from superlocalmemory.integrations.bounded_loops_v051 import (
    BoundedLoopsReceiptError,
    verify_v051_graph_receipt,
)


def _status(*, run_state: str = "SUCCEEDED", verified: bool = False) -> dict[str, object]:
    return {
        "organization_id": "org-a",
        "project_id": "project-a",
        "run_id": "run-a",
        "run_state": run_state,
        "receipt_head_hash": "a" * 64,
        "receipt_sequence": 9,
        "demonstration": True,
        "verified": verified,
        "nodes": [{"node_id": "verify", "artifact_digests": ["sha256:" + "b" * 64]}],
    }
def _fake_bl(monkeypatch: pytest.MonkeyPatch, status: object) -> None:
    executable = "/usr/local/bin/bl"
    monkeypatch.setattr(
        "superlocalmemory.integrations.bounded_loops_v051.shutil.which", lambda _: executable
    )
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    def run(_executable: str, *arguments: str) -> str:
        if arguments == ("--version",):
            return "bl 0.5.1\n"
        if arguments[:2] == ("graph", "status"):
            return json.dumps(status)
        raise AssertionError(arguments)

    monkeypatch.setattr("superlocalmemory.integrations.bounded_loops_v051._run_command", run)


def test_v051_receipt_uses_the_public_verified_projection_and_never_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fake_bl(monkeypatch, _status())

    receipt = verify_v051_graph_receipt(run_dir)

    assert receipt.run_id == "run-a"
    assert receipt.terminal_status == "SUCCEEDED"
    assert receipt.receipt_digest == "sha256:" + "a" * 64
    assert receipt.artifact_digests == ("sha256:" + "b" * 64,)
    assert receipt.trust_level == "local_unverified"
    assert receipt.eligible_for_learning is False
    assert receipt.demonstration is True


def test_v051_receipt_rejects_nonterminal_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fake_bl(monkeypatch, _status(run_state="RUNNING"))

    with pytest.raises(BoundedLoopsReceiptError, match="not terminal"):
        verify_v051_graph_receipt(run_dir)


def test_v051_receipt_rejects_unexpected_authority_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fake_bl(monkeypatch, _status(verified=True))

    with pytest.raises(BoundedLoopsReceiptError, match="verification state"):
        verify_v051_graph_receipt(run_dir)


def test_v051_receipt_rejects_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fake_bl(monkeypatch, _status())
    monkeypatch.setattr(
        "superlocalmemory.integrations.bounded_loops_v051._run_command",
        lambda *_args, **_kwargs: "bl 0.5.2\n",
    )

    with pytest.raises(BoundedLoopsReceiptError, match="exactly v0.5.1"):
        verify_v051_graph_receipt(run_dir)


def test_v051_receipt_rejects_relative_or_symlinked_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(run_dir, target_is_directory=True)

    with pytest.raises(BoundedLoopsReceiptError, match="absolute"):
        verify_v051_graph_receipt("relative-run")
    with pytest.raises(BoundedLoopsReceiptError, match="real directory"):
        verify_v051_graph_receipt(linked)


def test_v051_receipt_rejects_unbound_or_malformed_node_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    status = _status()
    status["nodes"] = [{"node_id": "verify", "artifact_digests": ["not-a-digest"]}]
    _fake_bl(monkeypatch, status)

    with pytest.raises(BoundedLoopsReceiptError, match="node artifact"):
        verify_v051_graph_receipt(run_dir)


def test_v051_metadata_cannot_change_learning_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _fake_bl(monkeypatch, _status())

    receipt = verify_v051_graph_receipt(run_dir)
    assert receipt.eligible_for_learning is False
    assert receipt.trust_level == "local_unverified"


def test_v051_command_bounds_stdout_and_stderr() -> None:
    code = "import sys; sys.stdout.write('x' * 2100000); sys.stderr.write('y' * 2100000)"

    with pytest.raises(BoundedLoopsReceiptError, match="size limit"):
        adapter._run_command(sys.executable, "-c", code)


def test_v051_command_reaps_a_process_that_closes_pipes_then_hangs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "_TIMEOUT_SECONDS", 0.05)
    code = "import os, time; os.close(1); os.close(2); time.sleep(10)"
    started = time.monotonic()

    with pytest.raises(BoundedLoopsReceiptError, match="did not complete"):
        adapter._run_command(sys.executable, "-c", code)

    assert time.monotonic() - started < 1
