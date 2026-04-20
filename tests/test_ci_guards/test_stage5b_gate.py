"""Tests for scripts/ci/stage5b_gate.sh — the Stage-5b contract gate.

Per IMPLEMENTATION-MANIFEST-v3.4.22-FINAL.md P0.1. The gate enforces LLD-00
contracts at CI time by grep-scanning the source tree for retired patterns
and contract violations.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "stage5b_gate.sh"


def _init_tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo rooted at tmp_path with a src/ subtree.

    The gate script uses `git rev-parse --show-toplevel`, so tests need a
    real repo. Returns the path (same as tmp_path for convenience).
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit",
         "--allow-empty", "-qm", "init"],
        cwd=tmp_path, check=True,
    )
    (tmp_path / "src").mkdir()
    return tmp_path


def _copy_script_into(tmp_path: Path) -> Path:
    """Copy the real gate script into a temp repo so its git-toplevel is scoped.

    Preserves executable bit.
    """
    dst = tmp_path / "scripts" / "ci" / "stage5b_gate.sh"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return dst


def _run_gate(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(cwd / "scripts" / "ci" / "stage5b_gate.sh")],
        cwd=cwd, capture_output=True, text=True,
    )


def test_gate_script_exists_and_is_executable() -> None:
    assert SCRIPT.exists(), f"missing gate script: {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "gate script must be executable"


def test_gate_passes_on_clean_code(tmp_path: Path) -> None:
    """A clean src/ tree (no retired patterns) should exit 0."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "clean.py").write_text(
        "def foo():\n    return 'no forbidden patterns here'\n"
    )
    result = _run_gate(repo)
    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_gate_fails_on_pending_observations(tmp_path: Path) -> None:
    """LLD-00 §1.2 retired pending_observations — any occurrence fails."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "bad.py").write_text(
        "CREATE_SQL = 'CREATE TABLE pending_observations (id TEXT)'\n"
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "pending_observations" in result.stdout


def test_gate_fails_on_wrong_finalize_outcome_signature(tmp_path: Path) -> None:
    """LLD-00 §2 finalize_outcome takes outcome_id only, not query_id."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "bad_call.py").write_text(
        "model.finalize_outcome(query_id='x', signals={})\n"
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "finalize_outcome" in result.stdout


def test_gate_fails_on_bare_fact_id_scan(tmp_path: Path) -> None:
    """LLD-00 §3 — must use HMAC validator, not bare substring scan."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "bad_hook.py").write_text(
        "for fid in known_fact_ids:\n"
        "    if fid in response_text:\n"
        "        signals.append(fid)\n"
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "fid in response_text" in result.stdout


def test_gate_fails_on_opus_model_reference(tmp_path: Path) -> None:
    """MASTER-PLAN D2 — no Opus in SLM-initiated LLM calls."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "bad_model.py").write_text(
        'LLM_MODEL = "claude-opus-4-7"\n'
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "claude-opus-4" in result.stdout


def test_gate_fails_on_action_outcomes_insert_pattern(tmp_path: Path) -> None:
    """SEC-C-05 — action_outcomes writes must go through the canonical API."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    # pattern is literal per manifest: "action_outcomes.*INSERT.*VALUES"
    (repo / "src" / "bad_sql.py").write_text(
        'SQL = "action_outcomes helper INSERT row VALUES ()"\n'
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "action_outcomes" in result.stdout


def test_gate_reports_all_failures_before_exit(tmp_path: Path) -> None:
    """Gate should accumulate multiple failures and report each."""
    repo = _init_tmp_repo(tmp_path)
    _copy_script_into(repo)
    (repo / "src" / "many_bad.py").write_text(
        "CREATE TABLE pending_observations (id TEXT);\n"
        'MODEL = "claude-opus-4-7"\n'
    )
    result = _run_gate(repo)
    assert result.returncode == 1
    assert "pending_observations" in result.stdout
    assert "claude-opus-4" in result.stdout


def test_gate_passes_on_real_slm_src_tree() -> None:
    """Regression: the live SLM src/ currently satisfies all 5 checks."""
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"live src/ tree violates Stage-5b gate\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
