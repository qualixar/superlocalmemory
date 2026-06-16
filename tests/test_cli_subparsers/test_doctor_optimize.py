# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for WP-03: slm doctor Surface-B (optimize) stats.

Drives the ACTIVE cmd_doctor in cli/commands.py (dispatched via commands.py:123).
Does NOT extend or modify the orphan test_doctor.py (which tests doctor_cmd.py).
"""

from __future__ import annotations

import inspect
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> Namespace:
    defaults = {"json": False, "quick": True}
    defaults.update(kwargs)
    return Namespace(**defaults)


def _run_doctor_json(monkeypatch_fixture=None, patch_gather=None) -> list[dict]:
    """Run cmd_doctor with --json and return the checks list."""
    import io
    from superlocalmemory.cli.commands import cmd_doctor

    captured: list[str] = []

    def _fake_json_print(event, data=None, next_actions=None, **kw):
        captured.append(json.dumps(data or {}))

    with patch(
        "superlocalmemory.cli.json_output.json_print", side_effect=_fake_json_print
    ):
        if patch_gather is not None:
            with patch(
                "superlocalmemory.cli.commands._gather_optimize_surface_b",
                return_value=patch_gather,
            ):
                cmd_doctor(_make_args(json=True, quick=True))
        else:
            cmd_doctor(_make_args(json=True, quick=True))

    assert captured, "json_print was never called — doctor did not run"
    data = json.loads(captured[-1])
    return data.get("checks", [])


# ---------------------------------------------------------------------------
# Test 1 — enabled + healthy → PASS with all 4 stat tokens
# ---------------------------------------------------------------------------

def test_doctor_shows_optimize_section_when_enabled():
    """cmd_doctor --json must include an 'Optimize (Surface B)' check with
    PASS status and all 4 stat tokens when the gather helper returns healthy
    data."""
    gather_result = {
        "enabled": True,
        "cache_enabled": True,
        "compress_enabled": True,
        "proxy_enabled": False,
        "compress_runs": 42,
        "tokens_saved": 1000,
        "cache_hits": 7,
        "cache_misses": 3,
        "db_present": True,
        "error": "",
    }

    checks = _run_doctor_json(patch_gather=gather_result)

    opt_checks = [c for c in checks if c.get("name") == "Optimize (Surface B)"]
    assert opt_checks, f"'Optimize (Surface B)' not found in checks: {[c['name'] for c in checks]}"

    opt = opt_checks[0]
    assert opt["status"] == "PASS", f"Expected PASS, got {opt['status']!r}: {opt['detail']}"

    detail = opt["detail"]
    assert "compress_runs=" in detail, f"compress_runs= missing from detail: {detail!r}"
    assert "tokens_saved=" in detail, f"tokens_saved= missing from detail: {detail!r}"
    assert "cache_hits=" in detail, f"cache_hits= missing from detail: {detail!r}"
    assert "cache_misses=" in detail, f"cache_misses= missing from detail: {detail!r}"
    # Active surfaces must be reflected (cache + compress are on, proxy is off).
    assert "cache,compress" in detail, f"surfaces string missing from detail: {detail!r}"


# ---------------------------------------------------------------------------
# Test 2 — disabled → WARN + "disabled" + non-empty fix
# ---------------------------------------------------------------------------

def test_doctor_graceful_line_when_disabled():
    """When enabled=False, doctor must emit WARN containing 'disabled' and a
    non-empty fix string."""
    gather_result = {
        "enabled": False,
        "cache_enabled": True,
        "compress_enabled": False,
        "proxy_enabled": False,
        "compress_runs": 0,
        "tokens_saved": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "db_present": False,
        "error": "",
    }

    checks = _run_doctor_json(patch_gather=gather_result)

    opt_checks = [c for c in checks if c.get("name") == "Optimize (Surface B)"]
    assert opt_checks, f"'Optimize (Surface B)' not found in checks: {[c['name'] for c in checks]}"

    opt = opt_checks[0]
    assert opt["status"] == "WARN", f"Expected WARN, got {opt['status']!r}"
    assert "disabled" in opt["detail"], f"'disabled' not in detail: {opt['detail']!r}"
    assert opt.get("fix"), f"fix must be non-empty for disabled state, got: {opt.get('fix')!r}"


# ---------------------------------------------------------------------------
# Test 3 — real gather, no db, optimize.json enabled=true → no raise
# ---------------------------------------------------------------------------

def test_doctor_no_crash_when_metrics_empty_and_db_absent(tmp_path: Path):
    """With home→tmp_path (no llmcache.db), optimize.json enabled=true,
    _gather_optimize_surface_b must return enabled=True, db_present=False,
    compress_runs=0, and must NOT raise."""
    from superlocalmemory.cli.commands import _gather_optimize_surface_b

    # Write an optimize.json with enabled=true to tmp_path
    slm_dir = tmp_path / ".superlocalmemory"
    slm_dir.mkdir(parents=True, exist_ok=True)
    optimize_json = slm_dir / "optimize.json"
    optimize_json.write_text(json.dumps({"enabled": True}))

    # ConfigStore resolves its path from the MODULE-LEVEL _DEFAULT_CONFIG_PATH
    # (bound at import via Path.home()). Patching Path.home alone does NOT
    # redirect it — we must patch the module global too, or the helper reads
    # the developer's real ~/.superlocalmemory/optimize.json (machine-dependent).
    with patch("pathlib.Path.home", return_value=tmp_path), patch(
        "superlocalmemory.optimize.config.store._DEFAULT_CONFIG_PATH",
        optimize_json,
    ):
        result = _gather_optimize_surface_b()

    assert result["enabled"] is True, f"expected enabled=True, got {result['enabled']}"
    assert result["db_present"] is False, f"expected db_present=False, got {result['db_present']}"
    assert result["compress_runs"] == 0, f"expected compress_runs=0, got {result['compress_runs']}"

    # Also verify cmd_doctor itself doesn't raise
    from superlocalmemory.cli.commands import cmd_doctor
    with patch("pathlib.Path.home", return_value=tmp_path), patch(
        "superlocalmemory.optimize.config.store._DEFAULT_CONFIG_PATH",
        optimize_json,
    ):
        result_cmd = cmd_doctor(_make_args(json=False, quick=True))
    assert result_cmd is None  # cmd_doctor returns None on success


# ---------------------------------------------------------------------------
# Test 4 — metrics_load raises → error contains "metrics read failed", no raise
# ---------------------------------------------------------------------------

def test_doctor_no_crash_when_metrics_load_raises(tmp_path: Path):
    """If CacheDB.get_default().metrics_load raises, _gather_optimize_surface_b
    must catch it, set error containing 'metrics read failed', and NOT re-raise."""
    from superlocalmemory.cli.commands import _gather_optimize_surface_b

    mock_db = MagicMock()
    mock_db.metrics_load.side_effect = RuntimeError("sqlite gone")

    # Use a REAL on-disk llmcache.db so the helper's db_path.exists() is genuinely
    # True (avoids a brittle two-level __truediv__ mock that returns truthy MagicMocks
    # for the wrong reason). Only metrics_load is mocked to raise.
    slm_dir = tmp_path / ".superlocalmemory"
    slm_dir.mkdir(parents=True, exist_ok=True)
    (slm_dir / "llmcache.db").touch()

    # Patch CacheDB at the module where it lives (local import inside the helper).
    with patch("pathlib.Path.home", return_value=tmp_path), patch(
        "superlocalmemory.optimize.storage.db.CacheDB"
    ) as mock_cachedb_cls:
        mock_cachedb_cls.get_default.return_value = mock_db
        result = _gather_optimize_surface_b()

    assert result["db_present"] is True, "precondition: db file must exist on disk"
    assert "metrics read failed" in result.get("error", ""), (
        f"expected 'metrics read failed' in error, got: {result.get('error')!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — source of _gather_optimize_surface_b must NOT reference KV counters
# ---------------------------------------------------------------------------

def test_doctor_never_reads_kv_counters():
    """_gather_optimize_surface_b must NOT import tools_optimize or reference
    _kv_hits / _kv_misses — reading those would always return 0 in the doctor
    process (correctness bug)."""
    from superlocalmemory.cli.commands import _gather_optimize_surface_b

    source = inspect.getsource(_gather_optimize_surface_b)
    assert "tools_optimize" not in source, (
        "FORBIDDEN: _gather_optimize_surface_b references 'tools_optimize'"
    )
    assert "_kv_hits" not in source, (
        "FORBIDDEN: _gather_optimize_surface_b references '_kv_hits'"
    )
    assert "_kv_misses" not in source, (
        "FORBIDDEN: _gather_optimize_surface_b references '_kv_misses'"
    )


# ---------------------------------------------------------------------------
# Test 6 — existing checks unchanged; Optimize (Surface B) is last
# ---------------------------------------------------------------------------

def test_existing_doctor_checks_unchanged():
    """The pre-existing doctor checks must still be present by name, and
    'Optimize (Surface B)' must be the LAST entry in checks[]."""
    required_existing = {"Python", "Core deps", "Database"}

    checks = _run_doctor_json()

    names = [c["name"] for c in checks]
    names_set = set(names)

    missing = required_existing - names_set
    assert not missing, f"Pre-existing checks missing from doctor output: {missing}"

    assert names[-1] == "Optimize (Surface B)", (
        f"'Optimize (Surface B)' must be last check, but last is {names[-1]!r}. "
        f"All checks: {names}"
    )
