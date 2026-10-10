"""The memory-gate script, run end to end against the fake model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_media_ram.py"


def _run(*args: str, timeout=120):
    env = {**os.environ, "SLM_TEST_ISOLATION": "1"}
    env["PYTHONPATH"] = str(SCRIPT.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    done = subprocess.run([sys.executable, str(SCRIPT), "--fake", *args], capture_output=True,
                          text=True, timeout=timeout, env=env)
    out = done.stdout.strip().splitlines()
    return done, json.loads("\n".join(out[:-1])), out[-1]


def test_reports_idle_and_peak_for_both_loadouts_and_passes_a_generous_budget():
    done, report, verdict = _run("--images", "20", "--text-rounds", "2", "--budget-mb", "2000")
    assert done.returncode == 0, done.stderr
    assert set(report) == {"full", "text"}
    for loadout in report.values():
        assert loadout["idle_mb"] > 0
        assert loadout["peak_mb"] >= loadout["idle_mb"]
        assert loadout["text_peak_mb"] is not None
    assert report["full"]["picture_peak_mb"] is not None
    assert report["text"]["picture_peak_mb"] is None  # the text-only loadout has no picture tower
    assert verdict.startswith("PASS") and "2000 MB" in verdict


def test_an_impossible_budget_fails_the_verdict_with_exit_code_1():
    done, report, verdict = _run("--images", "4", "--text-rounds", "1", "--budget-mb", "1")
    assert done.returncode == 1
    assert verdict.startswith("FAIL") and "over" in verdict


def test_one_loadout_can_be_measured_alone():
    done, report, _verdict = _run("--loadouts", "text", "--images", "2", "--text-rounds", "1")
    assert done.returncode == 0 and set(report) == {"text"}


def test_without_the_fake_flag_a_missing_environment_is_a_clear_error(tmp_path):
    env = {**os.environ, "SLM_DATA_DIR": str(tmp_path), "SLM_TEST_ISOLATION": "1",
           "PYTHONPATH": str(SCRIPT.parents[1] / "src")}
    done = subprocess.run([sys.executable, str(SCRIPT), "--images", "1"], capture_output=True, text=True,
                          timeout=60, env=env)
    assert done.returncode != 0 and "slm media enable" in (done.stderr + done.stdout)
