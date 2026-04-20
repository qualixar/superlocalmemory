# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regressions — outcome-hook common security hardening.

Covers:
  - SEC-M3: ``session_state`` dir created 0700.
  - SEC-M4: ``hook-perf.log`` parent dir 0700; log rotates at the
    configured cap.
  - SEC-M6: ``~/.superlocalmemory`` (SLM_HOME) first-creation 0700.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest


# POSIX-only — chmod semantics are different on Windows.
if sys.platform == "win32":  # pragma: no cover — not targeted on Windows
    pytest.skip("stage8 dir-perm tests are POSIX-only",
                allow_module_level=True)


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_sec_m6_slm_home_is_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "slm"))
    from superlocalmemory.hooks import _outcome_common as oc
    base = oc.slm_home()
    assert base.exists()
    assert _mode(base) == 0o700, oct(_mode(base))


def test_sec_m3_session_state_dir_is_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "slm"))
    from superlocalmemory.hooks import _outcome_common as oc
    d = oc.session_state_dir()
    assert d.exists()
    assert _mode(d) == 0o700, oct(_mode(d))


def test_sec_m4_logs_dir_is_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "slm"))
    from superlocalmemory.hooks import _outcome_common as oc
    p = oc.perf_log_path()
    # Parent dir is logs/ under SLM_HOME.
    assert p.parent.exists()
    assert _mode(p.parent) == 0o700, oct(_mode(p.parent))


def test_sec_m4_perf_log_rotates_over_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "slm"))
    from superlocalmemory.hooks import _outcome_common as oc

    # Reset module cached fd/path so this test doesn't bleed into others.
    oc._PERF_LOG_FD = None  # noqa: SLF001
    oc._PERF_LOG_PATH = None  # noqa: SLF001
    oc._PERF_LOG_WRITE_COUNT = 0  # noqa: SLF001

    # Tiny cap so we can exercise the rotate path without 10 MB of IO.
    monkeypatch.setattr(oc, "PERF_LOG_MAX_BYTES", 1024, raising=True)
    monkeypatch.setattr(oc, "PERF_LOG_CHECK_EVERY", 4, raising=True)

    log_path = oc.perf_log_path()
    # Write enough lines to exceed the 1 KB cap and trigger rotation.
    for i in range(200):
        oc.log_perf("stage8", 1.234, f"outcome_{i}")

    rotated = log_path.with_suffix(log_path.suffix + ".1")
    assert rotated.exists(), "rotation did not produce hook-perf.log.1"
    # Live log is under the cap after rotation (re-opened empty/small).
    assert log_path.exists()


def test_sec_m4_perf_log_entry_is_valid_ndjson(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "slm"))
    from superlocalmemory.hooks import _outcome_common as oc

    oc._PERF_LOG_FD = None  # noqa: SLF001
    oc._PERF_LOG_PATH = None  # noqa: SLF001
    oc._PERF_LOG_WRITE_COUNT = 0  # noqa: SLF001

    oc.log_perf("stage8_nd", 0.5, "ok")
    # Flush the cached fd so the disk copy is complete for the assertion.
    if oc._PERF_LOG_FD is not None:  # noqa: SLF001
        oc._PERF_LOG_FD.flush()  # noqa: SLF001

    line = oc.perf_log_path().read_text(encoding="utf-8").splitlines()[-1]
    rec = json.loads(line)
    assert rec["hook"] == "stage8_nd"
    assert rec["outcome"] == "ok"
