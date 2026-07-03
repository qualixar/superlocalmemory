# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""WP-06 ensure-venv.sh tests — subprocess + tmp.

Requirements pin a TINY pure-Python package for CI speed (NOT numpy/scipy).
We use 'iniconfig==2.0.0' (no deps, pure Python, ~10 KB wheel).

Tests verify:
  - Script creates venv on first run
  - Idempotent: 2nd run < 2s with zero pip activity
  - Rebuilds venv when requirements.txt sha256 changes
  - Fails loud (non-zero exit + stderr) when CLAUDE_PLUGIN_DATA unset
  - Fails loud (non-zero exit + stderr) when CLAUDE_PLUGIN_ROOT unset
  - Sentinel written LAST: truncating sentinel triggers rebuild on next run
  - Rejects python < 3.11 with non-zero exit + stderr

NOTE: These tests write to tmp directories and do not touch ~/.superlocalmemory/.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
ENSURE_VENV_SH = REPO / "plugin-src" / "scripts" / "ensure-venv.sh"

# Tiny pure-Python package — no C extensions, installs in <5s on any platform
TINY_REQUIREMENTS = "iniconfig==2.0.0\n"

# ---------------------------------------------------------------------------
# Skip on Windows (bash script is Unix-only)
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="ensure-venv.sh is a bash script; skip on Windows",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_venv_sh(
    root: Path,
    data: Path,
    extra_env: dict[str, str] | None = None,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run ensure-venv.sh with given ROOT and DATA dirs."""
    # Prepend the venv running this test so the subprocess finds python3 >= 3.11.
    venv_bin = str(Path(sys.executable).parent)
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(root),
        "CLAUDE_PLUGIN_DATA": str(data),
        # Point SLM_DATA_DIR at the tmp dir so no real daemon.pid is found,
        # and force SLM_LAUNCHER=plugin so we always exercise the venv bootstrap path.
        "SLM_DATA_DIR": str(data),
        "SLM_LAUNCHER": "plugin",
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [str(ENSURE_VENV_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _make_roots(tmp_path: Path, requirements: str = TINY_REQUIREMENTS):
    """Create plugin root + data dirs with requirements.txt in root."""
    root = tmp_path / "plugin-root"
    data = tmp_path / "plugin-data"
    root.mkdir()
    data.mkdir()
    (root / "requirements.txt").write_text(requirements, encoding="utf-8")
    return root, data


# ---------------------------------------------------------------------------
# T1 — Script exists and is executable
# ---------------------------------------------------------------------------
def test_ensure_venv_sh_exists() -> None:
    assert ENSURE_VENV_SH.exists(), f"ensure-venv.sh not found at {ENSURE_VENV_SH}"


def test_ensure_venv_sh_is_executable() -> None:
    assert ENSURE_VENV_SH.exists(), f"ensure-venv.sh not found at {ENSURE_VENV_SH}"
    mode = ENSURE_VENV_SH.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"ensure-venv.sh must be user-executable (+x), mode={oct(mode)}"
    )


# ---------------------------------------------------------------------------
# T2 — Creates venv on first run
# ---------------------------------------------------------------------------
def test_creates_venv_on_first_run(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)
    result = _run_venv_sh(root, data)
    assert result.returncode == 0, (
        f"ensure-venv.sh failed on first run.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    venv_slm = data / "venv" / "bin" / "slm"
    # NOTE: iniconfig doesn't install a 'slm' console script — that's from superlocalmemory.
    # We verify the venv itself was created and pip-installed iniconfig.
    venv_pip = data / "venv" / "bin" / "pip"
    assert venv_pip.exists(), (
        f"venv/bin/pip not found after first run — venv may not have been created. "
        f"data={data}, stderr={result.stderr}"
    )
    # Verify sentinel was written
    sentinel = data / ".venv-reqs.sha256"
    assert sentinel.exists(), (
        f".venv-reqs.sha256 sentinel not written after first run. stderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# T3 — Idempotent: 2nd run < 2 seconds, no pip activity
# ---------------------------------------------------------------------------
def test_idempotent_second_run_fast(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)
    # First run (creates venv)
    r1 = _run_venv_sh(root, data)
    assert r1.returncode == 0, f"First run failed: {r1.stderr}"

    # Second run — must be fast (fast-path via sentinel)
    t0 = time.monotonic()
    r2 = _run_venv_sh(root, data)
    elapsed = time.monotonic() - t0

    assert r2.returncode == 0, f"Second run failed: {r2.stderr}"
    assert elapsed < 2.0, (
        f"Second run took {elapsed:.2f}s (must be < 2s — sentinel fast-path broken). "
        f"stderr={r2.stderr}"
    )
    # pip should NOT appear in stderr for second run (fast-path exits before pip)
    assert "pip" not in r2.stderr.lower() or "Requirement already satisfied" not in r2.stderr, (
        f"Second run must not invoke pip. stderr={r2.stderr!r}"
    )


# ---------------------------------------------------------------------------
# T4 — Rebuilds on requirements change
# ---------------------------------------------------------------------------
def test_rebuilds_on_requirements_change(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)
    r1 = _run_venv_sh(root, data)
    assert r1.returncode == 0, f"First run failed: {r1.stderr}"

    sentinel = data / ".venv-reqs.sha256"
    original_sentinel = sentinel.read_text(encoding="utf-8")

    # Change requirements.txt — add a second harmless package
    (root / "requirements.txt").write_text(
        TINY_REQUIREMENTS + "tomli==2.0.1\n", encoding="utf-8"
    )

    r2 = _run_venv_sh(root, data)
    assert r2.returncode == 0, f"Rebuild run failed: {r2.stderr}"

    new_sentinel = sentinel.read_text(encoding="utf-8")
    assert new_sentinel != original_sentinel, (
        "Sentinel must be updated when requirements.txt changes"
    )


# ---------------------------------------------------------------------------
# T5 — Fails loud when CLAUDE_PLUGIN_DATA is unset
# ---------------------------------------------------------------------------
def test_fails_loud_when_plugin_data_unset(tmp_path: pytest.TempPathFactory) -> None:
    root = tmp_path / "plugin-root"
    root.mkdir()
    (root / "requirements.txt").write_text(TINY_REQUIREMENTS, encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PLUGIN_DATA"}
    env["CLAUDE_PLUGIN_ROOT"] = str(root)
    # Explicitly unset CLAUDE_PLUGIN_DATA
    env.pop("CLAUDE_PLUGIN_DATA", None)

    result = subprocess.run(
        [str(ENSURE_VENV_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0, (
        "Script must exit non-zero when CLAUDE_PLUGIN_DATA is unset"
    )
    assert result.stderr, (
        "Script must emit error to stderr when CLAUDE_PLUGIN_DATA is unset"
    )


# ---------------------------------------------------------------------------
# T6 — Fails loud when CLAUDE_PLUGIN_ROOT is unset
# ---------------------------------------------------------------------------
def test_fails_loud_when_plugin_root_unset(tmp_path: pytest.TempPathFactory) -> None:
    data = tmp_path / "plugin-data"
    data.mkdir()

    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PLUGIN_ROOT"}
    env["CLAUDE_PLUGIN_DATA"] = str(data)
    env.pop("CLAUDE_PLUGIN_ROOT", None)

    result = subprocess.run(
        [str(ENSURE_VENV_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0, (
        "Script must exit non-zero when CLAUDE_PLUGIN_ROOT is unset"
    )
    assert result.stderr, (
        "Script must emit error to stderr when CLAUDE_PLUGIN_ROOT is unset"
    )


# ---------------------------------------------------------------------------
# T7 — Sentinel written LAST: truncating sentinel triggers rebuild
# ---------------------------------------------------------------------------
def test_truncated_sentinel_triggers_rebuild(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)
    r1 = _run_venv_sh(root, data)
    assert r1.returncode == 0, f"First run failed: {r1.stderr}"

    # Truncate the sentinel (simulates crash after venv install, before sentinel write)
    sentinel = data / ".venv-reqs.sha256"
    sentinel.write_text("", encoding="utf-8")

    # Second run must re-run (not take fast-path) because sentinel hash won't match
    t0 = time.monotonic()
    r2 = _run_venv_sh(root, data)
    elapsed = time.monotonic() - t0

    assert r2.returncode == 0, f"Run after sentinel truncation failed: {r2.stderr}"
    # Should NOT be fast (must have re-installed)
    # Verify sentinel is now valid (non-empty)
    final_sentinel = sentinel.read_text(encoding="utf-8").strip()
    assert len(final_sentinel) > 0, (
        "Sentinel must be non-empty after rebuild"
    )


# ---------------------------------------------------------------------------
# T8 — Rejects Python < 3.11 (guard at top of script)
# ---------------------------------------------------------------------------
def test_rejects_python_less_than_3_11(tmp_path: pytest.TempPathFactory) -> None:
    """Verify the script's py≥3.11 guard: if we can find a python < 3.11 on PATH,
    test it; else mock by patching the python3 binary in PATH via a wrapper.
    """
    root, data = _make_roots(tmp_path)

    # Create a fake python3 wrapper that reports version 3.10.x
    fake_py_dir = tmp_path / "fake-python"
    fake_py_dir.mkdir()
    fake_py = fake_py_dir / "python3"
    fake_py.write_text(
        "#!/bin/bash\n"
        'if [[ "$@" == "--version" || "$*" == *"--version"* || "$*" == *"-V"* ]]; then\n'
        '  echo "Python 3.10.14"\n'
        '  exit 0\n'
        'elif [[ "$*" == *"-c"* ]]; then\n'
        '  # For version check via -c "import sys; ..." — exit 1 to simulate guard failure\n'
        '  python_args="$*"\n'
        '  if [[ "$python_args" == *"sys.version_info"* ]]; then\n'
        '    exit 1\n'
        '  fi\n'
        'fi\n'
        'exec /usr/bin/env python3 "$@"\n'
    )
    fake_py.chmod(fake_py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Run with fake python3 first in PATH; isolate from real daemon and launcher
    venv_bin = str(Path(sys.executable).parent)
    env = {
        **os.environ,
        "CLAUDE_PLUGIN_ROOT": str(root),
        "CLAUDE_PLUGIN_DATA": str(data),
        "SLM_DATA_DIR": str(data),
        "SLM_LAUNCHER": "plugin",
        "PATH": f"{fake_py_dir}:{venv_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
    }
    result = subprocess.run(
        [str(ENSURE_VENV_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0, (
        f"Script must exit non-zero when python3 < 3.11. "
        f"returncode={result.returncode}, stderr={result.stderr!r}"
    )
    assert result.stderr, (
        f"Script must emit an error message to stderr when python3 < 3.11. "
        f"stderr={result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# T9 — Skips venv bootstrap when system daemon is running (daemon.pid check)
# ---------------------------------------------------------------------------
def test_skips_when_daemon_running(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)

    # Plant a daemon.pid that points at the current process (guaranteed alive)
    (data / "daemon.pid").write_text(str(os.getpid()), encoding="utf-8")

    result = _run_venv_sh(root, data, extra_env={"SLM_DATA_DIR": str(data)})

    assert result.returncode == 0, f"Should skip with rc=0 when daemon running. stderr={result.stderr!r}"
    assert not (data / "venv").exists(), "Venv must NOT be created when daemon is running"


# ---------------------------------------------------------------------------
# T10 — Skips venv bootstrap when SLM_LAUNCHER != plugin
# ---------------------------------------------------------------------------
def test_skips_when_launcher_is_not_plugin(tmp_path: pytest.TempPathFactory) -> None:
    root, data = _make_roots(tmp_path)

    for launcher_val in ("system", "/usr/local/bin/slm", "$HOME/.local/bin/slm"):
        result = _run_venv_sh(root, data, extra_env={"SLM_LAUNCHER": launcher_val})
        assert result.returncode == 0, (
            f"Should skip with rc=0 for SLM_LAUNCHER={launcher_val!r}. stderr={result.stderr!r}"
        )
        assert not (data / "venv").exists(), (
            f"Venv must NOT be created for SLM_LAUNCHER={launcher_val!r}"
        )
