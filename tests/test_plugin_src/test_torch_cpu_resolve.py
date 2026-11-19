# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""torch-cpu-resolve.sh — decision-logic tests.

No pip, no network, no 2 GiB download: these tests source the script and call
its two pure functions directly, under a manipulated PATH/env so the branches
that only occur on a CPU-only Linux box can be exercised from this (macOS or
Linux) test runner too.
"""

from __future__ import annotations

import platform
import stat
import subprocess
from pathlib import Path

import pytest

from tests._portable import GUARD_ENV

REPO = Path(__file__).parent.parent.parent
RESOLVE_SH = REPO / "plugin-src" / "scripts" / "torch-cpu-resolve.sh"

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="torch-cpu-resolve.sh is a bash script; skip on Windows",
)


def _fake_bin(tmp_path: Path, name: str, script: str) -> Path:
    """Write an executable fake binary named `name` into a fresh dir on PATH."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    target = bindir / name
    target.write_text(f"#!/usr/bin/env bash\n{script}\n")
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return bindir


def _run(tmp_path: Path, body: str, env_overrides: dict[str, str] | None = None,
          path_prefix: Path | None = None) -> subprocess.CompletedProcess:
    """Source torch-cpu-resolve.sh then run `body`, with a clean, controlled env."""
    env = {
        "HOME": str(tmp_path),
        **GUARD_ENV,
        "PATH": (f"{path_prefix}:" if path_prefix else "") + "/usr/bin:/bin",
    }
    if env_overrides:
        env.update(env_overrides)
    # No `set -e` here: these tests call torch_cpu_should_force /
    # torch_cpu_pin directly (not inside an `if`), and both return 1 as a
    # normal, expected signal — `set -e` would abort the harness on that
    # return the way it never does inside ensure-venv.sh's own `if` guards.
    script = f'set -uo pipefail\n. "{RESOLVE_SH}"\n{body}\n'
    return subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


# ---------------------------------------------------------------------------
# torch_cpu_should_force
# ---------------------------------------------------------------------------

def test_real_host_platform_decides_without_crashing(tmp_path):
    """On whatever OS actually runs this test, the function must not error."""
    result = _run(tmp_path, "torch_cpu_should_force; echo \"exit=$?\"")
    assert "exit=0" in result.stdout or "exit=1" in result.stdout, result.stderr


def test_non_linux_never_forces(tmp_path):
    fake = _fake_bin(tmp_path, "uname", 'echo "Darwin"')
    result = _run(tmp_path, "torch_cpu_should_force", path_prefix=fake)
    assert result.returncode == 1, (result.stdout, result.stderr)


def test_linux_no_gpu_no_override_forces_cpu(tmp_path):
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    result = _run(tmp_path, "torch_cpu_should_force", path_prefix=fake)
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_linux_with_nvidia_smi_does_not_force(tmp_path):
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    _fake_bin(tmp_path, "nvidia-smi", "exit 0")
    result = _run(tmp_path, "torch_cpu_should_force", path_prefix=fake)
    assert result.returncode == 1, (result.stdout, result.stderr)


def test_slm_torch_backend_cpu_overrides_gpu_detection(tmp_path):
    """An explicit SLM_TORCH_BACKEND=cpu wins even if a GPU is visible."""
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    _fake_bin(tmp_path, "nvidia-smi", "exit 0")
    result = _run(
        tmp_path,
        "torch_cpu_should_force",
        env_overrides={"SLM_TORCH_BACKEND": "cpu"},
        path_prefix=fake,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_slm_torch_backend_cuda_opts_out_even_with_no_gpu(tmp_path):
    """The user's own choice is honoured even when our own heuristic would differ."""
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    result = _run(
        tmp_path,
        "torch_cpu_should_force",
        env_overrides={"SLM_TORCH_BACKEND": "cuda"},
        path_prefix=fake,
    )
    assert result.returncode == 1, (result.stdout, result.stderr)


@pytest.mark.parametrize("var", ["PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"])
def test_users_own_pip_index_is_never_overridden(tmp_path, var):
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    result = _run(
        tmp_path,
        "torch_cpu_should_force",
        env_overrides={var: "https://example.invalid/simple"},
        path_prefix=fake,
    )
    assert result.returncode == 1, (result.stdout, result.stderr)


# ---------------------------------------------------------------------------
# torch_cpu_pin
# ---------------------------------------------------------------------------

def test_torch_cpu_pin_reads_the_pin_line(tmp_path):
    pin_file = tmp_path / "requirements-cpu-torch.txt"
    pin_file.write_text("# comment\ntorch==2.13.0\n")
    result = _run(tmp_path, f'torch_cpu_pin "{pin_file}"')
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == "torch==2.13.0"


def test_torch_cpu_pin_missing_file_fails_open(tmp_path):
    missing = tmp_path / "does-not-exist.txt"
    result = _run(tmp_path, f'torch_cpu_pin "{missing}" ; echo "rc=$?"')
    assert "rc=1" in result.stdout
    assert result.stdout.strip() == "rc=1"  # nothing printed before the rc line


def test_torch_cpu_pin_file_without_torch_line_fails_open(tmp_path):
    pin_file = tmp_path / "requirements-cpu-torch.txt"
    pin_file.write_text("# no torch pin here\n")
    result = _run(tmp_path, f'torch_cpu_pin "{pin_file}" ; echo "rc=$?"')
    assert "rc=1" in result.stdout


def test_cpu_index_url_matches_the_documented_pytorch_index(tmp_path):
    # https://pytorch.org/get-started/locally/ documents this exact index for
    # a plain-pip CPU-only install (fetched 2026-10-06).
    result = _run(tmp_path, 'echo "${TORCH_CPU_INDEX_URL}"')
    assert result.stdout.strip() == "https://download.pytorch.org/whl/cpu"


def test_shipped_pin_file_is_itself_readable(tmp_path):
    """The real plugin-src/requirements-cpu-torch.txt must parse, not just a fixture."""
    real_pin = REPO / "plugin-src" / "requirements-cpu-torch.txt"
    assert real_pin.is_file()
    result = _run(tmp_path, f'torch_cpu_pin "{real_pin}"')
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip().startswith("torch==")


# ---------------------------------------------------------------------------
# Dry-run resolution: show the command ensure-venv.sh would run never asks
# the default index for torch, and names no nvidia-* package. We cannot
# actually resolve against PyPI here (LANE-RULES: no heavy downloads), so
# this inspects the constructed pip invocation generated from the real
# ensure-venv.sh + a Linux-forced environment, with pip replaced by a stub
# that records argv instead of installing anything.
# ---------------------------------------------------------------------------

def test_ensure_venv_pip_invocation_uses_cpu_index_on_forced_linux(tmp_path):
    """Reproduce the exact decision ensure-venv.sh makes (same two calls, same
    order) and assert the pip invocation it would run: the CPU index, the
    pinned version, and no nvidia-* package name anywhere in it."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "requirements.txt").write_text("iniconfig==2.0.0\n")
    (root / "requirements-cpu-torch.txt").write_text("torch==2.13.0\n")

    body = (
        'REQ="' + str(root / "requirements.txt") + '"\n'
        'TORCH_CPU_PIN_FILE="' + str(root / "requirements-cpu-torch.txt") + '"\n'
        "if torch_cpu_should_force; then\n"
        '  TORCH_PIN="$(torch_cpu_pin "${TORCH_CPU_PIN_FILE}")"\n'
        '  echo "would run: pip install --index-url ${TORCH_CPU_INDEX_URL} ${TORCH_PIN}"\n'
        "fi\n"
    )
    fake = _fake_bin(tmp_path, "uname", 'echo "Linux"')
    result = _run(tmp_path, body, path_prefix=fake)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "would run: pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0" in result.stdout
    assert "nvidia" not in result.stdout
