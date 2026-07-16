"""Black-box contract for the Unix/macOS SuperLocalMemory installer.

The installer is allowed to manage executable code through an already-installed
isolated tool manager.  It is not allowed to mutate Python, the operating
system, shell startup files, or SuperLocalMemory's durable data.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix shell-installer contracts run on Linux and macOS only",
)

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install.sh"


def _tool_shim(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.write_text(
        """#!/bin/sh
printf '%s\\n' \"$0 $*\" >> \"$CALL_LOG\"
if [ -n \"${ARG_LOG:-}\" ]; then
    printf 'argc=%s\\n' \"$#\" >> \"$ARG_LOG\"
    for argument in \"$@\"; do
        printf '<%s>\\n' \"$argument\" >> \"$ARG_LOG\"
    done
fi
if [ \"$(basename \"$0\")\" = uv ] && [ \"$1 $2\" = \"tool list\" ]; then
    printf '%s\\n' \"${UV_LIST_OUTPUT:-}\"
fi
if [ \"$(basename \"$0\")\" = pipx ] && [ \"$1 $2\" = \"list --short\" ]; then
    printf '%s\\n' \"${PIPX_LIST_OUTPUT:-}\"
fi
exit \"${TOOL_EXIT_CODE:-0}\"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.fixture
def isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    home = tmp_path / "home"
    home.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    call_log = tmp_path / "calls.log"
    env = {
        "HOME": str(home),
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "CALL_LOG": str(call_log),
        "LC_ALL": "C",
    }
    return env, bin_dir, call_log


def _run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(INSTALLER), *args],
        cwd="/",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _calls(call_log: Path) -> list[str]:
    if not call_log.exists():
        return []
    return call_log.read_text(encoding="utf-8").splitlines()


def test_script_contains_no_privileged_or_python_bootstrap_path() -> None:
    source = INSTALLER.read_text(encoding="utf-8")
    forbidden = {
        "sudo": r"\bsudo\b",
        "system pip": r"(?:python\d*\s+-m\s+pip|\bpip\d*\s+install)",
        "PEP 668 bypass": r"--break-system-packages",
        "remote pipe to shell": r"(?:curl|wget)[^\n|]*\|[^\n]*(?:sh|bash)",
        "data root mutation": r"(?:SLM_DATA_DIR|SL_MEMORY_PATH|SLM_HOME|\.superlocalmemory)",
    }
    violations = [name for name, pattern in forbidden.items() if re.search(pattern, source)]
    assert violations == []


def test_install_prefers_uv_and_does_not_modify_home(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")
    _tool_shim(bin_dir, "pipx")
    home_before = sorted(Path(env["HOME"]).rglob("*"))

    result = _run(env, "install", "--non-interactive")

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/uv tool install {ROOT}"]
    assert sorted(Path(env["HOME"]).rglob("*")) == home_before


def test_install_falls_back_to_pipx(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "pipx")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/pipx install {ROOT}"]


def test_detached_installer_defaults_to_published_package(
    isolated_env: tuple[dict[str, str], Path, Path], tmp_path: Path
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")
    detached = tmp_path / "detached" / "scripts" / "install.sh"
    detached.parent.mkdir(parents=True)
    shutil.copy2(INSTALLER, detached)

    result = subprocess.run(
        ["/bin/bash", str(detached), "install"],
        cwd="/",
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/uv tool install superlocalmemory"]


@pytest.mark.parametrize(
    ("manager", "action", "expected"),
    [
        ("uv", "upgrade", "uv tool upgrade superlocalmemory"),
        ("uv", "uninstall", "uv tool uninstall superlocalmemory"),
        ("pipx", "upgrade", "pipx upgrade superlocalmemory"),
        ("pipx", "uninstall", "pipx uninstall superlocalmemory"),
    ],
)
def test_explicit_manager_lifecycle_commands(
    isolated_env: tuple[dict[str, str], Path, Path],
    manager: str,
    action: str,
    expected: str,
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, manager)

    result = _run(
        env,
        action,
        "--manager",
        manager,
        *(["--package", "superlocalmemory"] if action == "upgrade" else []),
    )

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/{expected}"]
    if action == "uninstall":
        assert "memory data was preserved" in result.stdout.lower()


def test_upgrade_discovers_the_manager_that_owns_installation(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")
    _tool_shim(bin_dir, "pipx")
    env["PIPX_LIST_OUTPUT"] = "superlocalmemory 3.6.23"

    result = _run(env, "upgrade")

    assert result.returncode == 0, result.stderr
    calls = _calls(call_log)
    assert calls[-1] == f"{bin_dir}/pipx install --force {ROOT}"
    assert not any("uv tool upgrade" in call for call in calls)


def test_ambiguous_installation_requires_explicit_manager(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")
    _tool_shim(bin_dir, "pipx")
    env["UV_LIST_OUTPUT"] = "superlocalmemory v3.6.23"
    env["PIPX_LIST_OUTPUT"] = "superlocalmemory 3.6.23"

    result = _run(env, "uninstall")

    assert result.returncode != 0
    assert "--manager" in result.stderr
    assert not any("uninstall superlocalmemory" in call for call in _calls(call_log))


def test_dry_run_prints_but_does_not_execute_mutating_command(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")

    result = _run(env, "install", "--manager=uv", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert f"uv tool install {ROOT}" in result.stdout
    assert _calls(call_log) == []


@pytest.mark.parametrize(
    ("manager", "expected_prefix"),
    [
        ("uv", "uv tool install"),
        ("pipx", "pipx install --force"),
    ],
)
def test_candidate_wheel_upgrade_replaces_isolated_tool_environment(
    isolated_env: tuple[dict[str, str], Path, Path],
    tmp_path: Path,
    manager: str,
    expected_prefix: str,
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, manager)
    wheel = tmp_path / "superlocalmemory-3.7.0-py3-none-any.whl"
    wheel.touch()

    result = _run(
        env,
        "upgrade",
        "--manager",
        manager,
        "--package",
        str(wheel),
    )

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/{expected_prefix} {wheel}"]


def test_explicit_package_is_one_shell_safe_argument(
    isolated_env: tuple[dict[str, str], Path, Path], tmp_path: Path
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")
    wheel = tmp_path / "candidate wheels" / "superlocalmemory.whl"
    wheel.parent.mkdir()
    wheel.touch()
    argument_log = tmp_path / "arguments.log"
    env["ARG_LOG"] = str(argument_log)

    result = _run(env, "install", "--manager=uv", "--package", str(wheel))

    assert result.returncode == 0, result.stderr
    assert _calls(call_log) == [f"{bin_dir}/uv tool install {wheel}"]
    assert argument_log.read_text(encoding="utf-8").splitlines() == [
        "argc=3",
        "<tool>",
        "<install>",
        f"<{wheel}>",
    ]


def test_uninstall_rejects_package_override(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")

    result = _run(
        env,
        "uninstall",
        "--manager=uv",
        "--package",
        "/tmp/candidate.whl",
    )

    assert result.returncode != 0
    assert "--package" in result.stderr
    assert _calls(call_log) == []


def test_missing_manager_fails_with_non_mutating_guidance(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, _, call_log = isolated_env

    result = _run(env, "install")

    assert result.returncode != 0
    assert "uv or pipx" in result.stderr
    assert "curl" not in result.stderr
    assert "sudo" not in result.stderr
    assert _calls(call_log) == []


def test_tool_failure_is_propagated(
    isolated_env: tuple[dict[str, str], Path, Path],
) -> None:
    env, bin_dir, _ = isolated_env
    _tool_shim(bin_dir, "uv")
    env["TOOL_EXIT_CODE"] = "42"

    result = _run(env, "install")

    assert result.returncode == 42


@pytest.mark.parametrize("argument", ["destroy", "--manager=conda", "--unknown"])
def test_invalid_input_fails_closed(
    isolated_env: tuple[dict[str, str], Path, Path], argument: str
) -> None:
    env, bin_dir, call_log = isolated_env
    _tool_shim(bin_dir, "uv")

    result = _run(env, argument)

    assert result.returncode != 0
    assert _calls(call_log) == []
