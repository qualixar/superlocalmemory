"""Safety and lifecycle contract for the shipped Windows installer."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from tree_sitter_language_pack import get_parser

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install.ps1"


def _source() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def test_windows_installer_is_valid_powershell() -> None:
    tree = get_parser("powershell").parse(INSTALLER.read_bytes())
    assert not tree.root_node.has_error


def test_windows_installer_exposes_isolated_lifecycle_contract() -> None:
    source = _source()

    assert '[ValidateSet("Install", "Upgrade", "Uninstall")]' in source
    assert '[ValidateSet("Auto", "uv", "pipx")]' in source
    assert "\n    [string]$Package," in source
    assert "[switch]$DryRun" in source
    assert "[switch]$NonInteractive" in source
    assert '"tool", "install"' in source
    assert '"tool", "upgrade"' in source
    assert '"tool", "uninstall"' in source
    assert '"install", $packageSpec' in source
    assert '"upgrade", $PackageName' in source
    assert '"uninstall", $PackageName' in source


def test_windows_automatic_lifecycle_uses_installation_ownership() -> None:
    source = _source()

    assert "function Test-ManagerOwnsPackage" in source
    assert 'uv tool list' in source
    assert 'pipx list --short' in source
    assert "both uv and pipx own an installation" in source
    assert "no isolated installation was found" in source


def test_windows_installer_cannot_mutate_data_or_machine_configuration() -> None:
    source = _source().lower()
    forbidden = (
        "slm_data_dir",
        "sl_memory_path",
        "slm_home",
        ".superlocalmemory",
        "executionpolicy",
        "setenvironmentvariable",
        "invoke-webrequest",
        "invoke-restmethod",
        "invoke-expression",
        "start-bitstransfer",
        "copy-item",
        "new-item",
        "remove-item",
        "reg.exe",
        "hkey_",
        "-m pip",
        "pip install",
        "--break-system-packages",
        "ensurepath",
        "update-shell",
    )
    found = [token for token in forbidden if token in source]
    assert found == []


def _powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime contract runs in Windows CI")
@pytest.mark.parametrize(
    ("manager", "action", "expected"),
    [
        ("uv", "Install", "tool install"),
        ("uv", "Upgrade", "tool upgrade superlocalmemory"),
        ("uv", "Uninstall", "tool uninstall superlocalmemory"),
        ("pipx", "Install", "install"),
        ("pipx", "Upgrade", "upgrade superlocalmemory"),
        ("pipx", "Uninstall", "uninstall superlocalmemory"),
    ],
)
def test_windows_dry_run_emits_exact_lifecycle_command(
    manager: str, action: str, expected: str
) -> None:
    powershell = _powershell()
    assert powershell is not None, "Windows CI must provide Windows PowerShell or pwsh"

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(INSTALLER),
            "-Action",
            action,
            "-ToolManager",
            manager,
            "-DryRun",
            "-NonInteractive",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout.lower()
    if action == "Install":
        assert str(ROOT).lower() in result.stdout.lower()


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime contract runs in Windows CI")
def test_windows_dry_run_supports_an_exact_release_version() -> None:
    powershell = _powershell()
    assert powershell is not None, "Windows CI must provide Windows PowerShell or pwsh"

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(INSTALLER),
            "-Action",
            "Install",
            "-ToolManager",
            "uv",
            "-Version",
            "3.7.0",
            "-DryRun",
            "-NonInteractive",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "tool install superlocalmemory==3.7.0" in result.stdout.lower()


@pytest.mark.skipif(os.name != "nt", reason="Windows runtime contract runs in Windows CI")
def test_windows_dry_run_accepts_candidate_wheel_as_one_argument(tmp_path: Path) -> None:
    powershell = _powershell()
    assert powershell is not None, "Windows CI must provide Windows PowerShell or pwsh"
    wheel = tmp_path / "candidate wheel.whl"

    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-File",
            str(INSTALLER),
            "-Action",
            "Install",
            "-ToolManager",
            "uv",
            "-Package",
            str(wheel),
            "-DryRun",
            "-NonInteractive",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(wheel).lower() in result.stdout.lower()
