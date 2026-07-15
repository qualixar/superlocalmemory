"""Release guard for supported installation channels and public guidance."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]

ACTIVE_INSTALL_SURFACES = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/install-linux.md",
    "docs/troubleshooting.md",
    "scripts/install.sh",
    "scripts/install.ps1",
    "scripts/postinstall.js",
    "bin/slm-npm",
    "ide/integrations/llamaindex/README.md",
    "ide/integrations/llamaindex/llama_index/storage/chat_store/superlocalmemory/base.py",
    "src/superlocalmemory/cli/commands.py",
)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_active_install_surfaces_never_recommend_protected_python_bypass() -> None:
    violations: list[str] = []
    for relative in ACTIVE_INSTALL_SURFACES:
        source = _text(relative)
        if "--break-system-packages" in source or "get-pip.py" in source:
            violations.append(relative)
    assert violations == []


def test_active_install_surfaces_never_pipe_remote_code_to_a_shell() -> None:
    remote_pipe = re.compile(
        r"(?:curl|wget|irm|invoke-restmethod)[^\n|]*\|[^\n]*(?:sh|bash|iex)",
        re.IGNORECASE,
    )
    violations = [
        relative
        for relative in ACTIVE_INSTALL_SURFACES
        if remote_pipe.search(_text(relative))
    ]
    assert violations == []


def test_active_install_surfaces_never_recommend_privileged_package_install() -> None:
    violations = [
        relative
        for relative in ACTIVE_INSTALL_SURFACES
        if re.search(r"\bsudo\s+(?:npm|pip|pip3)\s+install\b", _text(relative))
    ]
    assert violations == []


def test_readme_distinguishes_published_channels_from_clone_installers() -> None:
    readme = _text("README.md")
    assert "Node 18+" in readme
    assert "package-owned virtual environment" in readme
    assert "Python virtual environment" in readme
    assert "Repository clone" in readme
    assert "./scripts/install.sh" in readme
    assert r".\scripts\install.ps1" in readme


def test_contributor_and_integration_guidance_uses_supported_paths() -> None:
    contributing = _text("CONTRIBUTING.md")
    llama_readme = _text("ide/integrations/llamaindex/README.md")
    llama_runtime_error = _text(
        "ide/integrations/llamaindex/llama_index/storage/chat_store/"
        "superlocalmemory/base.py"
    )

    assert "./scripts/install.sh" in contributing
    assert r".\scripts\install.ps1" in contributing
    assert "pip install superlocalmemory" in llama_readme
    assert "pip install superlocalmemory" in llama_runtime_error
