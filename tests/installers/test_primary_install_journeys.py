"""Primary onboarding and installer-ownership contracts.

The two primary user journeys are:

* npm global CLI, backed by the npm package's private Python environment;
* ``python -m pip install`` after activating a user-created virtual environment.

Repository-clone lifecycle scripts are deliberately separate and are never
shipped as npm installation entry points.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ONBOARDING_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "api-reference.md",
    ROOT / "docs" / "cloud-backup.md",
    ROOT / "docs" / "distributed-deployment.md",
    ROOT / "docs" / "getting-started.md",
    ROOT / "docs" / "install-linux.md",
    ROOT / "docs" / "multi-machine.md",
    ROOT / "docs" / "optimize-overview.md",
    ROOT / "docs" / "proxy-setup.md",
    ROOT / "docs" / "troubleshooting.md",
    ROOT / "wiki-content" / "FAQ.md",
    ROOT / "wiki-content" / "Getting-Started.md",
    ROOT / "wiki-content" / "Home.md",
    ROOT / "wiki-content" / "Installation.md",
    ROOT / "wiki-content" / "Quick-Start-Tutorial.md",
)


def test_primary_docs_lead_with_npm_and_activated_venv_pip() -> None:
    for path in (ROOT / "README.md", ROOT / "wiki-content" / "Installation.md"):
        content = path.read_text(encoding="utf-8")
        npm_position = content.index("npm install -g superlocalmemory")
        venv_position = content.index("python3 -m venv .venv")
        pip_position = content.index("python -m pip install superlocalmemory", venv_position)
        clone_position = content.index("scripts/install.sh")

        assert npm_position < clone_position, path
        assert venv_position < clone_position, path
        assert venv_position < pip_position, path
        assert "primary" in content[npm_position - 200 : npm_position + 300].lower(), path
        assert "primary" in content[venv_position - 300 : pip_position + 200].lower(), path


def test_active_onboarding_never_recommends_bare_or_privileged_pip_for_slm() -> None:
    unsafe = re.compile(
        r"(?im)(?<!-m )\b(?:sudo\s+)?pip(?:3)?\s+install\b[^\n`]*superlocalmemory"
    )
    findings: list[str] = []
    for path in ONBOARDING_DOCS:
        content = path.read_text(encoding="utf-8")
        for match in unsafe.finditer(content):
            findings.append(f"{path.relative_to(ROOT)}: {match.group(0).strip()}")

    assert findings == []


def test_repository_clone_has_only_scoped_lifecycle_installers() -> None:
    assert not (ROOT / "install.sh").exists()
    assert not (ROOT / "install.ps1").exists()
    assert (ROOT / "scripts" / "install.sh").is_file()
    assert (ROOT / "scripts" / "install.ps1").is_file()

    for script_name in ("install.sh", "install.ps1"):
        references = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob(script_name)
            if ".backup" not in path.parts and ".claude" not in path.parts
        ]
        assert references == [f"scripts/{script_name}"]


def test_npm_artifact_owns_cli_runtime_but_not_repo_clone_installers() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["bin"] == {
        "slm": "./bin/slm-npm",
        "superlocalmemory": "./bin/slm-npm",
    }
    assert package["scripts"]["postinstall"] == "node scripts/postinstall.js"
    assert "scripts/install.sh" not in package["files"]
    assert "scripts/install.ps1" not in package["files"]

    result = subprocess.run(
        ["npm", "pack", "--dry-run", "--json", "--ignore-scripts"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    artifact = json.loads(result.stdout)[0]
    paths = {entry["path"] for entry in artifact["files"]}
    assert {"bin/slm-npm", "scripts/postinstall.js", "pyproject.toml"} <= paths
    assert "scripts/install.sh" not in paths
    assert "scripts/install.ps1" not in paths
    assert "install.sh" not in paths
    assert "install.ps1" not in paths


def test_python_artifact_exposes_slm_entrypoint_for_activated_venv() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.scripts]\nslm = "superlocalmemory.cli.main:main"' in pyproject


def test_npm_package_smoke_scripts_do_not_mutate_global_npm_state() -> None:
    for path in (
        ROOT / "scripts" / "test-npm-package.sh",
        ROOT / "scripts" / "test-npm-package.ps1",
    ):
        content = path.read_text(encoding="utf-8")
        assert "npm link" not in content
        assert "npm unlink" not in content
        assert "src/memory_store_v2.py" not in content
        assert '"install.sh"' not in content
        assert '"install.ps1"' not in content
