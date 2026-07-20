"""Every shipped version and license surface follows canonical metadata."""

import json
import re
import tomllib
from pathlib import Path

# Repo root is three levels up from this file:
# tests/ -> superlocalmemory/ -> (repo root)
_REPO_ROOT = Path(__file__).parent.parent

def _read_package_json_version() -> str:
    """Read 'version' field from package.json."""
    path = _REPO_ROOT / "package.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["version"]


def _read_pyproject_version() -> str:
    """Read version = '...' from [project] section in pyproject.toml."""
    path = _REPO_ROOT / "pyproject.toml"
    content = path.read_text(encoding="utf-8")
    # Match: version = "3.6.xx" (standard PEP 621 inline)
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    assert match, "Could not find version field in pyproject.toml"
    return match.group(1)


def _read_init_version() -> str:
    """Read __version__ = '...' from src/superlocalmemory/__init__.py."""
    path = _REPO_ROOT / "src" / "superlocalmemory" / "__init__.py"
    content = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    assert match, "Could not find __version__ in __init__.py"
    return match.group(1)


def _read_manifest_version() -> str:
    """Read 'version' field from plugin-src/manifest.json."""
    path = _REPO_ROOT / "plugin-src" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["version"]


def _read_requirements_pin() -> str:
    """Read the version pinned in plugin-src/requirements.txt.

    Expected format: superlocalmemory==3.6.14
    """
    path = _REPO_ROOT / "plugin-src" / "requirements.txt"
    content = path.read_text(encoding="utf-8").strip()
    match = re.search(r"superlocalmemory==([^\s]+)", content)
    assert match, "Could not find superlocalmemory== pin in requirements.txt"
    return match.group(1)


def _read_json_version(relative: str, *, package_root: bool = False) -> str:
    data = json.loads((_REPO_ROOT / relative).read_text(encoding="utf-8"))
    return data["packages"][""]["version"] if package_root else data["version"]


def _read_requirement_pin(relative: str) -> str:
    content = (_REPO_ROOT / relative).read_text(encoding="utf-8")
    match = re.search(r"superlocalmemory==([^\s]+)", content)
    assert match, f"Could not find superlocalmemory== pin in {relative}"
    return match.group(1)


def _read_citation_field(field: str) -> str:
    content = (_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(rf'^{re.escape(field)}:\s*["\']?([^"\'\n]+)', content, re.MULTILINE)
    assert match, f"Could not find {field} in CITATION.cff"
    return match.group(1).strip()


def _read_uv_lock_version() -> str:
    with (_REPO_ROOT / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    matches = [
        package["version"]
        for package in lock["package"]
        if package.get("name") == "superlocalmemory"
    ]
    assert len(matches) == 1, f"Expected one superlocalmemory uv.lock entry: {matches}"
    return matches[0]


EXPECTED_VERSION = _read_pyproject_version()


# ---------------------------------------------------------------------------
# Individual assertions — each named so failures pinpoint the broken source
# ---------------------------------------------------------------------------


def test_package_json_version() -> None:
    """package.json 'version' must equal 3.6.14."""
    assert _read_package_json_version() == EXPECTED_VERSION, (
        f"package.json version is {_read_package_json_version()!r}, expected {EXPECTED_VERSION!r}"
    )


def test_pyproject_toml_version() -> None:
    """pyproject.toml [project] version must equal 3.6.14."""
    assert _read_pyproject_version() == EXPECTED_VERSION, (
        f"pyproject.toml version is {_read_pyproject_version()!r}, expected {EXPECTED_VERSION!r}"
    )


def test_init_version() -> None:
    """src/superlocalmemory/__init__.__version__ must equal 3.6.14."""
    assert _read_init_version() == EXPECTED_VERSION, (
        f"__init__.__version__ is {_read_init_version()!r}, expected {EXPECTED_VERSION!r}"
    )


def test_manifest_json_version() -> None:
    """plugin-src/manifest.json 'version' must equal 3.6.14 (read-only, not fixed here)."""
    assert _read_manifest_version() == EXPECTED_VERSION, (
        f"manifest.json version is {_read_manifest_version()!r}, expected {EXPECTED_VERSION!r}"
    )


def test_requirements_txt_pin() -> None:
    """plugin-src/requirements.txt must pin superlocalmemory==3.6.14 (read-only, not fixed here)."""
    assert _read_requirements_pin() == EXPECTED_VERSION, (
        f"requirements.txt pin is {_read_requirements_pin()!r}, expected {EXPECTED_VERSION!r}"
    )


def test_all_versions_consistent() -> None:
    """All source, lock, citation, and built-plugin versions must agree."""
    sources = {
        "package.json": _read_package_json_version(),
        "pyproject.toml": _read_pyproject_version(),
        "__init__.__version__": _read_init_version(),
        "manifest.json": _read_manifest_version(),
        "requirements.txt pin": _read_requirements_pin(),
        "package-lock.json": _read_json_version("package-lock.json"),
        "package-lock.json root": _read_json_version(
            "package-lock.json", package_root=True
        ),
        "plugin/.claude-plugin/plugin.json": _read_json_version(
            "plugin/.claude-plugin/plugin.json"
        ),
        "plugin/requirements.txt": _read_requirement_pin(
            "plugin/requirements.txt"
        ),
        "CITATION.cff": _read_citation_field("version"),
        "uv.lock": _read_uv_lock_version(),
    }
    mismatches = {src: ver for src, ver in sources.items() if ver != EXPECTED_VERSION}
    assert not mismatches, (
        f"Version mismatch — sources not at {EXPECTED_VERSION!r}: {mismatches}"
    )


def test_current_license_surfaces_are_agpl() -> None:
    package = json.loads((_REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (_REPO_ROOT / "package-lock.json").read_text(encoding="utf-8")
    )
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    signer = (
        _REPO_ROOT / "src" / "superlocalmemory" / "attribution" / "signer.py"
    ).read_text(encoding="utf-8")

    assert package["license"] == "AGPL-3.0-or-later"
    assert package_lock["packages"][""]["license"] == "AGPL-3.0-or-later"
    assert 'license = "AGPL-3.0-or-later"' in pyproject
    assert _read_citation_field("license") == "AGPL-3.0-or-later"
    assert '_LICENSE: str = "AGPL-3.0-or-later"' in signer


def test_product_owned_verification_scripts_use_current_license() -> None:
    scripts = (
        "scripts/verify-install.sh",
        "scripts/verify-install.ps1",
        "scripts/verify-v27.sh",
        "scripts/verify-v27.ps1",
        "scripts/test-npm-package.sh",
        "scripts/test-npm-package.ps1",
    )
    stale = {
        relative: "Licensed under MIT" in (_REPO_ROOT / relative).read_text(encoding="utf-8")
        for relative in scripts
    }
    assert not any(stale.values()), stale


def test_runtime_attribution_emits_current_license() -> None:
    from superlocalmemory.attribution.signer import QualixarSigner

    attribution = QualixarSigner("release-gate-test-key").sign("candidate")

    assert attribution["license"] == "AGPL-3.0-or-later"


def test_packaging_uses_pep639_license_metadata() -> None:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["project"]["license"] == "AGPL-3.0-or-later"
    assert metadata["project"]["license-files"] == ["LICENSE", "NOTICE"]
    assert not any(
        classifier.startswith("License ::")
        for classifier in metadata["project"]["classifiers"]
    )
    assert any(
        requirement.startswith("setuptools>=77.0.3")
        for requirement in metadata["build-system"]["requires"]
    )
