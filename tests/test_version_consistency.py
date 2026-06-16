"""Regression test: all five version sources must agree on 3.6.14.

Work Package G — publish-blocker: skew between package.json (3.6.13),
pyproject.toml (3.6.13), __init__.__version__ (3.6.10), plugin-src/manifest.json
(3.6.14) and plugin-src/requirements.txt (==3.6.14) caused pip install to fail.

This test reads every source directly (no imports of heavy deps) so it runs
fast in any environment — including CI with no ML packages installed.
"""

import json
import re
from pathlib import Path

# Repo root is three levels up from this file:
# tests/ -> superlocalmemory/ -> (repo root)
_REPO_ROOT = Path(__file__).parent.parent

EXPECTED_VERSION = "3.6.14"


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
    """All five version sources must agree on the same value (3.6.14)."""
    sources = {
        "package.json": _read_package_json_version(),
        "pyproject.toml": _read_pyproject_version(),
        "__init__.__version__": _read_init_version(),
        "manifest.json": _read_manifest_version(),
        "requirements.txt pin": _read_requirements_pin(),
    }
    mismatches = {src: ver for src, ver in sources.items() if ver != EXPECTED_VERSION}
    assert not mismatches, (
        f"Version mismatch — sources not at {EXPECTED_VERSION!r}: {mismatches}"
    )
