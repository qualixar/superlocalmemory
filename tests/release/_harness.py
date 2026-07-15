"""Fixture-owned helpers for black-box wheel and sdist verification."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import site
import subprocess
import sys
import tarfile
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

_SNAPSHOT_FILES = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "NOTICE",
    "AUTHORS.md",
    "ATTRIBUTION.md",
)


def _ignore_snapshot_paths(_directory: str, names: list[str]) -> set[str]:
    """Exclude generated/local state while retaining every package source."""
    ignored = {
        ".git",
        ".backup",
        ".claude",
        ".gitnexus",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
    }
    return {name for name in names if name in ignored or name.endswith(".egg-info")}


def copy_release_snapshot(destination: Path) -> Path:
    """Copy the current packaging inputs without mutating the checkout.

    A git archive is intentionally not used: the V3.7 implementation is still
    an uncommitted release candidate and must be present in the artifact under
    test.  Copying only the PEP 517 inputs also prevents historical ``dist/``
    files from being selected accidentally.
    """
    destination.mkdir(parents=True, exist_ok=False)
    for relative in _SNAPSHOT_FILES:
        source = REPO_ROOT / relative
        if source.exists():
            shutil.copy2(source, destination / relative)
    shutil.copytree(
        REPO_ROOT / "src",
        destination / "src",
        symlinks=True,
        ignore=_ignore_snapshot_paths,
    )
    return destination


def safe_child_env(home: Path) -> dict[str, str]:
    """Return a minimal test-owned process environment with no source path."""
    env: dict[str, str] = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "SLM_DATA_DIR": str(home / "slm-data"),
        "SL_MEMORY_PATH": str(home / "slm-data"),
        "SLM_HOME": str(home / "slm-data"),
    }
    for name in ("TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    return env


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded subprocess and retain useful failure evidence."""
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            "artifact subprocess failed\n"
            f"command: {command!r}\n"
            f"cwd: {cwd}\n"
            f"exit: {result.returncode}\n"
            f"stdout:\n{result.stdout[-8000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
    return result


@dataclass(frozen=True)
class BuiltArtifacts:
    snapshot: Path
    output_dir: Path
    wheel: Path
    sdist: Path


def build_snapshot(snapshot: Path, output_dir: Path) -> BuiltArtifacts:
    """Build exactly one wheel and one sdist from the copied snapshot."""
    output_dir.mkdir(parents=True, exist_ok=False)
    home = output_dir.parent / "build-home"
    home.mkdir(parents=True, exist_ok=True)
    env = safe_child_env(home)
    # The local build frontend may live in the invoking interpreter's user
    # site, whose location changes when HOME is isolated. Expose only the
    # build tool's site-packages directory; never expose the repository src/.
    build_spec = importlib.util.find_spec("build")
    assert build_spec is not None and build_spec.origin is not None, (
        "python-build is required for release artifact verification"
    )
    env["PYTHONPATH"] = str(Path(build_spec.origin).resolve().parent.parent)
    run_checked(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--sdist",
            "--wheel",
            "--outdir",
            str(output_dir),
        ],
        cwd=snapshot,
        env=env,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    sdists = sorted(output_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel, found: {wheels}"
    assert len(sdists) == 1, f"expected one sdist, found: {sdists}"
    return BuiltArtifacts(snapshot, output_dir, wheels[0], sdists[0])


def _venv_python(path: Path) -> Path:
    """Return the platform-specific interpreter path for ``path``."""
    if sys.platform == "win32":
        return path / "Scripts" / "python.exe"
    return path / "bin" / "python"


def _venv_has_pip(python: Path, home: Path) -> tuple[bool, str]:
    """Probe pip without importing the checkout or mutating the environment."""
    home.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [str(python), "-I", "-m", "pip", "--version"],
            cwd=home,
            env=safe_child_env(home),
            text=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def _expose_invoking_venv_dependencies(path: Path) -> None:
    """Expose only the invoking venv's installed dependencies to ``path``.

    ``--system-site-packages`` means the base interpreter's site, not the
    parent virtual environment that runs pytest.  Release tests intentionally
    reuse heavyweight runtime dependencies from that parent environment while
    installing SuperLocalMemory itself only from the candidate artifact.
    """
    if sys.prefix == sys.base_prefix:
        return
    prefix = Path(sys.prefix).resolve()
    parent_sites = []
    for raw_path in site.getsitepackages():
        candidate = Path(raw_path).resolve()
        if candidate.is_dir() and candidate.is_relative_to(prefix):
            parent_sites.append(str(candidate))
    if not parent_sites:
        return
    if sys.platform == "win32":
        target_site = path / "Lib" / "site-packages"
    else:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        target_site = path / "lib" / version / "site-packages"
    target_site.mkdir(parents=True, exist_ok=True)
    (target_site / "slm-release-parent-site.pth").write_text(
        "\n".join(parent_sites) + "\n",
        encoding="utf-8",
    )


def create_venv(path: Path) -> Path:
    """Create an artifact-test venv without invoking stdlib ``ensurepip``.

    Prefer an installed ``uv`` because uv-managed standalone Python builds on
    macOS require virtual-environment launchers with the correct dynamic-library
    layout; a stdlib-created launcher can fail before Python starts.  ``uv`` is
    run offline, without seed downloads, and inherits the base interpreter's
    system site packages (including pip).

    When ``uv`` is unavailable, release tests use a stdlib no-seed venv with
    system site packages.  Avoiding ``with_pip=True`` matters on Python builds
    where the ensurepip subprocess can abort before pytest can report a useful
    failure.

    If the shared pip is unavailable, an already-installed ``virtualenv`` is
    allowed to seed the same directory from its bundled wheels with network
    downloads disabled.  No installer or remote script is fetched.
    """
    attempts: list[str] = []
    uv = shutil.which("uv")
    if uv:
        home = path.parent / "uv-venv-home"
        home.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    uv,
                    "venv",
                    "--offline",
                    "--system-site-packages",
                    "--python",
                    sys.executable,
                    str(path),
                ],
                cwd=path.parent,
                env=safe_child_env(home),
                text=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(f"uv venv: {type(exc).__name__}: {exc}")
            result = None
        python = _venv_python(path)
        if result is not None and result.returncode == 0 and python.is_file():
            _expose_invoking_venv_dependencies(path)
            available, detail = _venv_has_pip(
                python, path.parent / "uv-venv-probe-home",
            )
            if available:
                return python
            attempts.append(f"uv venv pip probe: {detail or 'pip unavailable'}")
        elif result is not None:
            detail = (result.stderr or result.stdout).strip()
            attempts.append(
                f"uv venv: exit {result.returncode}: "
                f"{detail[-2000:] or 'no diagnostic output'}"
            )
    else:
        attempts.append("uv venv: executable not installed")

    try:
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(path)
    except Exception as exc:  # pragma: no cover - platform-specific failure
        attempts.append(f"stdlib venv: {type(exc).__name__}: {exc}")

    python = _venv_python(path)
    if python.is_file():
        _expose_invoking_venv_dependencies(path)
        available, detail = _venv_has_pip(python, path.parent / "venv-probe-home")
        if available:
            return python
        attempts.append(f"shared pip probe: {detail or 'pip unavailable'}")

    if importlib.util.find_spec("virtualenv") is not None:
        home = path.parent / "virtualenv-home"
        home.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "virtualenv",
                    "--system-site-packages",
                    "--no-download",
                    str(path),
                ],
                cwd=path.parent,
                env=safe_child_env(home),
                text=True,
                capture_output=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(f"virtualenv: {type(exc).__name__}: {exc}")
            result = None
        if result is not None and result.returncode == 0 and python.is_file():
            _expose_invoking_venv_dependencies(path)
            available, detail = _venv_has_pip(
                python, path.parent / "virtualenv-probe-home",
            )
            if available:
                return python
            attempts.append(
                f"virtualenv pip probe: {detail or 'pip unavailable'}"
            )
        elif result is not None:
            detail = (result.stderr or result.stdout).strip()
            attempts.append(
                f"virtualenv: exit {result.returncode}: "
                f"{detail[-2000:] or 'no diagnostic output'}"
            )
    else:
        attempts.append("virtualenv: module not installed")

    raise AssertionError(
        "could not create an offline release-test environment with pip\n"
        + "\n".join(f"- {attempt}" for attempt in attempts)
    )


def install_artifact(python: Path, artifact: Path, work_dir: Path) -> None:
    """Install only the artifact; dependencies come from system site packages."""
    home = work_dir / "install-home"
    home.mkdir(parents=True, exist_ok=True)
    env = safe_child_env(home)
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
    ]
    if artifact.name.endswith(".tar.gz"):
        command.append("--no-build-isolation")
        # A uv-created venv inherits the standalone interpreter's site packages,
        # not the invoking project's venv.  Expose only the already-installed
        # PEP 517 backend modules needed to build the local sdist; never expose
        # the repository or fetch build requirements from the network.
        backend_paths: list[str] = []
        for module_name in ("setuptools", "wheel"):
            spec = importlib.util.find_spec(module_name)
            if spec is not None and spec.origin is not None:
                site_packages = str(Path(spec.origin).resolve().parent.parent)
                if site_packages not in backend_paths:
                    backend_paths.append(site_packages)
        assert backend_paths, "setuptools is required for sdist verification"
        env["PYTHONPATH"] = os.pathsep.join(backend_paths)
    command.append(str(artifact))
    run_checked(command, cwd=work_dir, env=env)


def inspect_installed_package(python: Path, work_dir: Path) -> dict:
    """Inspect the installed distribution from outside the checkout."""
    code = """
import importlib.metadata as md
import json
import pathlib
import sys
import superlocalmemory
from superlocalmemory.infra import daemon_identity

root = pathlib.Path(superlocalmemory.__file__).resolve().parent
payload = {
    "module_file": str(pathlib.Path(superlocalmemory.__file__).resolve()),
    "identity_file": str(pathlib.Path(daemon_identity.__file__).resolve()),
    "version": md.version("superlocalmemory"),
    "ui_index": str(root / "ui" / "index.html"),
    "optimize_notice": str(root / "optimize" / "NOTICE"),
    "sys_path": [str(pathlib.Path(p).resolve()) for p in sys.path if p],
}
print(json.dumps(payload, sort_keys=True))
"""
    home = work_dir / "runtime-home"
    home.mkdir(parents=True, exist_ok=True)
    result = run_checked(
        [str(python), "-I", "-c", code],
        cwd=work_dir,
        env=safe_child_env(home),
    )
    return json.loads(result.stdout)


def wheel_names(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def sdist_names(sdist: Path) -> set[str]:
    with tarfile.open(sdist, "r:gz") as archive:
        return set(archive.getnames())


def unsafe_archive_names(names: Iterable[str]) -> list[str]:
    """Return absolute, parent-traversing, or platform-ambiguous members."""
    unsafe: list[str] = []
    for raw in names:
        normalized = raw.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or "\\" in raw:
            unsafe.append(raw)
    return unsafe


def forbidden_wheel_names(names: Iterable[str]) -> list[str]:
    """Reject local state, credentials, tests, and generated bytecode."""
    forbidden: list[str] = []
    for raw in names:
        path = PurePosixPath(raw)
        lowered = raw.lower()
        if (
            "__pycache__" in path.parts
            or ".backup" in path.parts
            or "tests" in path.parts
            or path.suffix in {".pyc", ".pyo", ".db", ".sqlite", ".pem", ".key"}
            or path.name in {".env", ".env.local", ".ds_store"}
            or "/.git/" in lowered
        ):
            forbidden.append(raw)
    return forbidden
