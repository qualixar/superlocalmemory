"""Install wheel and sdist, then prove execution comes from site-packages."""

from __future__ import annotations

from pathlib import Path

import pytest

import superlocalmemory as _slm

from ._harness import (
    REPO_ROOT,
    BuiltArtifacts,
    create_venv,
    inspect_installed_package,
    install_artifact,
)

# When SLM is installed EDITABLE in the interpreter running the tests (a dev
# checkout: `pip install -e .`), site-packages carries an `__editable__` .pth
# that points at REPO_ROOT/src. The release harness builds its probe venv with
# --system-site-packages (to reuse heavy ML deps), so that .pth leaks src/ onto
# the probe's sys.path and this clean-install isolation check cannot hold. The
# check is only meaningful for a non-editable/packaged install (CI), so skip it
# under an editable dev install rather than report a false failure.
_EDITABLE_DEV_INSTALL = Path(_slm.__file__).resolve().is_relative_to(
    REPO_ROOT / "src"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        _EDITABLE_DEV_INSTALL,
        reason="editable dev install exposes src/ via --system-site-packages; "
               "clean-install isolation is only verifiable for a packaged install (CI)",
    ),
]


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_installed_artifact_imports_outside_checkout(
    artifact_kind: str,
    built_artifacts: BuiltArtifacts,
    tmp_path: Path,
) -> None:
    artifact = getattr(built_artifacts, artifact_kind)
    venv_root = tmp_path / f"venv-{artifact_kind}"
    python = create_venv(venv_root)
    work_dir = tmp_path / f"runtime-{artifact_kind}"
    work_dir.mkdir()
    install_artifact(python, artifact, work_dir)

    inspection = inspect_installed_package(python, work_dir)
    module_file = Path(inspection["module_file"])
    identity_file = Path(inspection["identity_file"])

    assert module_file.is_relative_to(venv_root)
    assert identity_file.is_relative_to(venv_root)
    assert not module_file.is_relative_to(REPO_ROOT)
    assert not identity_file.is_relative_to(REPO_ROOT)
    assert Path(inspection["ui_index"]).is_file()
    assert Path(inspection["optimize_notice"]).is_file()
    assert all(
        not Path(entry).is_relative_to(REPO_ROOT / "src")
        for entry in inspection["sys_path"]
    )
