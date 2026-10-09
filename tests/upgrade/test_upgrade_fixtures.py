# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""End-to-end upgrade/downgrade over built fixtures (slow, needs real installs).

Environment:
  SLM_UPG_FIXTURES  directory holding ``<version>/manifest.json`` + ``<version>/data``
                    (default: ``tests/upgrade/fixtures``)
  SLM_UPG_OLD_PY    python of a venv with superlocalmemory 4.1.24 installed
  SLM_UPG_NEW_PY    python of a venv running this checkout (default: this interpreter)

Run with: ``pytest tests/upgrade -m slow``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _slm_env as env  # noqa: E402
import upgrade_check as uc  # noqa: E402

FIXTURES = Path(os.environ.get("SLM_UPG_FIXTURES", Path(__file__).resolve().parent / "fixtures"))
VERSIONS = sorted(p.parent.name for p in FIXTURES.glob("*/manifest.json") if (p.parent / "data" / "memory.db").exists())

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("version", VERSIONS or ["<none>"])
def test_fixture_upgrades_and_downgrades(version, tmp_path):
    if not VERSIONS:
        pytest.skip(f"no fixtures under {FIXTURES}; build with build_fixture.py")
    old_py = os.environ.get("SLM_UPG_OLD_PY")
    if not old_py:
        pytest.skip("SLM_UPG_OLD_PY (python with superlocalmemory 4.1.24) is not set")
    new_py = Path(os.environ.get("SLM_UPG_NEW_PY", sys.executable))
    work = env.make_work_dir()
    verdict = uc.run_checks(FIXTURES / version, Path(old_py), new_py, work)
    failed = {name: c.get("errors") for name, c in verdict["checks"].items() if not c.get("passed")}
    assert not failed, failed
