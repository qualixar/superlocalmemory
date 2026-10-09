# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""End-to-end upgrade/downgrade over built fixtures (slow, needs real installs).

Environment:
  SLM_UPG_FIXTURES  directory holding ``<version>/manifest.json`` + ``<version>/data``
                    (default: ``tests/upgrade/fixtures``)
  SLM_UPG_OLD_PY    python of a venv with superlocalmemory 4.1.24 installed
  SLM_UPG_NEW_PY    python of a venv running this checkout (default: this interpreter)
  SLM_UPG_DOWN_PY   python of a venv with superlocalmemory 4.1.20 (optional; adds a second downgrade)
  SLM_UPG_VERSIONS  comma-separated fixture versions to run (default: all found)
  SLM_UPG_OUT       directory to write one verdict JSON per version (optional)

Run with: ``pytest tests/upgrade -m slow``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upgrade_check as uc  # noqa: E402

FIXTURES = Path(os.environ.get("SLM_UPG_FIXTURES", Path(__file__).resolve().parent / "fixtures"))
_ONLY = {v.strip() for v in os.environ.get("SLM_UPG_VERSIONS", "").split(",") if v.strip()}
VERSIONS = sorted(p.parent.name for p in FIXTURES.glob("*/manifest.json")
                  if (p.parent / "data" / "memory.db").exists() and (not _ONLY or p.parent.name in _ONLY))

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("version", VERSIONS or ["<none>"])
def test_fixture_upgrades_and_downgrades(version, tmp_path):
    if not VERSIONS:
        pytest.skip(f"no fixtures under {FIXTURES}; build with build_fixture.py")
    old_py = os.environ.get("SLM_UPG_OLD_PY")
    if not old_py:
        pytest.skip("SLM_UPG_OLD_PY (python with superlocalmemory 4.1.24) is not set")
    new_py = Path(os.environ.get("SLM_UPG_NEW_PY", sys.executable))
    down_py = os.environ.get("SLM_UPG_DOWN_PY")
    verdict = uc.run_checks(FIXTURES / version, Path(old_py), new_py, tmp_path,
                            Path(down_py) if down_py else None)
    out = os.environ.get("SLM_UPG_OUT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / f"verdict-{version}.json").write_text(json.dumps(verdict, indent=2, sort_keys=True))
    failed = {name: verdict["checks"][name].get("errors") for name in verdict["failing"]}
    assert not failed, failed
