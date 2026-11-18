# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Hook tests run against their own data root, never the machine's real one.

The per-prompt hook writes a small per-turn marker under the data root (A-17), so a hook test that
reached the real ``~/.superlocalmemory`` would write there, and two tests sending the same prompt in
one session would see each other's marker.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _hook_tests_own_data_root(tmp_path, monkeypatch):
    if not os.environ.get("SLM_DATA_DIR"):
        monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
