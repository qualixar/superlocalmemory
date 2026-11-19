# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Shipped defaults and nothing else: no ``SLM_FEATURE_*`` variable, no kill switch, a data folder of its own.

Every test here proves what a user gets on the day after the upgrade, with the instruction firewall ON
(its default). Nothing in this folder may set a flag.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _shipped_defaults(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("SLM_FEATURE_") or name == "SLM_4230_OFF":
            monkeypatch.delenv(name)
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    from superlocalmemory.trust import firewall

    assert firewall.active(), "the firewall must be ON at the shipped defaults"
