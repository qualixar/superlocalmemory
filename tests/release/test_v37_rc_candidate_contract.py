# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The laptop-only V3.7 witness build must be unmistakably non-publishable."""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_CANDIDATE = "3.7.0+rc.1"


def test_v37_witness_candidate_has_an_explicit_local_build_label() -> None:
    """Avoid confusing a local witness artifact with a registry publication."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None
    assert match.group(1) == _LOCAL_CANDIDATE
