# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The V3.7 release source must carry final publishable metadata."""

from __future__ import annotations

import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_FINAL_RELEASE = "3.7.3"


def test_v37_release_source_has_final_package_version() -> None:
    """Release packaging must not retain a local-version label."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None
    assert match.group(1) == _FINAL_RELEASE
