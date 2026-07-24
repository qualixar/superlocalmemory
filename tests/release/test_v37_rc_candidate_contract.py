# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The V3.7 release source must carry final publishable metadata."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_FINAL_RELEASE = "3.8.2"
_FINAL_RELEASE_DATE = "2026-07-23"


def test_v37_release_source_has_final_package_version() -> None:
    """Release packaging must not retain a local-version label."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None
    assert match.group(1) == _FINAL_RELEASE


def test_v37_citation_metadata_matches_the_release() -> None:
    citation = (_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = re.search(r'^version:\s*"([^"]+)"', citation, re.MULTILINE)
    release_date = re.search(r'^date-released:\s*"([^"]+)"', citation, re.MULTILINE)

    assert version is not None
    assert release_date is not None
    assert version.group(1) == _FINAL_RELEASE
    assert release_date.group(1) == _FINAL_RELEASE_DATE
