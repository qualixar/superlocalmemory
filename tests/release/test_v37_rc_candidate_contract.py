# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Release metadata must be internally consistent and publishable."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_release_source_has_a_final_pep440_package_version() -> None:
    """Release packaging must not retain a development or local-version label."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)

    assert match is not None
    version = match.group(1)
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version)


def test_citation_metadata_matches_the_package_release() -> None:
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    citation = (_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = re.search(r'^version:\s*"([^"]+)"', citation, re.MULTILINE)
    release_date = re.search(r'^date-released:\s*"([^"]+)"', citation, re.MULTILINE)

    assert package_version is not None
    assert version is not None
    assert release_date is not None
    assert version.group(1) == package_version.group(1)
    assert date.fromisoformat(release_date.group(1)) <= date.today()
