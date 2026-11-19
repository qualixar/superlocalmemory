# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A registered flag must be read by something (Fable F-09).

``truth.leases`` was registered, documented and listed by ``slm flags``, and nothing read it:
``slm flags set truth.leases off`` changed nothing. The flag was removed (source leases follow
``truth.slots``). This keeps the next one from being added without a reader: every flag id must
appear, as a string, in a source file other than the registry itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import superlocalmemory
from superlocalmemory.core.feature_flags import FLAGS

SRC = Path(superlocalmemory.__file__).parent
REGISTRY = SRC / "core" / "feature_flags.py"


def _readers(flag_id: str) -> list[Path]:
    pattern = re.compile(rf"""["']{re.escape(flag_id)}["']""")
    return [path for path in SRC.rglob("*.py")
            if path != REGISTRY and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))]


@pytest.mark.parametrize("flag_id", sorted(FLAGS))
def test_every_registered_flag_has_a_reader(flag_id: str) -> None:
    assert _readers(flag_id), f"{flag_id} is registered but nothing in the source reads it"


def test_the_flag_without_a_reader_is_gone() -> None:
    assert "truth.leases" not in FLAGS
    assert _readers("truth.leases") == []
