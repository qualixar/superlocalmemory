"""Domain packages must stay importable without loading a machine-learning stack.

Heavy models live in worker processes. Importing a domain package in a fresh
interpreter must not pull torch, transformers or sentence_transformers into
``sys.modules``. Packages append themselves to ``PACKAGES`` as they are added.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"

PACKAGES = [
    "superlocalmemory.daemon",
    "superlocalmemory.mesh",
    "superlocalmemory.memory_core",
    "superlocalmemory.cache",
]

_PROBE = (
    "import sys, importlib; importlib.import_module({module!r}); "
    "print(sorted(m for m in ('torch','transformers','sentence_transformers') "
    "if m in sys.modules))"
)


@pytest.mark.parametrize("module", PACKAGES)
def test_import_loads_no_heavy_ml(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(_SRC)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", result.stdout
