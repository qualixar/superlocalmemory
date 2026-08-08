"""Invariant F-07: trial_distribution must be COMPUTED from the actual n.

Confirmed defect: a 4-trial run emitted the 200-trial literal
"100 type-A (even index) + 100 type-B (odd index)". Provenance metadata that
is a literal is provenance metadata that lies at every n except one.

Paths are resolved relative to this file. Never hardcode an absolute path here:
this repository is published, and an author's home directory in a tracked file
is both an information leak and unrunnable for everyone else (see F-12).
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BENCHMARK = _REPO_ROOT / "benchmark"

# Experiments bind to the source tree under test via SLM_SOURCE_ROOT.
os.environ.setdefault("SLM_SOURCE_ROOT", str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_BENCHMARK))

exp2b = pytest.importorskip(
    "exp2b_real_owner_manifest",
    reason="benchmark suite not present",
)


# n=200 is deliberately excluded: it is the single value at which the old
# hardcoded literal was accidentally correct, so it cannot discriminate a
# computed value from a literal one. These four can, and run in a fraction
# of the time.
@pytest.mark.parametrize("n", (4, 5, 7, 10))
def test_distribution_is_computed_not_literal(n: int) -> None:
    result = exp2b.run(n_trials=n)
    dist = result.extra.get("trial_distribution", "")
    expected = f"{(n + 1) // 2} type-A (even index) + {n // 2} type-B (odd index)"
    assert dist == expected, f"n={n}: got {dist!r}, expected {expected!r}"
