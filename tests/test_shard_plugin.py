# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""CI's suite parts: every test in exactly one, the same one every run."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests._portable import child_env_base
from tests._shard_plugin import parse_shard, shard_of

REPO = Path(__file__).resolve().parents[1]


def test_a_test_always_lands_in_the_same_part() -> None:
    # Pinned values: a change of hash would silently reshuffle CI's parts.
    assert [shard_of(f"tests/x.py::test_{i}", 3) for i in range(6)] == [2, 0, 1, 2, 1, 0]
    assert shard_of("tests/x.py::test_0", 1) == 0


def test_parts_cover_every_test_once_and_are_not_lopsided() -> None:
    ids = [f"tests/m{i % 40}.py::test_{i}" for i in range(3000)]
    sizes = [sum(1 for n in ids if shard_of(n, 3) == part) for part in range(3)]
    assert sum(sizes) == len(ids)
    assert min(sizes) > 0.9 * len(ids) / 3


@pytest.mark.parametrize("value", ["0/3", "4/3", "1/0", "x", "1-3"])
def test_a_bad_shard_is_refused(value) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard(value)


def _collected(tmp_path: Path, *extra: str) -> list[str]:
    # A child pytest of its own: this run's data root would read as live to it.
    env = {k: v for k, v in os.environ.items()
           if k not in ("SLM_DATA_DIR", "SL_MEMORY_PATH", "SLM_HOME")}
    env.update(PYTHONPATH=str(REPO), **child_env_base(tmp_path / "home"))
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
         "-p", "tests._shard_plugin", "--rootdir", str(tmp_path), "-c", os.devnull,
         str(tmp_path), *extra],
        capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert out.returncode in (0, 5), out.stdout + out.stderr
    return re.findall(r"^\S+::test_\w+", out.stdout, re.M)


def test_the_parts_of_a_real_run_add_up_to_the_whole(tmp_path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "".join(f"def test_{i}():\n    pass\n" for i in range(30)), encoding="utf-8")
    whole = _collected(tmp_path)
    parts = [_collected(tmp_path, "--shard", f"{p}/3") for p in (1, 2, 3)]
    assert len(whole) == 30
    assert sorted(sum(parts, [])) == sorted(whole)
    assert all(parts), "a part with no tests"
