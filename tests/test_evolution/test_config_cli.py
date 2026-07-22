# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.7.9 — evolution model config CLI

"""Tests for ``slm config set evolution.*_model`` and the enable advisory.

Covers validation, the ``auto``→"" normalisation, the cost advisory on
enable, and a full persist→load round-trip through ``SLMConfig`` (v3.7.9).

Author: Varun Pratap Bhardwaj / Qualixar
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from superlocalmemory.cli.commands import cmd_config
from superlocalmemory.core.config import SLMConfig


@pytest.fixture()
def isolated_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Point the data root at a tempdir so config writes are sandboxed."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


def _set(key: str, value: str) -> None:
    cmd_config(Namespace(action="set", key=key, value=value, json=False))


def test_set_valid_model_persists_and_loads(isolated_data_dir: Path) -> None:
    _set("evolution.mutation_model", "sonnet")

    cfg_file = isolated_data_dir / "config.json"
    raw = json.loads(cfg_file.read_text())
    assert raw["evolution"]["mutation_model"] == "sonnet"

    # Round-trips through the loader into a typed EvolutionConfig.
    loaded = SLMConfig.load(cfg_file)
    assert loaded.evolution.mutation_model == "sonnet"


def test_auto_normalises_to_empty_string(isolated_data_dir: Path) -> None:
    _set("evolution.verify_model", "auto")
    raw = json.loads((isolated_data_dir / "config.json").read_text())
    assert raw["evolution"]["verify_model"] == ""


def test_invalid_model_is_rejected(isolated_data_dir: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        _set("evolution.mutation_model", "gpt-5-ultra")
    assert exc.value.code == 1
    # Nothing persisted for the bad value.
    cfg_file = isolated_data_dir / "config.json"
    if cfg_file.exists():
        raw = json.loads(cfg_file.read_text())
        assert raw.get("evolution", {}).get("mutation_model") != "gpt-5-ultra"


def test_enable_prints_cost_advisory(
    isolated_data_dir: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _set("evolution.enabled", "true")
    out = capsys.readouterr().out
    assert "Skill evolution is now ON" in out
    assert "lowest-cost model" in out
    assert "evolution.enabled false" in out  # tells the user how to undo
