# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""A first switch to a mode keeps the sections no dataclass models (4.2.14).

Codex review: with no mode_<new>.json yet, switch_mode() built the new config
with dataclasses.replace(), which drops the private copy of the loaded file.
Sections the dataclass does not model (temporal_validator, encoding, any
forward-compat key) were then left out of config.json and reverted to their
defaults on the next start.
"""

from __future__ import annotations

import json
from pathlib import Path

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def _seed_mode_b(base: Path, *, llm: dict) -> None:
    (base / "config.json").write_text(json.dumps({
        "mode": "b",
        "llm": llm,
        "temporal_validator": {"max_llm_checks": 0},
        "encoding": {"context_generation_llm": False},
        "x_forward_compat": {"kept": True},
    }), encoding="utf-8")


def _assert_kept_after_restart(base: Path) -> None:
    restarted = SLMConfig.load(base / "config.json")
    assert restarted.mode == Mode.C
    assert restarted.temporal_validator.max_llm_checks == 0
    assert restarted.encoding.context_generation_llm is False
    on_disk = json.loads((base / "config.json").read_text(encoding="utf-8"))
    assert on_disk["x_forward_compat"] == {"kept": True}


def test_first_switch_keeps_unmodelled_sections(tmp_path: Path) -> None:
    _seed_mode_b(tmp_path, llm={"provider": "ollama", "model": "llama3.2",
                                "base_url": "http://localhost:11434"})
    assert not (tmp_path / "mode_c.json").exists()
    SLMConfig.switch_mode("c", base_dir=tmp_path)
    _assert_kept_after_restart(tmp_path)


def test_first_switch_with_llm_defaults_filled_keeps_them(tmp_path: Path) -> None:
    # No provider: the switch fills in Mode C's LLM default, a second replace().
    _seed_mode_b(tmp_path, llm={"provider": "", "model": ""})
    switched = SLMConfig.switch_mode("c", base_dir=tmp_path)
    assert switched.llm.provider
    _assert_kept_after_restart(tmp_path)
