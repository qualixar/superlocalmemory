# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Two Mode B/C cost settings, read from config.json (#309).

Each stored fact costs one model call for its context sentence and up to five
for contradiction checks. Neither could be changed without patching SLM. Both
settings default to exactly what 4.2.13 did, so nobody's store changes on
upgrade; a user who sets them trades enrichment depth for throughput:

* ``temporal_validator.max_llm_checks``: 0 = the sheaf check only (the path
  Mode A, or Mode B with its model down, already takes);
* ``encoding.context_generation_llm``: false = rule-based context sentences.
"""

from __future__ import annotations

import dataclasses
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_4_2_13_config_loads_to_the_same_settings(tmp_path: Path) -> None:
    loaded = SLMConfig.load(_write(tmp_path / "config.json", {"mode": "b"}))
    preset = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    assert loaded.temporal_validator == preset.temporal_validator
    assert loaded.temporal_validator.max_llm_checks == 5
    assert loaded.encoding.context_generation_llm is True


def test_the_two_keys_change_only_their_own_fields(tmp_path: Path) -> None:
    loaded = SLMConfig.load(_write(tmp_path / "config.json", {
        "mode": "b",
        "temporal_validator": {"max_llm_checks": 0,
                               "superseded_demotion_factor": 0.0},  # not settable here
        "encoding": {"context_generation_llm": False},
    }))
    preset = SLMConfig.for_mode(Mode.B, base_dir=tmp_path).temporal_validator
    assert loaded.temporal_validator == dataclasses.replace(preset, max_llm_checks=0)
    assert loaded.encoding.context_generation_llm is False


@pytest.mark.parametrize("bad", [-1, True, "0", 2.5, None])
def test_a_bad_check_count_keeps_the_default(tmp_path: Path, bad: object) -> None:
    loaded = SLMConfig.load(_write(tmp_path / "config.json", {
        "mode": "b", "temporal_validator": {"max_llm_checks": bad}}))
    assert loaded.temporal_validator.max_llm_checks == 5


@pytest.mark.parametrize("bad", [0, "false", None, 1])
def test_a_bad_context_switch_keeps_the_default(tmp_path: Path, bad: object) -> None:
    loaded = SLMConfig.load(_write(tmp_path / "config.json", {
        "mode": "b", "encoding": {"context_generation_llm": bad}}))
    assert loaded.encoding.context_generation_llm is True


def test_the_keys_survive_save_and_a_mode_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("superlocalmemory.core.config.DEFAULT_BASE_DIR", tmp_path)
    monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
    _write(tmp_path / "config.json", {"mode": "b",
                                      "temporal_validator": {"max_llm_checks": 1},
                                      "encoding": {"context_generation_llm": False}})
    SLMConfig.load(tmp_path / "config.json").save(mode_change=True)
    SLMConfig.switch_mode("a", base_dir=tmp_path)
    back = SLMConfig.switch_mode("b", base_dir=tmp_path)
    assert back.temporal_validator.max_llm_checks == 1
    assert back.encoding.context_generation_llm is False


def test_zero_checks_builds_the_sheaf_only_validator(tmp_path: Path) -> None:
    from superlocalmemory.core.engine_wiring import _init_temporal

    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    llm = MagicMock()
    default = _init_temporal(config, MagicMock(), MagicMock(), llm)
    assert default._llm is llm
    config.temporal_validator = dataclasses.replace(config.temporal_validator,
                                                    max_llm_checks=0)
    sheaf_only = _init_temporal(config, MagicMock(), MagicMock(), llm)
    assert sheaf_only._llm is None


def test_context_switch_off_keeps_a_rule_based_generator(tmp_path: Path) -> None:
    from superlocalmemory.core.engine_wiring import _init_context_generator

    config = SLMConfig.for_mode(Mode.B, base_dir=tmp_path)
    llm = MagicMock()
    assert _init_context_generator(config, llm)._llm is llm
    config.encoding.context_generation_llm = False
    rules_only = _init_context_generator(config, llm)
    assert rules_only is not None  # auto_linker skips its work for a missing generator
    assert rules_only._llm is None


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


def _set(key: str, value: str) -> None:
    cmd = __import__("superlocalmemory.cli.commands", fromlist=["cmd_config"]).cmd_config
    cmd(Namespace(action="set", key=key, value=value, json=False))


def test_slm_config_set_writes_both_keys(data_dir: Path, capsys) -> None:
    _set("temporal_validator.max_llm_checks", "0")
    _set("encoding.context_generation_llm", "false")
    raw = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
    assert raw["temporal_validator"] == {"max_llm_checks": 0}
    assert raw["encoding"] == {"context_generation_llm": False}
    assert "slm restart" in capsys.readouterr().out


@pytest.mark.parametrize(("key", "value"), [
    ("temporal_validator.max_llm_checks", "-1"),
    ("temporal_validator.max_llm_checks", "many"),
    ("encoding.context_generation_llm", "2"),
])
def test_slm_config_set_refuses_bad_values(data_dir: Path, key: str, value: str) -> None:
    with pytest.raises(SystemExit):
        _set(key, value)
    assert not (data_dir / "config.json").exists()


def test_an_unreadable_config_is_left_alone(data_dir: Path, capsys) -> None:
    # 4.2.13 started from {} and wrote that back: every other setting was lost.
    (data_dir / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        _set("encoding.context_generation_llm", "false")
    assert (data_dir / "config.json").read_text(encoding="utf-8") == "{not json"
    assert "nothing was changed" in capsys.readouterr().out


def test_a_later_save_by_the_daemon_keeps_a_cli_change(tmp_path: Path) -> None:
    # Codex review, 4.2.14: the daemon's config object, loaded at start-up,
    # wrote sections it does not model from that old copy, so a dashboard save
    # after `slm config set` silently undid the CLI's change.
    from superlocalmemory.server.config_file import update_config

    path = _write(tmp_path / "config.json", {"mode": "b"})
    daemon_copy = SLMConfig.load(path)

    def cli_set(data: dict) -> None:
        data.setdefault("encoding", {})["context_generation_llm"] = False
        data.setdefault("temporal_validator", {})["max_llm_checks"] = 0

    update_config(path, cli_set)
    daemon_copy.save()  # a later save by the same long-lived object
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["encoding"]["context_generation_llm"] is False
    assert on_disk["temporal_validator"]["max_llm_checks"] == 0
    # Modelled sections (injection, retrieval) are the object's to write; the
    # daemon route now writes only the fields it changed
    # (tests/test_server/test_runtime_config_route.py).
