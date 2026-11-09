# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Memory-kind settings: one home for every mode, strict about consent.

The Jev typing consent sends every memory off the device, so it is only ever
the literal boolean ``True``; a damaged or hand-edited file reads as "no".
The settings live in their own file (like the answer check's) so a mode
switch can never bring back a consent someone withdrew.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest

from superlocalmemory.core.memory_kind_config import (
    STATE_FILE,
    MemoryKindConfig,
    load_memory_kind_config,
    memory_kind_config_from,
    save_memory_kind_settings,
    settings_dict,
)


def test_unknown_keys_ignored() -> None:
    cfg = memory_kind_config_from({"enabled": False, "made_up": 1, "backend": "rules",
                                   "rate_per_second": {"rules": 50.0, "warp": 9.0}})
    assert cfg.enabled is False
    assert cfg.backend == "rules"
    assert cfg.rate_per_second["rules"] == 50.0
    assert "warp" not in cfg.rate_per_second
    assert not hasattr(cfg, "made_up")


def test_defaults_are_immutable() -> None:
    a = MemoryKindConfig()
    b = MemoryKindConfig()
    with pytest.raises(TypeError):
        a.rate_per_second["rules"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        a.batch_size["rules"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.enabled = False  # type: ignore[misc]
    assert a == b
    # deepcopy / asdict / replace must keep working on the whole SLMConfig.
    assert copy.deepcopy(a) == a
    assert dataclasses.replace(a, enabled=False).enabled is False
    json.dumps(settings_dict(a))


def test_jev_consent_is_only_the_literal_true() -> None:
    for value in ("true", "yes", 1, 1.0, "True", [True], None):
        assert memory_kind_config_from({"jev_consent": value}).jev_consent is False
    assert memory_kind_config_from({"jev_consent": True}).jev_consent is True


def test_bad_values_fall_back_to_safe_defaults() -> None:
    cfg = memory_kind_config_from({
        "enabled": "no", "backend": "skynet", "display_min_confidence": 7,
        "batch_size": {"rules": -5, "laya": "eight"},
        "rate_per_second": {"rules": float("nan"), "laya": 0},
    })
    default = MemoryKindConfig()
    assert cfg.enabled is True
    assert cfg.backend == "auto"
    assert cfg.display_min_confidence == default.display_min_confidence
    assert cfg.batch_size == default.batch_size
    assert cfg.rate_per_second == default.rate_per_second


def test_never_raises_on_garbage() -> None:
    for garbage in (None, 5, "x", [1, 2], {"rate_per_second": "fast"}):
        assert isinstance(memory_kind_config_from(garbage), MemoryKindConfig)


def test_file_wins_over_config_section_and_survives_a_mode_switch(tmp_path: Path) -> None:
    save_memory_kind_settings(tmp_path, {"jev_consent": False, "backend": "rules"})
    # A stale per-mode copy still says yes: the settings file is the authority.
    cfg = load_memory_kind_config(tmp_path, {"jev_consent": True, "backend": "jev"})
    assert cfg.jev_consent is False
    assert cfg.backend == "rules"


def test_section_used_when_no_file(tmp_path: Path) -> None:
    cfg = load_memory_kind_config(tmp_path, {"backend": "laya"})
    assert cfg.backend == "laya"


def test_damaged_file_fails_closed(tmp_path: Path) -> None:
    (tmp_path / STATE_FILE).write_text("{not json", encoding="utf-8")
    cfg = load_memory_kind_config(tmp_path, {"jev_consent": True})
    assert cfg.jev_consent is False


def test_save_is_private_and_merges(tmp_path: Path) -> None:
    save_memory_kind_settings(tmp_path, {"backend": "laya"})
    cfg = save_memory_kind_settings(tmp_path, {"enabled": False})
    assert cfg.backend == "laya" and cfg.enabled is False
    path = tmp_path / STATE_FILE
    if os.name != "nt":  # POSIX mode bits; Windows has none
        assert path.stat().st_mode & 0o777 == 0o600
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["backend"] == "laya" and stored["enabled"] is False


def test_slm_config_loads_the_section(tmp_path: Path) -> None:
    from superlocalmemory.core.config import SLMConfig

    (tmp_path / "config.json").write_text(json.dumps({"mode": "a"}), encoding="utf-8")
    save_memory_kind_settings(tmp_path, {"backend": "rules", "standing_rules_in_session": False})
    cfg = SLMConfig.load(tmp_path / "config.json")
    assert isinstance(cfg.memory_kinds, MemoryKindConfig)
    assert cfg.memory_kinds.backend == "rules"
    assert cfg.memory_kinds.standing_rules_in_session is False
    # The attribute names WP-3 and WP-7 read with getattr defaults.
    for name in ("enabled", "backend", "jev_consent", "display_min_confidence",
                 "standing_rules_in_session", "llm_extract_kinds"):
        assert hasattr(cfg.memory_kinds, name)
