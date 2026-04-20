# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 SB-5

"""Tests for core.slm_disabled — the global SLM kill switch."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core import slm_disabled as sd


@pytest.fixture
def tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "slm_home"
    home.mkdir()
    monkeypatch.setenv("SLM_HOME", str(home))
    monkeypatch.delenv("SLM_DISABLE", raising=False)
    return home


def test_is_disabled_false_by_default(tmp_home: Path) -> None:
    assert sd.is_disabled() is False


def test_is_disabled_true_when_marker_present(tmp_home: Path) -> None:
    (tmp_home / ".disabled").write_text("reason", encoding="utf-8")
    assert sd.is_disabled() is True


def test_is_disabled_true_when_env_set(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLM_DISABLE", "1")
    assert sd.is_disabled() is True


def test_is_disabled_false_when_env_is_zero(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for val in ("", "0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("SLM_DISABLE", val)
        assert sd.is_disabled() is False, val


def test_is_disabled_true_when_env_is_yes_style(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for val in ("1", "true", "yes", "on", "anything_else"):
        monkeypatch.setenv("SLM_DISABLE", val)
        assert sd.is_disabled() is True, val


def test_write_marker_creates_file(tmp_home: Path) -> None:
    path = sd.write_marker("testing")
    assert path.exists()
    assert "testing" in path.read_text(encoding="utf-8")


def test_write_marker_creates_home_dir_if_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "not_yet_there"
    monkeypatch.setenv("SLM_HOME", str(home))
    monkeypatch.delenv("SLM_DISABLE", raising=False)
    assert not home.exists()
    path = sd.write_marker("")
    assert path.exists()


def test_remove_marker_true_on_hit(tmp_home: Path) -> None:
    sd.write_marker()
    assert sd.remove_marker() is True
    assert sd.is_disabled() is False


def test_remove_marker_false_on_miss(tmp_home: Path) -> None:
    assert sd.remove_marker() is False


def test_marker_path_respects_slm_home_override(tmp_home: Path) -> None:
    assert sd.marker_path().parent == tmp_home


def test_env_takes_precedence_over_missing_marker(
    tmp_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Marker absent, env set — still disabled.
    monkeypatch.setenv("SLM_DISABLE", "yes")
    assert sd.is_disabled() is True
