# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The first-run default: one model for text and pictures, behind one constant (off)."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core import one_model_default as omd
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_models

GIB = 1024 ** 3
EG2 = "google/embeddinggemma-2"


def test_the_constant_is_off_and_the_floor_is_sixteen_gigabytes():
    assert media_models.ONE_MODEL_DEFAULT_ENABLED is False
    assert media_models.ONE_MODEL_MIN_RAM_GB == 16


def _fresh(tmp_path):
    return SLMConfig.for_mode(__import__("superlocalmemory.storage.models", fromlist=["Mode"]).Mode.A,
                              base_dir=tmp_path)


def test_flag_off_leaves_a_first_run_config_byte_identical(tmp_path):
    config = _fresh(tmp_path)
    before = json.dumps(config.embedding.__dict__, sort_keys=True)
    assert omd.apply_first_run_default(config, tmp_path, total_ram_bytes=64 * GIB) is False
    assert json.dumps(config.embedding.__dict__, sort_keys=True) == before
    assert list(p.name for p in tmp_path.iterdir() if p.name == "features.json") == []


def test_flag_on_with_sixteen_gigabytes_picks_the_managed_model_and_records_the_request(tmp_path, monkeypatch):
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    config = _fresh(tmp_path)
    assert omd.apply_first_run_default(config, tmp_path, total_ram_bytes=16 * GIB) is True
    assert (config.embedding.provider, config.embedding.model_name, config.embedding.dimension) == (
        "slm-media", EG2, 768)
    media = feat.read_features(tmp_path)["media"]
    assert media["requested"] is True and not media["enabled"]


def test_flag_on_with_four_gigabytes_keeps_the_built_in_model(tmp_path, monkeypatch):
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    config = _fresh(tmp_path)
    before = config.embedding
    assert omd.apply_first_run_default(config, tmp_path, total_ram_bytes=4 * GIB) is False
    assert config.embedding == before and not (tmp_path / "features.json").exists()


def test_flag_on_with_eight_gigabytes_is_still_below_the_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    assert omd.apply_first_run_default(_fresh(tmp_path), tmp_path, total_ram_bytes=8 * GIB) is False


def test_unknown_memory_never_flips_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    assert omd.apply_first_run_default(_fresh(tmp_path), tmp_path, total_ram_bytes=0) is False


@pytest.mark.parametrize("marker", ["memory.db", "config.json"])
def test_an_existing_store_is_never_touched(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    (tmp_path / marker).write_text("{}" if marker.endswith("json") else "")
    config = _fresh(tmp_path)
    before = config.embedding
    assert omd.apply_first_run_default(config, tmp_path, total_ram_bytes=64 * GIB) is False
    assert config.embedding == before and not (tmp_path / "features.json").exists()


def test_the_first_boot_path_goes_through_the_gate(tmp_path, monkeypatch):
    """migrate_to_3mode on an empty folder writes the managed default only when the flag is on."""
    monkeypatch.setattr(media_models, "ONE_MODEL_DEFAULT_ENABLED", True)
    monkeypatch.setattr(omd, "_total_ram_bytes", lambda: 32 * GIB)
    assert SLMConfig.migrate_to_3mode(tmp_path) is True
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["embedding"]["provider"] == "slm-media"
    assert feat.read_features(tmp_path)["media"]["requested"] is True


def test_the_first_boot_path_is_unchanged_with_the_flag_off(tmp_path, monkeypatch):
    monkeypatch.setattr(omd, "_total_ram_bytes", lambda: 32 * GIB)
    assert SLMConfig.migrate_to_3mode(tmp_path) is True
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["embedding"]["provider"] != "slm-media"
    assert not (tmp_path / "features.json").exists()
