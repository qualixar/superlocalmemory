# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SLMConfig.save() merges modelled fields three ways (4.2.14).

Codex review (GPT-6.1 Sol), two P1s:

P1-A: a long-lived config object (the daemon's start-up copy) wrote every
modelled field it held, so a later full save() undid `slm config set` for
fields the object never touched (injection.enabled, auto_invoke.enabled).

P1-B: a successful save did not record where it wrote. After a first-time
mode switch the returned object had no source file, so its later saves to
config.json used its stale loaded copy instead of the file as it is now.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.server.config_file import update_config
from superlocalmemory.storage.models import Mode

_BASE_DIR_TOKEN = "<base_dir>"

#: A config shaped like a real install: every modelled section, typed sections
#: with subkeys this version does not model, unmodelled sections, an opaque
#: embedding signature.
_REALISTIC = {
    "mode": "b",
    "active_profile": "default",
    "graph_backend": "auto",
    "vector_backend": "auto",
    "llm": {"provider": "ollama", "model": "llama3.2",
            "api_key": "", "base_url": "http://localhost:11434"},
    "embedding": {"model_name": "nomic-embed-text", "dimension": 768,
                  "provider": "ollama", "ollama_model": "nomic-embed-text",
                  "ollama_base_url": "http://localhost:11434"},
    "retrieval": {"top_k": 15, "use_cross_encoder": False,
                  "sufficiency_judge": "auto"},
    "evolution": {"enabled": False, "backend": "auto"},
    "injection": {"enabled": True, "core_block_max_facts": 7},
    "scope_weights": {"personal": 1.0, "shared": 0.8, "global_": 0.6},
    "graph_pruning": {"max_degree_per_node": 40, "min_edge_weight": 0.05,
                      "enabled": True},
    "hopfield": {"x_vendor_knob": 3},
    "forgetting": {"half_life_days": 45},
    "auto_invoke": {"enabled": True,
                    "weights": {"similarity": 0.4, "recency": 0.25,
                                "frequency": 0.2, "trust": 0.15}},
    "mesh_enabled": True,
    "mesh": {"retention_days": 9},
    "daemon_port": 8765,
    "temporal_validator": {"max_llm_checks": 0},
    "encoding": {"context_generation_llm": False},
    "x_forward_compat": {"kept": True},
    "embedding_signature": "opaque-signature-v1",
}

_GOLDEN = Path(__file__).with_name("data") / "config_three_way_golden.json"


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalised_bytes(path: Path) -> str:
    """The saved file with its machine-specific base_dir replaced by a token."""
    data = _read(path)
    text = path.read_text(encoding="utf-8")
    return text.replace(json.dumps(data["base_dir"])[1:-1], _BASE_DIR_TOKEN)


def test_unchanged_load_save_is_byte_identical_to_before(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    config = SLMConfig.load(path)
    config.save(path)
    first = _normalised_bytes(path)
    assert first == _GOLDEN.read_text(encoding="utf-8")
    # A second save of the same object, and a reload and save, change nothing.
    config.save(path)
    assert _normalised_bytes(path) == first
    SLMConfig.load(path).save(path)
    assert _normalised_bytes(path) == first


def _cli_set(path: Path, section: str, **values) -> None:
    def change(data: dict) -> None:
        data[section] = {**(data.get(section) or {}), **values}

    update_config(path, change)


def test_cli_change_to_modelled_fields_survives_a_full_save(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    long_lived = SLMConfig.load(path)
    assert long_lived.injection.enabled is True
    assert long_lived.auto_invoke.enabled is True

    _cli_set(path, "injection", enabled=False)
    _cli_set(path, "auto_invoke", enabled=False)
    long_lived.save(path)

    on_disk = _read(path)
    assert on_disk["injection"]["enabled"] is False
    assert on_disk["auto_invoke"]["enabled"] is False


def test_cli_change_to_nested_modelled_leaf_survives(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    long_lived = SLMConfig.load(path)
    _cli_set(path, "auto_invoke", weights={"similarity": 0.7, "recency": 0.1,
                                           "frequency": 0.1, "trust": 0.1})
    long_lived.save(path)
    assert _read(path)["auto_invoke"]["weights"]["similarity"] == 0.7


def test_a_field_the_object_changed_still_wins(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    config = SLMConfig.load(path)
    _cli_set(path, "injection", enabled=False, core_block_max_facts=9)
    _cli_set(path, "auto_invoke", enabled=False)

    config.injection = replace(config.injection, core_block_max_facts=3)
    config.auto_invoke = replace(config.auto_invoke, enabled=True)  # unchanged
    config.evolution = replace(config.evolution, enabled=True)
    config.save(path)

    on_disk = _read(path)
    assert on_disk["injection"]["core_block_max_facts"] == 3
    assert on_disk["injection"]["enabled"] is False
    assert on_disk["auto_invoke"]["enabled"] is False
    assert on_disk["evolution"]["enabled"] is True


def test_object_change_wins_over_a_conflicting_cli_change(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    config = SLMConfig.load(path)
    _cli_set(path, "injection", enabled=False)
    config.injection = replace(config.injection, core_block_max_facts=1)
    _cli_set(path, "injection", core_block_max_facts=20)
    config.save(path)
    assert _read(path)["injection"]["core_block_max_facts"] == 1


def _seed_mode_b(base: Path) -> None:
    _write(base / "config.json", {**_REALISTIC, "mode": "b"})


def test_first_switch_then_cli_change_then_save_keeps_the_cli_change(
    tmp_path: Path,
) -> None:
    _seed_mode_b(tmp_path)
    assert not (tmp_path / "mode_c.json").exists()
    switched = SLMConfig.switch_mode("c", base_dir=tmp_path)
    path = tmp_path / "config.json"

    _cli_set(path, "injection", enabled=False)
    update_config(path, lambda data: data.update(x_forward_compat={"kept": "cli"}))
    switched.save(path)

    on_disk = _read(path)
    assert on_disk["mode"] == "c"
    assert on_disk["injection"]["enabled"] is False
    assert on_disk["x_forward_compat"] == {"kept": "cli"}


def test_fresh_config_saved_twice_keeps_a_cli_change(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    fresh = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    fresh.save(path)
    _cli_set(path, "injection", enabled=False)
    update_config(path, lambda data: data.update(x_cli_only=1))
    fresh.save(path)
    on_disk = _read(path)
    assert on_disk["injection"]["enabled"] is False
    assert on_disk["x_cli_only"] == 1


def test_saving_elsewhere_keeps_what_it_knows_about_the_loaded_file(
    tmp_path: Path,
) -> None:
    # The dashboard's mode switch: save the mode being left to mode_<x>.json,
    # change the mode, then save config.json.
    path = _write(tmp_path / "config.json", _REALISTIC)
    config = SLMConfig.load(path)
    config.save(tmp_path / "mode_b.json")
    _cli_set(path, "injection", enabled=False)
    config.mode = Mode.C
    config.save(path, mode_change=True)
    on_disk = _read(path)
    assert on_disk["mode"] == "c"
    assert on_disk["injection"]["enabled"] is False


def test_first_save_to_another_file_writes_the_loaded_values(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    other = _write(tmp_path / "mode_b.json", {
        **_REALISTIC,
        "injection": {"enabled": False},
        "x_forward_compat": {"kept": "stale"},
    })
    SLMConfig.load(path).save(other)
    on_disk = _read(other)
    assert on_disk["injection"]["enabled"] is True
    assert on_disk["x_forward_compat"] == {"kept": True}


def test_mode_protection_is_unchanged_on_the_same_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.json", _REALISTIC)
    config = SLMConfig.load(path)
    update_config(path, lambda data: data.update(mode="c"))
    config.save(path)
    assert _read(path)["mode"] == "c"
    config.mode = Mode.A
    config.save(path)
    assert _read(path)["mode"] == "c"
    config.save(path, mode_change=True)
    assert _read(path)["mode"] == "a"


@pytest.mark.skipif(not hasattr(os, "O_DIRECTORY"), reason="POSIX only")
def test_full_save_fsyncs_the_parent_directory(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.json"
    opened: dict[int, str] = {}
    synced: list[str] = []
    real_open, real_fsync = os.open, os.fsync

    def recording_open(target, flags, *args, **kwargs):
        fd = real_open(target, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            opened[fd] = str(Path(target).resolve())
        return fd

    def recording_fsync(fd):
        if fd in opened:
            synced.append(opened[fd])
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    SLMConfig.for_mode(Mode.A, base_dir=tmp_path).save(path)
    assert str(tmp_path.resolve()) in synced


def test_cli_change_survives_repeated_saves(tmp_path: Path) -> None:
    # The first save keeps the CLI's value; the second must not take it to be
    # this object's own write and put the object's value back.
    path = _write(tmp_path / "config.json", _REALISTIC)
    long_lived = SLMConfig.load(path)
    _cli_set(path, "injection", enabled=False)
    long_lived.save(path)
    long_lived.save(path)
    assert _read(path)["injection"]["enabled"] is False


def test_load_time_normalisation_is_still_written(tmp_path: Path) -> None:
    # load() reads mesh.retention_days 0 as 7; nobody changed the file since,
    # so the save writes the normalised value, as it always did.
    path = _write(tmp_path / "config.json", {**_REALISTIC, "mesh": {"retention_days": 0}})
    config = SLMConfig.load(path)
    assert config.mesh_retention_days == 7
    config.save(path)
    assert _read(path)["mesh"]["retention_days"] == 7
