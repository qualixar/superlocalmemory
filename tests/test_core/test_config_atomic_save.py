# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Stage-8 data-loss regression: config.json save is atomic + load recovers on corrupt.

CRITICAL-1: SLMConfig.save() used a bare write_text (truncate-then-write). A crash
mid-write left a corrupt config.json, and load() did an unguarded json.loads ->
EVERY subsequent `slm` invocation crashed with no recovery. Fix: atomic .tmp +
os.replace on save; load() degrades to Mode A (with a warning) on a corrupt file.
"""
import json
import tempfile
from pathlib import Path

from superlocalmemory.core.config import Mode, SLMConfig


def test_load_recovers_to_mode_a_on_corrupt_config():
    d = Path(tempfile.mkdtemp())
    cfg = d / "config.json"
    cfg.write_text('{\n  "mode')  # truncated mid-write (the crash scenario)
    loaded = SLMConfig.load(config_path=cfg)  # must NOT raise
    assert loaded.mode == Mode.A


def test_load_recovers_on_empty_config():
    d = Path(tempfile.mkdtemp())
    cfg = d / "config.json"
    cfg.write_text("")  # zero-byte file from a crash after truncate
    assert SLMConfig.load(config_path=cfg).mode == Mode.A


def test_save_is_atomic_valid_json_no_leftover_tmp():
    d = Path(tempfile.mkdtemp())
    cfg = d / "config.json"
    SLMConfig.for_mode(Mode.A).save(config_path=cfg)
    json.loads(cfg.read_text())  # valid JSON, no partial write
    assert not (d / "config.json.tmp").exists()  # tmp cleaned up by os.replace
