# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for ``slm compress`` CLI handlers."""

import json
from argparse import Namespace

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    from superlocalmemory.optimize.config.store import ConfigStore

    _store = ConfigStore(config_path=tmp_path / "optimize.json")
    monkeypatch.setattr(
        "superlocalmemory.cli.compress_cmd._get_store", lambda: _store
    )
    return _store


def test_compress_mode_safe(_isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_mode

    cmd_compress_mode(Namespace(json=False, mode_value="safe"))
    assert _isolate.get().compress_mode == "safe"


def test_compress_mode_aggressive(capsys, _isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_mode

    cmd_compress_mode(Namespace(json=False, mode_value="aggressive"))
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert _isolate.get().compress_mode == "aggressive"


def test_compress_mode_aggressive_json_has_warning(capsys, _isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_mode

    cmd_compress_mode(Namespace(json=True, mode_value="aggressive"))
    out = capsys.readouterr().out
    assert "WARNING" in out


def test_compress_code_on(_isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_code

    cmd_compress_code(Namespace(json=False, code_value="on"))
    assert _isolate.get().compress_code is True


def test_compress_code_off(_isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_code

    cmd_compress_code(Namespace(json=False, code_value="off"))
    assert _isolate.get().compress_code is False


def test_compress_ccr_on(_isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_ccr

    cmd_compress_ccr(Namespace(json=False, ccr_value="on"))
    assert _isolate.get().compress_ccr is True


def test_compress_ccr_off(_isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_ccr

    cmd_compress_ccr(Namespace(json=False, ccr_value="off"))
    assert _isolate.get().compress_ccr is False


def test_compress_prose_enables_compress(_isolate):
    import dataclasses
    from superlocalmemory.cli.compress_cmd import cmd_compress_prose

    # Pre-disable compress
    _isolate.save(dataclasses.replace(_isolate.get(), compress_enabled=False))
    assert _isolate.get().compress_enabled is False

    cmd_compress_prose(Namespace(json=False, prose_value="on"))
    cfg = _isolate.get()
    assert cfg.compress_prose is True
    assert cfg.compress_enabled is True  # auto-enabled


def test_compress_status_json(capsys, _isolate):
    from superlocalmemory.cli.compress_cmd import cmd_compress_status

    cmd_compress_status(Namespace(json=True))
    data = json.loads(capsys.readouterr().out)
    assert "compress_enabled" in data
    assert "compress_mode" in data
