# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Tests for ``slm proxy`` CLI handler."""

import json
from argparse import Namespace
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    from superlocalmemory.optimize.config.store import ConfigStore

    _store = ConfigStore(config_path=tmp_path / "optimize.json")
    monkeypatch.setattr(
        "superlocalmemory.cli.proxy_cmd._get_store", lambda: _store
    )
    # Mock ensure_running to avoid actually starting a proxy
    monkeypatch.setattr(
        "superlocalmemory.cli.proxy_cmd._ensure_running", lambda port: True
    )
    return _store


def test_proxy_invalid_port_low():
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    with pytest.raises(SystemExit) as exc:
        cmd_proxy(Namespace(json=False, port=80, provider="anthropic",
                            no_compress=False, semantic=False))
    assert exc.value.code == 1


def test_proxy_invalid_port_high():
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    with pytest.raises(SystemExit) as exc:
        cmd_proxy(Namespace(json=False, port=99999, provider="anthropic",
                            no_compress=False, semantic=False))
    assert exc.value.code == 1


def test_proxy_writes_base_url(_isolate):
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    cmd_proxy(Namespace(json=False, port=8765, provider="anthropic",
                        no_compress=False, semantic=False))
    cfg = _isolate.get()
    providers = cfg.providers if isinstance(cfg.providers, dict) else {}
    anthropic = providers.get("anthropic")
    assert anthropic is not None
    assert "8765" in anthropic.base_url


def test_proxy_json_output(capsys, _isolate):
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    cmd_proxy(Namespace(json=True, port=8765, provider="anthropic",
                        no_compress=False, semantic=False))
    data = json.loads(capsys.readouterr().out)
    assert data["port"] == 8765
    assert "8765" in data["anthropic_url"]


def test_proxy_no_compress_flag(_isolate):
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    cmd_proxy(Namespace(json=False, port=8765, provider="anthropic",
                        no_compress=True, semantic=False))
    assert _isolate.get().compress_enabled is False


def test_proxy_semantic_flag(_isolate):
    from superlocalmemory.cli.proxy_cmd import cmd_proxy

    cmd_proxy(Namespace(json=False, port=8765, provider="anthropic",
                        no_compress=False, semantic=True))
    assert _isolate.get().semantic_enabled is True
