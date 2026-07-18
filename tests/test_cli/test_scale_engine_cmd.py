"""CLI contracts for explicit Scale Engine adoption."""
from __future__ import annotations

from argparse import Namespace

from superlocalmemory.cli.scale_engine_cmd import cmd_db_scale
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.scale_engine import ScaleEngineManager


def test_adopt_uses_default_projection_without_changing_selected_profile(monkeypatch, capsys):
    config = SLMConfig()
    config.active_profile = "client-acme"
    captured: dict[str, object] = {}

    monkeypatch.setattr(SLMConfig, "load", staticmethod(lambda: config))

    def fake_init(self, received_config, *, backend_factory=None, profile_id=None):
        captured["config"] = received_config
        captured["profile_id"] = profile_id

    monkeypatch.setattr(ScaleEngineManager, "__init__", fake_init)
    monkeypatch.setattr(
        ScaleEngineManager,
        "adopt_legacy_projection",
        lambda self: {"state": "promoted", "stage_id": "stage-1"},
    )

    result = cmd_db_scale(Namespace(scale_action="adopt", stage_id=None, backup_id=None))

    assert result == 0
    assert captured["config"] is config
    assert captured["profile_id"] == "default"
    assert config.active_profile == "client-acme"
    assert '"restart_required": true' in capsys.readouterr().out
