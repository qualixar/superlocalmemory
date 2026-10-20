"""CLI contracts for explicit Scale Engine adoption."""
from __future__ import annotations

from argparse import Namespace

from superlocalmemory.cli.scale_engine_cmd import cmd_db_scale
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.scale_engine import ScaleEngineManager
from superlocalmemory.storage.sqlite_vectors import CanonicalVectorError


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


def test_vector_contract_failure_is_reported_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(SLMConfig, "load", staticmethod(SLMConfig))
    monkeypatch.setattr(
        ScaleEngineManager,
        "prepare",
        lambda self: (_ for _ in ()).throw(
            CanonicalVectorError("canonical vector mapping is incomplete")
        ),
    )

    result = cmd_db_scale(
        Namespace(scale_action="prepare", stage_id=None, backup_id=None)
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "Scale Engine: canonical vector mapping is incomplete" in output
    assert "Traceback" not in output


# -- #296: promote/rollback swap the Cozo/LanceDB stores a running daemon holds open --------------

import pytest  # noqa: E402

from superlocalmemory.cli import daemon as _daemon  # noqa: E402


@pytest.mark.parametrize("action,arg", [("promote", "stage_id"), ("rollback", "backup_id")])
def test_promote_and_rollback_refuse_while_the_daemon_runs(monkeypatch, capsys, action, arg):
    monkeypatch.setattr(SLMConfig, "load", staticmethod(SLMConfig))
    monkeypatch.setattr(_daemon, "owned_daemon_process_alive", lambda: True)
    called: list[str] = []
    monkeypatch.setattr(ScaleEngineManager, action, lambda self, _id: called.append(_id) or {})

    args = {"scale_action": action, "stage_id": None, "backup_id": None, arg: "x-1"}
    assert cmd_db_scale(Namespace(**args)) == 1
    assert called == []
    assert "daemon is running" in capsys.readouterr().out


@pytest.mark.parametrize("action,arg", [("promote", "stage_id"), ("rollback", "backup_id")])
def test_promote_and_rollback_run_with_the_daemon_stopped(monkeypatch, capsys, action, arg):
    monkeypatch.setattr(SLMConfig, "load", staticmethod(SLMConfig))
    monkeypatch.setattr(_daemon, "owned_daemon_process_alive", lambda: False)
    called: list[str] = []
    monkeypatch.setattr(ScaleEngineManager, action, lambda self, _id: called.append(_id) or {"ok": True})

    args = {"scale_action": action, "stage_id": None, "backup_id": None, arg: "x-1"}
    assert cmd_db_scale(Namespace(**args)) == 0
    assert called == ["x-1"]
