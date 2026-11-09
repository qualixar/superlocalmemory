# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``slm models`` — installed + recommended local models, and the hosted
catalogue (#4.1.22). Reads Ollama through the one reviewed
``setup_wizard._ollama_installed_models`` call site; no HTTP of its own.
"""

from __future__ import annotations

import json
from argparse import Namespace

import httpx

import superlocalmemory.cli.models_cmd as models_cmd
import superlocalmemory.cli.setup_wizard as sw


class _Response:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _stub_installed(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Response(
        200, {"models": [{"name": n} for n in names]},
    ))
    monkeypatch.setattr(sw, "_get_ram_gb", lambda: 24.0)
    monkeypatch.setattr(models_cmd, "_get_ram_gb", lambda: 24.0)


class TestTextOutput:
    def test_the_recommended_installed_model_is_named_first(
        self, monkeypatch, capsys,
    ) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])

        models_cmd.cmd_models(Namespace(json=False))

        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        installed_block = out.split("Installed Ollama models", 1)[1]
        first_model_line = next(
            line for line in installed_block.splitlines() if line.strip().startswith(("gemma", "llama", "qwen"))
        )
        assert "gemma3:4b" in first_model_line
        assert "This computer: 24.0 GB" in out
        assert "Hosted (Mode C)" in out
        assert lines  # sanity: something was printed


class TestJsonOutput:
    def test_json_has_machine_installed_and_recommendations(
        self, monkeypatch, capsys,
    ) -> None:
        _stub_installed(monkeypatch, ["llama3.2:latest", "gemma3:4b"])

        models_cmd.cmd_models(Namespace(json=True))

        body = json.loads(capsys.readouterr().out)
        data = body["data"]
        assert data["machine"]["ram_gb"] == 24.0
        assert data["installed"] == ["llama3.2:latest", "gemma3:4b"]
        assert data["local_recommendations"][0]["model_id"] == "gemma3:4b"
        assert data["local_recommendations"][0]["installed"] is True
        # same catalogue the wizard and the dashboard read
        assert "local_llms" in data
        assert "hosted_llms" in data
