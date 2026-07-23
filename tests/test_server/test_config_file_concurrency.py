"""Config control-plane writers must not lose concurrent section updates."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path


def _write_key(path: str, key: str, value: str, delay: float) -> None:
    from superlocalmemory.server.config_file import update_config

    def mutate(data: dict) -> None:
        time.sleep(delay)
        data[key] = value

    update_config(Path(path), mutate)


def test_interprocess_config_updates_preserve_unrelated_sections(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"mode": "b"}), encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    first = context.Process(
        target=_write_key,
        args=(str(config_path), "mesh_enabled", "false", 0.15),
    )
    second = context.Process(
        target=_write_key,
        args=(str(config_path), "evolution", "enabled", 0.0),
    )
    first.start()
    time.sleep(0.03)
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "mode": "b",
        "mesh_enabled": "false",
        "evolution": "enabled",
    }
