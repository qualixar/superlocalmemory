# DASH-V4 (3.7.9): active-config load must honor the current_mode file
# (single source of truth) when config.json has drifted from it.
import json

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def _write(p, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_active_load_reconciles_to_current_mode(tmp_path):
    """config.json drifted to Mode A but current_mode='b' → load Mode B config."""
    _write(tmp_path / "config.json", {"mode": "a", "llm": {"provider": "", "model": ""}})
    (tmp_path / "current_mode").write_text("b", encoding="utf-8")
    _write(tmp_path / "mode_b.json",
           {"mode": "b", "llm": {"provider": "ollama", "model": "llama3.2"}})

    cfg = SLMConfig.load(tmp_path / "config.json")

    assert cfg.mode == Mode.B, "must honor current_mode file, not stale config.json"
    assert cfg.llm.provider == "ollama", "must load the mode's real settings, not just the letter"


def test_no_divergence_loads_normally(tmp_path):
    """When config.json and current_mode agree, load config.json as-is."""
    _write(tmp_path / "config.json",
           {"mode": "b", "llm": {"provider": "ollama", "model": "llama3.2"}})
    (tmp_path / "current_mode").write_text("b", encoding="utf-8")

    cfg = SLMConfig.load(tmp_path / "config.json")
    assert cfg.mode == Mode.B


def test_per_mode_file_load_is_not_reconciled(tmp_path):
    """Explicit per-mode file loads (used by switch_mode) must NOT reconcile."""
    _write(tmp_path / "mode_a.json", {"mode": "a"})
    (tmp_path / "current_mode").write_text("b", encoding="utf-8")

    cfg = SLMConfig.load(tmp_path / "mode_a.json")
    assert cfg.mode == Mode.A, "loading mode_a.json explicitly must stay Mode A"


def test_missing_mode_file_falls_open(tmp_path):
    """Divergence but no mode_b.json → fail-open to config.json's mode (no crash)."""
    _write(tmp_path / "config.json", {"mode": "a"})
    (tmp_path / "current_mode").write_text("b", encoding="utf-8")

    cfg = SLMConfig.load(tmp_path / "config.json")
    assert cfg.mode == Mode.A
