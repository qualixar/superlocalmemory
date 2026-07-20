"""LLD-06 §8 — slm wrap tests + adapter tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from superlocalmemory.optimize.adapters import withSLM
from superlocalmemory.optimize.adapters.wrap import (
    list_agents,
    wrap_agent,
)
from superlocalmemory.optimize.config import (
    _reset_config_store,
    _set_config_store,
)
from superlocalmemory.optimize.config.store import ConfigStore
from superlocalmemory.optimize.proxy.lifecycle import ensure_proxy_running, proxy_port


# ---- list_agents ----

def test_list_agents_includes_claude() -> None:
    keys = list_agents()
    assert "claude" in keys
    assert "claude-settings" in keys
    assert "codex" in keys
    assert "aider" in keys
    assert "cline" in keys
    assert "generic" in keys


# ---- wrap_agent: print-only path ----

def test_wrap_agent_unknown_returns_1() -> None:
    assert wrap_agent("no-such-agent", []) == 1


def test_wrap_agent_proxy_disabled_returns_1(tmp_path: Path) -> None:
    """proxy_enabled=False → fail with clear message."""
    cfg_path = tmp_path / "optimize.json"
    cfg_path.write_text(json.dumps({"proxy_enabled": False, "enabled": True}))
    store = ConfigStore(config_path=cfg_path, poll_interval=3600.0)
    _set_config_store(store)
    try:
        # Override proxy check: in this setup proxy is disabled
        # The check `ensure_proxy_running()` returns cfg.proxy_enabled → False
        rc = wrap_agent("claude", [], dry_run=True)
        assert rc == 1
    finally:
        _reset_config_store()


def test_wrap_agent_print_only_prints_instructions(capsys) -> None:
    """print-only agents print help text and return 0."""
    cfg_path = Path("/tmp/__not_used__.json")
    # ensure_proxy_running checks module-level store; we need it to think proxy is on
    from superlocalmemory.optimize.config import _set_config_store, get_optimize_config
    from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG
    # Use a config with proxy_enabled=True
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    from superlocalmemory.optimize.config.store import ConfigStore
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        path = f.name
    store = ConfigStore(config_path=Path(path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        rc = wrap_agent("antigravity", [], dry_run=True)
        out = capsys.readouterr().out
        assert rc == 0
        assert "antigravity" in out.lower() or "redirect" in out.lower()
    finally:
        _reset_config_store()
        try:
            os.unlink(path)
        except OSError:
            pass


# ---- proxy_port ----

def test_proxy_port_is_8765() -> None:
    assert proxy_port() == 8765


def test_ensure_proxy_running_returns_bool() -> None:
    assert isinstance(ensure_proxy_running(), bool)


# ---- withSLM ----

def test_withSLM_returns_unchanged_when_disabled(monkeypatch, tmp_path: Path) -> None:
    """enabled=False → pass-through (DEFAULT STATE)."""
    # Force a config that returns enabled=False
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.config.schema import OptimizeConfig

    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": False}))
    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)
    try:
        sentinel = object()
        assert withSLM(sentinel) is sentinel
    finally:
        _reset_config_store()


def test_withSLM_returns_unchanged_for_unknown_type(monkeypatch, tmp_path: Path) -> None:
    """Unrecognized client type → pass-through, log WARNING."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore

    p = tmp_path / "oc.json"
    p.write_text(json.dumps({"enabled": True}))
    store = ConfigStore(config_path=p, poll_interval=3600.0)
    _set_config_store(store)
    try:
        sentinel = object()
        assert withSLM(sentinel) is sentinel
    finally:
        _reset_config_store()


# ---- wrap_agent: settings-file path ----

def test_wrap_agent_settings_file_dry_run(tmp_path: Path) -> None:
    """settings-file mechanism dry_run prints but does not write."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    from superlocalmemory.optimize.config.store import ConfigStore
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        rc = wrap_agent("claude-settings", [], dry_run=True)
        assert rc == 0
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_settings_file_writes(tmp_path: Path) -> None:
    """settings-file mechanism writes env vars to settings file."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    import tempfile

    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    settings_path = settings_dir / "settings.json"
    settings_path.write_text(json.dumps({"existing": "value"}))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        from superlocalmemory.optimize.adapters import _agent_registry
        from superlocalmemory.optimize.adapters.wrap import _atomic_write_text
        import copy
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        # Override the settings_path to use tmp_path
        _agent_registry.AGENT_REGISTRY["claude-settings"]["settings_path"] = str(settings_path)
        try:
            rc = wrap_agent("claude-settings", [])
            assert rc == 0
            written = json.loads(settings_path.read_text())
            assert "env" in written
            assert "existing" in written
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_settings_file_no_env_vars(tmp_path: Path) -> None:
    """settings-file with no env_vars → return 1."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["claude-settings"] = {
            "binary": None,
            "mechanism": "settings-file",
            "settings_path": str(tmp_path / "settings.json"),
            "env_vars": {},  # empty
            "protocol": "anthropic",
        }
        try:
            rc = wrap_agent("claude-settings", [], dry_run=True)
            assert rc == 1  # no env vars to set
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ---- wrap_agent: config-file path ----

def test_wrap_agent_config_file_dry_run(tmp_path: Path) -> None:
    """config-file mechanism dry_run prints but does not write."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        rc = wrap_agent("cline", [], dry_run=True)
        assert rc == 0
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_config_file_writes(tmp_path: Path) -> None:
    """config-file mechanism writes config value to settings.json."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    settings_dir = tmp_path / "Code" / "User"
    settings_dir.mkdir(parents=True)
    settings_path = settings_dir / "settings.json"
    settings_path.write_text(json.dumps({}))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["cline"] = {
            "binary": None,
            "mechanism": "config-file",
            "config_path": str(settings_path),
            "config_key": "cline.testKey",
            "config_value": "http://127.0.0.1:8765",
            "protocol": "openai",
        }
        try:
            rc = wrap_agent("cline", [])
            assert rc == 0
            written = json.loads(settings_path.read_text())
            assert "cline.testKey" in written
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_config_file_no_config_path(tmp_path: Path) -> None:
    """config-file with no config_path → return 1."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["cline"] = {
            "binary": None,
            "mechanism": "config-file",
            "config_path": "",  # empty
            "config_key": "test",
            "config_value": "test",
            "protocol": "openai",
        }
        try:
            rc = wrap_agent("cline", [])
            assert rc == 1
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_config_file_vscode_template_not_found(tmp_path: Path, monkeypatch) -> None:
    """config-file with {vscode_user_dir} template → vscode dir not found."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile
    from superlocalmemory.optimize.adapters import wrap as _wrap_mod

    # Force _vscode_user_dir to return None
    monkeypatch.setattr(_wrap_mod, "_vscode_user_dir", lambda: None)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["cline"] = {
            "binary": None,
            "mechanism": "config-file",
            "config_path": "{vscode_user_dir}/settings.json",
            "config_key": "test",
            "config_value": "test",
            "protocol": "openai",
        }
        try:
            rc = wrap_agent("cline", [])
            assert rc == 1  # vscode dir not found
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ---- wrap_agent: env mechanism ----

def test_wrap_agent_env_dry_run(tmp_path: Path) -> None:
    """env mechanism dry_run prints env vars without launching."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        rc = wrap_agent("claude", [], dry_run=True)
        assert rc == 0
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_env_binary_not_found(tmp_path: Path) -> None:
    """env mechanism with missing binary → return 1."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["test-nonexist"] = {
            "binary": "definitely-not-a-real-binary-xyz",
            "mechanism": "env",
            "env_vars": {"TEST": "1"},
            "protocol": "anthropic",
        }
        try:
            rc = wrap_agent("test-nonexist", [])
            assert rc == 1  # binary not found
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_env_no_binary(tmp_path: Path) -> None:
    """env mechanism with no binary specified → return 1."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["test-nobin"] = {
            "binary": None,
            "mechanism": "env",
            "env_vars": {"TEST": "1"},
            "protocol": "anthropic",
        }
        try:
            rc = wrap_agent("test-nobin", [])
            assert rc == 1  # no binary specified
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


def test_wrap_agent_unknown_mechanism(tmp_path: Path) -> None:
    """Unknown mechanism → return 1."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["test-unk"] = {
            "binary": None,
            "mechanism": "super-weird-mechanism",
            "protocol": "anthropic",
        }
        try:
            rc = wrap_agent("test-unk", [])
            assert rc == 1
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ---- _vscode_user_dir ----

def test_vscode_user_dir_darwin() -> None:
    """_vscode_user_dir returns Code/User path on darwin when VS Code is installed."""
    from superlocalmemory.optimize.adapters.wrap import _vscode_user_dir
    import sys
    from pathlib import Path
    if sys.platform == "darwin":
        vscode_parent = Path.home() / "Library" / "Application Support" / "Code"
        if not vscode_parent.exists():
            pytest.skip("VS Code not installed on this machine")
        p = _vscode_user_dir()
        assert p is not None
        assert "Code" in str(p)


# ---- wrap_agent: config-file JSONDecodeError recovery ----

def test_wrap_agent_config_file_json_decode_error(tmp_path: Path) -> None:
    """config-file path where settings.json has corrupt JSON → recover and write."""
    from superlocalmemory.optimize.config import _set_config_store, _reset_config_store
    from superlocalmemory.optimize.config.store import ConfigStore
    from superlocalmemory.optimize.adapters import _agent_registry
    import copy, tempfile

    settings_dir = tmp_path / "Code" / "User"
    settings_dir.mkdir(parents=True)
    settings_path = settings_dir / "settings.json"
    settings_path.write_text("not valid json {{{")  # corrupt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"proxy_enabled": True, "enabled": True}, f)
        cfg_path = f.name
    store = ConfigStore(config_path=Path(cfg_path), poll_interval=3600.0)
    _set_config_store(store)
    try:
        orig_registry = copy.deepcopy(_agent_registry.AGENT_REGISTRY)
        _agent_registry.AGENT_REGISTRY["cline"] = {
            "binary": None,
            "mechanism": "config-file",
            "config_path": str(settings_path),
            "config_key": "cline.testKey",
            "config_value": "http://127.0.0.1:8765",
            "protocol": "openai",
        }
        try:
            rc = wrap_agent("cline", [])
            assert rc == 0
            written = json.loads(settings_path.read_text())
            assert "cline.testKey" in written
        finally:
            _agent_registry.AGENT_REGISTRY.clear()
            _agent_registry.AGENT_REGISTRY.update(orig_registry)
    finally:
        _reset_config_store()
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ---- _atomic_write_text ----

def test_atomic_write_text(tmp_path: Path) -> None:
    """_atomic_write_text writes file atomically."""
    from superlocalmemory.optimize.adapters.wrap import _atomic_write_text
    fpath = tmp_path / "test.json"
    _atomic_write_text(fpath, json.dumps({"key": "val"}))
    assert fpath.exists()
    assert json.loads(fpath.read_text()) == {"key": "val"}
    # Verify no .tmp file left behind
    assert not list(tmp_path.glob("*.tmp"))
