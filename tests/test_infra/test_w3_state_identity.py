"""W3 contracts for cache, optimize, auth, and signing state identity."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


def _signer_subprocess(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    """Run signer code in a fresh interpreter with an isolated namespace."""
    home = tmp_path / "home"
    data_root = tmp_path / "data"
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["SLM_DATA_DIR"] = str(data_root)
    env.pop("SL_MEMORY_PATH", None)
    env.pop("SLM_HOME", None)
    env.pop("SLM_SIGNER_KEY", None)
    env["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_signer_import_is_read_only(tmp_path: Path) -> None:
    result = _signer_subprocess(
        tmp_path,
        "import superlocalmemory.attribution.signer",
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home" / ".superlocalmemory").exists()
    assert not (tmp_path / "data").exists()


def test_default_signer_creates_key_lazily_in_canonical_root(tmp_path: Path) -> None:
    result = _signer_subprocess(
        tmp_path,
        "from superlocalmemory.attribution.signer import QualixarSigner; "
        "s = QualixarSigner(); assert s.verify('x', s.sign('x'))",
    )
    assert result.returncode == 0, result.stderr
    key_file = tmp_path / "data" / ".signer_key"
    assert key_file.exists()
    assert len(key_file.read_text(encoding="utf-8").strip()) == 64
    if os.name != "nt":
        assert key_file.stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "home" / ".superlocalmemory").exists()


def test_explicit_signer_key_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    result = _signer_subprocess(
        tmp_path,
        "from superlocalmemory.attribution.signer import QualixarSigner; "
        "s = QualixarSigner('explicit'); assert s.verify('x', s.sign('x'))",
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home" / ".superlocalmemory").exists()
    assert not (tmp_path / "data").exists()


def test_environment_signer_key_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    result = _signer_subprocess(
        tmp_path,
        "import os; os.environ['SLM_SIGNER_KEY'] = 'environment'; "
        "from superlocalmemory.attribution.signer import QualixarSigner; "
        "s = QualixarSigner(); assert s.verify('x', s.sign('x'))",
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home" / ".superlocalmemory").exists()
    assert not (tmp_path / "data").exists()


def test_context_cache_defaults_and_explicit_home_are_install_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.core.context_cache import ContextCache

    canonical = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    cache = ContextCache()
    cache.close()
    assert (canonical / "active_brain_cache.db").exists()
    assert (canonical / ".install_token").exists()

    explicit = tmp_path / "explicit"
    explicit_db = explicit / "custom-cache.db"
    cache = ContextCache(db_path=explicit_db, home_dir=explicit)
    cache.close()
    assert explicit_db.exists()
    assert (explicit / ".install_token").exists()


def test_optimize_config_default_is_dynamic_and_explicit_path_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.optimize.config.defaults import DEFAULT_OPTIMIZE_CONFIG
    from superlocalmemory.optimize.config.store import ConfigStore

    canonical = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    store = ConfigStore()
    store.save(DEFAULT_OPTIMIZE_CONFIG)
    assert (canonical / "optimize.json").exists()

    explicit = tmp_path / "explicit" / "settings.json"
    store = ConfigStore(config_path=explicit)
    store.save(DEFAULT_OPTIMIZE_CONFIG)
    assert explicit.exists()


def test_auth_default_is_dynamic_and_explicit_path_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.infra.auth_middleware import check_api_key

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "api_key").write_text("canonical-secret", encoding="utf-8")
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    assert check_api_key(
        {"x-slm-api-key": "canonical-secret"}, is_write=True,
    )
    assert not check_api_key(
        {"x-slm-api-key": "wrong"}, is_write=True,
    )

    explicit = tmp_path / "explicit-key"
    explicit.write_text("explicit-secret", encoding="utf-8")
    assert check_api_key(
        {"x-slm-api-key": "explicit-secret"},
        is_write=True,
        key_file=explicit,
    )


def test_cache_db_defaults_to_canonical_root_and_explicit_db_path_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.optimize.storage.db import CacheDB

    canonical = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    db = CacheDB()
    db.close()
    assert Path(db.db_path) == canonical / "llmcache.db"
    assert (canonical / "opt-key.bin").exists()

    explicit = tmp_path / "explicit" / "cache.db"
    db = CacheDB(explicit)
    db.close()
    assert Path(db.db_path) == explicit
    assert explicit.exists()


def test_linux_machine_id_precedes_canonical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.optimize.storage.db import CacheDB

    canonical = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    original_read_text = Path.read_text

    def fake_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == Path("/etc/machine-id"):
            return "system-machine-id\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    instance = object.__new__(CacheDB)
    assert instance._get_machine_id() == "system-machine-id"
    assert not (canonical / ".llmcache_key").exists()


def test_machine_id_fallback_uses_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from superlocalmemory.optimize.storage.db import CacheDB

    canonical = tmp_path / "canonical"
    monkeypatch.setenv("SLM_DATA_DIR", str(canonical))
    monkeypatch.setattr(platform, "system", lambda: "Other")
    instance = object.__new__(CacheDB)
    machine_id = instance._get_machine_id()
    assert machine_id
    assert (canonical / ".llmcache_key").read_text(encoding="utf-8") == machine_id
def test_cloud_backup_defaults_follow_canonical_root_after_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.infra import cloud_backup

    selected = (tmp_path / "cloud-root").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    assert cloud_backup._memory_dir() == selected
    assert cloud_backup._default_db_path() == selected / "memory.db"
    assert cloud_backup._get_credential_store() == selected / ".credentials.json"


def test_optimize_capture_path_follows_canonical_root_after_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from superlocalmemory.optimize.proxy import capture

    selected = (tmp_path / "capture-root").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    assert capture._capture_path() == selected / "optimize_capture.jsonl"
