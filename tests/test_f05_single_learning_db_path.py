# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""F-05 invariant: daemon migration and engine share one learning.db path.

With a custom root via SLMConfig and no env aliases, only one learning.db
namespace may be touched by daemon startup path resolution + engine init path.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.infra.data_root import DATA_ROOT_ALIASES, canonical_data_root
from superlocalmemory.storage.models import Mode


def _clear_data_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in DATA_ROOT_ALIASES:
        monkeypatch.delenv(name, raising=False)


def _migration_block_source() -> str:
    """Return the lifespan source segment that resolves learning.db for apply_all."""
    import superlocalmemory.server.unified_daemon as ud

    source = inspect.getsource(ud.lifespan)
    # First apply_all site is the pre-engine migration block (F-05 target).
    marker = "from superlocalmemory.storage.migration_runner import apply_all"
    assert marker in source, "lifespan no longer imports apply_all"
    start = source.index(marker)
    # Bound the block to the following major section or next bare except.
    tail = source[start : start + 1200]
    return tail


def test_lifespan_migration_block_does_not_call_bare_canonical_data_root() -> None:
    """The apply_all home must not be bare canonical_data_root() with no args."""
    block = _migration_block_source()
    # Defect shape: `_home = canonical_data_root()` then `_home / "learning.db"`.
    bare = re.search(r"canonical_data_root\s*\(\s*\)", block)
    assert bare is None, (
        "migration block still calls bare canonical_data_root() — "
        "configured_base_dir / config.base_dir is skipped"
    )
    assert "_learning_db_for_config" in block, (
        "migration block must resolve learning.db via _learning_db_for_config"
    )
    assert "apply_all" in block


def test_lifespan_migration_resolves_learning_db_from_config_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With custom SLMConfig base_dir and no env, daemon path == engine path."""
    _clear_data_root_env(monkeypatch)
    custom = tmp_path / "custom-root"
    custom.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    cfg = SLMConfig.for_mode(Mode.A, base_dir=custom)
    engine_learning = (cfg.base_dir / "learning.db").resolve()

    # Product helper — single resolution point for daemon migration.
    import superlocalmemory.server.unified_daemon as ud

    assert hasattr(ud, "_learning_db_for_config"), (
        "unified_daemon must expose _learning_db_for_config so migration and "
        "engine cannot diverge"
    )
    daemon_learning = Path(ud._learning_db_for_config(cfg)).resolve()
    assert daemon_learning == engine_learning

    # Guard: bare root without configured base would still be wrong here.
    bare = (canonical_data_root() / "learning.db").resolve()
    assert bare != engine_learning


def test_exactly_one_learning_db_path_for_custom_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_data_root_env(monkeypatch)
    custom = tmp_path / "ns"
    custom.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    cfg = SLMConfig.for_mode(Mode.A, base_dir=custom)
    import superlocalmemory.server.unified_daemon as ud

    paths = {
        Path(ud._learning_db_for_config(cfg)).resolve(),
        (cfg.base_dir / "learning.db").resolve(),
        (
            canonical_data_root(configured_base_dir=cfg.base_dir) / "learning.db"
        ).resolve(),
    }
    assert len(paths) == 1, f"multiple learning.db paths: {paths}"
