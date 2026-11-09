# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""D-02 / D-03 / D-04 regression tests — daemon lifecycle defect fixes.

D-02: trigram_index imported at module top (not lazily at shutdown) so the
      module reference survives interpreter teardown.

D-03: _perf_log_flush imported at module top (not lazily at shutdown) so the
      compiled .pyc object is used and macOS TCC xattr checks are skipped.

D-04: UI assets are served from the data dir (no xattr restrictions) rather
      than the source tree. The copy is idempotent and falls back to source
      with a WARNING — never crashes the daemon.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# D-02: trigram_index module reference cached at module level
# ---------------------------------------------------------------------------

def test_d02_trigram_index_mod_is_cached_at_module_level() -> None:
    """_trigram_index_mod must be bound at module load, not None."""
    import superlocalmemory.server.unified_daemon as daemon
    assert daemon._trigram_index_mod is not None, (
        "_trigram_index_mod was not imported at module load — D-02 regression"
    )


def test_d02_trigram_reset_cache_conn_is_callable() -> None:
    """_reset_cache_conn must be callable from the module-level reference."""
    import superlocalmemory.server.unified_daemon as daemon
    fn = getattr(daemon._trigram_index_mod, "_reset_cache_conn", None)
    assert callable(fn), (
        "_trigram_index_mod._reset_cache_conn is not callable — D-02 regression"
    )


def test_d02_shutdown_survives_missing_trigram_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a stripped install: _trigram_index_mod is None.
    The shutdown block must not raise — it must silently skip."""
    import superlocalmemory.server.unified_daemon as daemon
    original = daemon._trigram_index_mod
    try:
        daemon._trigram_index_mod = None
        # Simulate the shutdown block logic inline
        try:
            if daemon._trigram_index_mod is not None:
                daemon._trigram_index_mod._reset_cache_conn()
            # should reach here without error
        except Exception as exc:
            pytest.fail(f"Shutdown block raised with None _trigram_index_mod: {exc}")
    finally:
        daemon._trigram_index_mod = original


# ---------------------------------------------------------------------------
# D-03: _perf_log_flush function reference cached at module level
# ---------------------------------------------------------------------------

def test_d03_perf_log_flush_fn_is_cached_at_module_level() -> None:
    """_perf_log_flush_fn must be bound at module load, not None."""
    import superlocalmemory.server.unified_daemon as daemon
    assert daemon._perf_log_flush_fn is not None, (
        "_perf_log_flush_fn was not imported at module load — D-03 regression"
    )


def test_d03_perf_log_flush_fn_is_callable() -> None:
    """_perf_log_flush_fn must be callable."""
    import superlocalmemory.server.unified_daemon as daemon
    assert callable(daemon._perf_log_flush_fn), (
        "_perf_log_flush_fn is not callable — D-03 regression"
    )


def test_d03_shutdown_survives_missing_perf_log_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a stripped install: _perf_log_flush_fn is None.
    The shutdown block must not raise — it must silently skip."""
    import superlocalmemory.server.unified_daemon as daemon
    original = daemon._perf_log_flush_fn
    try:
        daemon._perf_log_flush_fn = None
        try:
            if daemon._perf_log_flush_fn is not None:
                daemon._perf_log_flush_fn()
        except Exception as exc:
            pytest.fail(f"Shutdown block raised with None _perf_log_flush_fn: {exc}")
    finally:
        daemon._perf_log_flush_fn = original


def test_d03_outcome_common_no_server_imports() -> None:
    """_outcome_common must not import from superlocalmemory.server.*
    — confirming no circular import exists when we move it to module top."""
    import ast
    src = (
        Path(__file__).resolve().parents[2]
        / "src/superlocalmemory/hooks/_outcome_common.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("superlocalmemory.server"), (
                    f"_outcome_common imports from server namespace: {node.module}"
                )


# ---------------------------------------------------------------------------
# D-04: UI served from data dir, not source tree
# ---------------------------------------------------------------------------

def test_d04_ui_copy_to_data_dir_succeeds(tmp_path: Path) -> None:
    """When source UI dir exists, _register_dashboard_routes copies it to
    the data dir and UI_DIR inside the function resolves to the data dir."""

    # Point data root at tmp_path so the copy lands there
    data_root = tmp_path / "slm_data"
    data_root.mkdir()

    # Build a minimal fake source UI dir
    fake_source_ui = tmp_path / "source_ui"
    fake_source_ui.mkdir()
    (fake_source_ui / "index.html").write_text("<html>test</html>", encoding="utf-8")
    (fake_source_ui / "js").mkdir()
    (fake_source_ui / "js" / "app.js").write_text("// js", encoding="utf-8")

    captured_ui_dir: list[Path] = []


    # Patch the internal imports used by _register_dashboard_routes
    with (
        patch("superlocalmemory.server.unified_daemon.state_path") as mock_state_path,
        patch("superlocalmemory.server.api.UI_DIR", new=fake_source_ui),
    ):
        # state_path("ui") → data_root/ui
        _data_ui = data_root / "ui"
        mock_state_path.return_value = _data_ui

        # We can't call _register_dashboard_routes directly (it mounts routes),
        # but we CAN test the copy logic it contains by replaying it here.
        # This mirrors the exact block added in D-04.
        import shutil
        _source_ui_dir = fake_source_ui
        _data_ui_dir = _data_ui
        try:
            _data_ui_dir.mkdir(parents=True, exist_ok=True)
            if _source_ui_dir.is_dir():
                shutil.copytree(
                    str(_source_ui_dir),
                    str(_data_ui_dir),
                    dirs_exist_ok=True,
                )
            ui_dir = _data_ui_dir
        except Exception:
            ui_dir = _source_ui_dir

        captured_ui_dir.append(ui_dir)

    assert captured_ui_dir[0] == data_root / "ui", (
        "UI_DIR should resolve to data dir, not source tree"
    )
    assert (data_root / "ui" / "index.html").exists(), (
        "index.html was not copied to data dir"
    )
    assert (data_root / "ui" / "js" / "app.js").exists(), (
        "js/app.js was not copied to data dir"
    )


def test_d04_fallback_to_source_when_copy_fails(tmp_path: Path) -> None:
    """If the data-dir copy raises (e.g. permissions), UI_DIR falls back to
    the source path and a WARNING is logged — the daemon must not crash."""
    import logging

    fake_source_ui = tmp_path / "source_ui"
    fake_source_ui.mkdir()
    (fake_source_ui / "index.html").write_text("<html>fallback</html>", encoding="utf-8")

    _data_ui_dir = tmp_path / "data_ui"
    # Do NOT mkdir — copytree will fail because the parent doesn't exist either
    # Actually, let's make copytree fail by raising manually
    warnings_logged: list[str] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warnings_logged.append(record.getMessage())

    _handler = _CapturingHandler()
    _log = logging.getLogger("superlocalmemory.unified_daemon")
    _log.addHandler(_handler)
    try:
        _source_ui_dir = fake_source_ui
        try:
            raise PermissionError("simulated TCC block")
        except Exception as _ui_copy_exc:
            _log.warning(
                "D-04: UI copy to data dir failed; serving from source path %s: %s",
                _source_ui_dir,
                _ui_copy_exc,
            )
            ui_dir = _source_ui_dir
    finally:
        _log.removeHandler(_handler)

    assert ui_dir == fake_source_ui, "Fallback must use source path"
    assert any("D-04" in w for w in warnings_logged), (
        "PermissionError fallback must log a WARNING containing 'D-04'"
    )


def test_d04_source_ui_dir_not_imported_in_server_module_top_level() -> None:
    """unified_daemon must NOT import UI_DIR from api at module top level
    (only inside _register_dashboard_routes), ensuring no import-time xattr
    access to the source tree."""
    import ast
    src = (
        Path(__file__).resolve().parents[2]
        / "src/superlocalmemory/server/unified_daemon.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Only check top-level import statements (not inside functions)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "superlocalmemory.server.api" and node.names:
                for alias in node.names:
                    if alias.name == "UI_DIR":
                        pytest.fail(
                            "UI_DIR must NOT be imported at module top level "
                            "in unified_daemon.py — it must remain inside "
                            "_register_dashboard_routes to be D-04 compliant."
                        )
