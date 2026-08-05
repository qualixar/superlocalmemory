# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""pytest fixtures for tests/optimize/proxy/ tests.

Port-isolation guard
--------------------
``ensure_proxy_running()`` in lifecycle.py probes ``http://127.0.0.1:8765/health``
when ``cfg.proxy_enabled`` is True.  If a live SLM daemon is running on port 8765
at the time the gate runs (the common case on a developer machine), the urlopen
call can block for up to 1 second per test invocation, and under exotic network
conditions could deadlock the worker.

The ``_proxy_config_disabled`` autouse fixture resets the module-level
``optimize.config._store`` to None before each test so that
``get_optimize_config()`` returns ``DEFAULT_OPTIMIZE_CONFIG`` (which has
``proxy_enabled=False``).  This guarantees that ``ensure_proxy_running()``
returns False immediately without touching port 8765.

Tests that intentionally want a proxy-enabled config can still call
``_set_config_store(store)`` inside their own body — the autouse fixture only
ensures the *default* state is proxy-disabled at the start of each test, and
restores the previous value at teardown via ``monkeypatch``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _proxy_config_disabled(monkeypatch) -> None:
    """Ensure the optimize config store is reset to None for every proxy test.

    This prevents ``ensure_proxy_running()`` from probing port 8765 even when
    the developer's real ``optimize.json`` has ``proxy_enabled=True``.  The
    monkeypatch restores the original ``_store`` value at test teardown,
    leaving other test modules unaffected.
    """
    import superlocalmemory.optimize.config as _opt_cfg_mod
    monkeypatch.setattr(_opt_cfg_mod, "_store", None)
