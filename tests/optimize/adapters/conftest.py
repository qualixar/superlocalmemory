# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""pytest fixtures for tests/optimize/adapters/ tests.

Port-isolation guard
--------------------
``test_ensure_proxy_running_returns_bool()`` calls ``ensure_proxy_running()``
which probes ``http://127.0.0.1:8765/health`` when ``cfg.proxy_enabled`` is True.
If the developer's ``optimize.json`` has ``proxy_enabled=True`` and a live daemon
is running on 8765, this creates a 1-second blocking call per test invocation
and can, in edge cases, cause a gate deadlock.

The ``_proxy_config_disabled`` autouse fixture resets the module-level
``optimize.config._store`` to None before each test so that
``get_optimize_config()`` returns ``DEFAULT_OPTIMIZE_CONFIG`` (``proxy_enabled=False``).

Tests that need a specific config (e.g., ``test_wrap_agent_proxy_disabled_returns_1``)
call ``_set_config_store()`` inside their own body after the autouse fixture has run;
``monkeypatch`` automatically restores ``_store`` to None at teardown.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _proxy_config_disabled(monkeypatch) -> None:
    """Reset the optimize config store to None before each adapter test.

    Prevents ``ensure_proxy_running()`` from probing port 8765, even when
    the real ``optimize.json`` has ``proxy_enabled=True``.
    """
    import superlocalmemory.optimize.config as _opt_cfg_mod
    monkeypatch.setattr(_opt_cfg_mod, "_store", None)
