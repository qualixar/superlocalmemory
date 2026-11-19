# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Test isolation that does not depend on ``tests/conftest.py`` being loaded.

Importing this module (once per process) installs, before any product module
is imported:

* an audit hook that refuses every file and SQLite touch under a live data root
  (the account's ``~/.superlocalmemory`` and any root the invoking shell set),
  and every connection to the live daemon ports;
* a pytest-owned HOME and data root in the environment.

It is reached three ways, so no single switch turns it off:

* ``pyproject.toml`` loads it as a plugin (``addopts = -p tests._isolation_plugin``)
  -- this covers ``--noconftest``;
* ``tests/__init__.py`` imports it, so any test module under ``tests/`` installs
  it even when ``-o addopts=`` or another ini file drops the plugin;
* ``tests/conftest.py`` registers it as a plugin when the addopts did not.

As a registered plugin it also fails any test whose refused touch was swallowed
by product code.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from tests.isolation_guard import LiveRootGuard, live_data_roots

PLUGIN_NAME = "tests._isolation_plugin"
LIVE_DAEMON_PORTS = frozenset({8765, 8767})

# Capture the live roots before the environment below is rewritten.
#
# A process spawned by a test (multiprocessing "spawn") imports this module
# again, but by then the parent has already pointed SLM_DATA_DIR at the test's
# own tmp root. Recomputing would protect that tmp root and refuse the child's
# legitimate writes, so a child inherits the parent's list instead. The
# account's own ~/.superlocalmemory is always added, so an inherited list can
# only widen protection, never drop the real store.
_LIVE_ROOTS_ENV = "SLM_TEST_LIVE_ROOTS"
_REAL_HOME = Path.home().resolve()
_INHERITED_ROOTS = (
    os.environ.get(_LIVE_ROOTS_ENV, "")
    if os.environ.get("SLM_TEST_ISOLATION") == "1" else ""
)
if _INHERITED_ROOTS:
    _LIVE_DATA_ROOTS = frozenset(
        Path(root) for root in _INHERITED_ROOTS.split(os.pathsep) if root
    ) | live_data_roots(env={})
    _REAL_DATA_ROOT = Path(
        os.environ.get("SLM_TEST_REAL_DATA_ROOT") or _REAL_HOME / ".superlocalmemory"
    )
else:
    _LIVE_DATA_ROOTS = live_data_roots()
    # The env-selected (or default) root, kept for SLM_TEST_REAL_DATA_ROOT below.
    _REAL_DATA_ROOT = Path(
        os.environ.get("SLM_DATA_DIR")
        or os.environ.get("SL_MEMORY_PATH")
        or os.environ.get("SLM_HOME")
        or (_REAL_HOME / ".superlocalmemory")
    ).expanduser().resolve(strict=False)
_LIVE_ROOT_GUARD = LiveRootGuard(_LIVE_DATA_ROOTS)


def _pytest_isolation_audit(event: str, args: tuple) -> None:
    """Deny live-state access and public-daemon socket connections."""
    _LIVE_ROOT_GUARD.audit(event, args)
    if event == "socket.connect" and len(args) >= 2:
        address = args[1]
        if (
            isinstance(address, tuple)
            and len(address) >= 2
            and str(address[0]).lower() in {"127.0.0.1", "localhost", "::1"}
            and int(address[1]) in LIVE_DAEMON_PORTS
        ):
            raise PermissionError(
                f"pytest denied live SLM daemon port: {address[1]}"
            )


sys.addaudithook(_pytest_isolation_audit)

# Establish a pytest-owned namespace before test modules are imported. Several
# production modules resolve HOME and daemon paths at import time, so a normal
# fixture is too late to protect the user's live installation.
#
# A spawned child of an isolated session keeps the parent's namespace as it
# found it in the environment: minting a fresh HOME and data root per child
# gave every child its own lock and state files, so cross-process contracts
# (one embedding worker, one row allocator) were never actually exercised.
if _INHERITED_ROOTS:
    _TEST_ISOLATION_DIR = None
    _TEST_ROOT = Path(os.environ["SLM_TEST_ROOT"])
else:
    _TEST_ISOLATION_DIR = tempfile.TemporaryDirectory(prefix="slm-pytest-")
    _TEST_ROOT = Path(_TEST_ISOLATION_DIR.name).resolve()
    _TEST_HOME = _TEST_ROOT / "home"
    _TEST_DATA_DIR = _TEST_ROOT / "canonical-data"
    _TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["SLM_TEST_ISOLATION"] = "1"
    os.environ["SLM_TEST_ROOT"] = str(_TEST_ROOT)
    os.environ["SLM_TEST_REAL_DATA_ROOT"] = str(_REAL_DATA_ROOT)
    os.environ[_LIVE_ROOTS_ENV] = os.pathsep.join(sorted(str(r) for r in _LIVE_DATA_ROOTS))
    os.environ["HOME"] = str(_TEST_HOME)
    os.environ["USERPROFILE"] = str(_TEST_HOME)
    os.environ["XDG_CONFIG_HOME"] = str(_TEST_HOME / ".config")
    os.environ["XDG_CACHE_HOME"] = str(_TEST_HOME / ".cache")
    os.environ["XDG_DATA_HOME"] = str(_TEST_HOME / ".local" / "share")
    os.environ["SLM_DATA_DIR"] = str(_TEST_DATA_DIR)
    os.environ["SL_MEMORY_PATH"] = str(_TEST_ROOT / "wrong-legacy-alias")
    os.environ["SLM_HOME"] = str(_TEST_ROOT / "wrong-hook-alias")
    os.environ["SLM_DAEMON_PORT"] = str(20_000 + secrets.randbelow(40_000))
    os.environ["SLM_TEST_INSTANCE_CAPABILITY"] = secrets.token_urlsafe(32)
    # 4.1.25 installs the answer check (~1 GB) by default; no test may start
    # that download by accident. Tests of the default clear this themselves.
    os.environ["SLM_SKIP_LAYA"] = "1"
    # Pictures, documents and the model packs download by themselves in a real daemon on a 16 GB+
    # computer (4.2.30). No test or CI run may ever start that: the daemon loop reads this and stays idle.
    # Tests of the automatic path pass their own environment to ``media_auto.tick``.
    os.environ["SLM_NO_AUTO_DOWNLOAD"] = "1"
    # No test may register a real login service. A test HOME keeps the plist
    # out of the user's LaunchAgents, but ``launchctl load`` still hands the job
    # to the user's launchd, which keeps its daemon running (and restarts it)
    # after the suite: on 16 Nov a ``-n 3`` run left eight such jobs holding
    # ports and memory (F-38). A shim first on PATH, inherited by every child
    # process, logs the verbs that register a job and passes the rest through.
    _SHIM_DIR = _TEST_ROOT / "bin"
    _SHIM_DIR.mkdir(parents=True, exist_ok=True)
    _real_launchctl = shutil.which("launchctl")
    if _real_launchctl:
        _shim = _SHIM_DIR / "launchctl"
        _shim.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  load|bootstrap|submit|enable|kickstart|start)\n"
            "    echo \"$*\" >> \"$SLM_TEST_ROOT/launchctl-refused.log\"\n"
            "    exit 0 ;;\n"
            "esac\n"
            f"exec '{_real_launchctl}' \"$@\"\n",
            encoding="utf-8")
        _shim.chmod(0o755)
        os.environ["PATH"] = f"{_SHIM_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
for _unsafe_env in (
    "SLM_TEST_ALLOW_LIVE_HOME",
    "SLM_HOOK_DAEMON_URL",
    "SLM_MESH_PEER_URL",
    "SLM_MESH_SHARED_SECRET",
    "SLM_MESH_HOST",
    "SLM_MESH_WS_PORT",
    "SLM_MESH_DISCOVERY",
    "SLM_DAEMON_HOST",
    "SLM_HOST",
):
    os.environ.pop(_unsafe_env, None)


def ensure_registered(config: pytest.Config) -> None:
    """Register this module as a plugin if the command line did not."""
    manager = config.pluginmanager
    module = sys.modules[__name__]
    if not manager.is_registered(module):
        manager.register(module, PLUGIN_NAME)


@pytest.fixture(autouse=True, scope="function")
def _block_live_slm_home_writes(tmp_path, monkeypatch):
    """Give each test an isolated data root with no live-home escape hatch.

    Afterwards, fail the test if anything tried to touch a live root -- even
    when the code under test caught the PermissionError and carried on.
    """
    _LIVE_ROOT_GUARD.drain()
    data_dir = tmp_path
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SLM_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong-legacy-alias"))
    monkeypatch.setenv("SLM_HOME", str(tmp_path / "wrong-hook-alias"))
    yield
    refused = _LIVE_ROOT_GUARD.drain()
    if refused:
        pytest.fail(
            "test reached the live SLM data root (refused, but it must use a "
            "tmp root):\n  " + "\n  ".join(refused[:10]),
            pytrace=False,
        )


def pytest_unconfigure(config) -> None:
    """Release the collection-time namespace before interpreter teardown.

    ``TemporaryDirectory`` otherwise waits for its weakref finalizer, which
    emits a ``ResourceWarning`` under ``-X dev`` and hides real fixture leaks
    in the same shutdown window.
    """
    if _TEST_ISOLATION_DIR is not None:
        _TEST_ISOLATION_DIR.cleanup()
