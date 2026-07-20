# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Fail-closed proof that pytest cannot inherit the user's live SLM state.

These values are captured during collection, before function-scoped fixtures.
That is intentional: modules import daemon path/port constants at collection
time, so isolating only inside a test body is already too late.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

_COLLECTION_HOME = Path.home().resolve()
_COLLECTION_DATA_DIR = Path(os.environ.get("SLM_DATA_DIR", "")).resolve()
_COLLECTION_TEST_ROOT = Path(os.environ.get("SLM_TEST_ROOT", "")).resolve()
_COLLECTION_DAEMON_PORT = os.environ.get("SLM_DAEMON_PORT")
_COLLECTION_CAPABILITY = os.environ.get("SLM_TEST_INSTANCE_CAPABILITY")


def test_collection_uses_pytest_owned_home() -> None:
    """Collection must already be under the pytest-owned state root."""
    isolation = os.environ.get("SLM_TEST_ISOLATION")
    assert isolation == "1", "pytest isolation was not enabled before collection"
    assert _COLLECTION_TEST_ROOT.name.startswith("slm-pytest-")
    assert _COLLECTION_HOME.is_relative_to(_COLLECTION_TEST_ROOT)
    assert _COLLECTION_DATA_DIR.is_relative_to(_COLLECTION_TEST_ROOT)


def test_collection_cannot_probe_live_daemon_ports() -> None:
    """A test process must not inherit either public daemon port."""
    assert _COLLECTION_DAEMON_PORT is not None
    assert int(_COLLECTION_DAEMON_PORT) not in {8765, 8767}


def test_collection_has_unique_instance_capability() -> None:
    """Test-owned daemons require an unguessable per-run capability."""
    assert _COLLECTION_CAPABILITY is not None
    assert len(_COLLECTION_CAPABILITY) >= 43


def test_real_slm_root_is_denied_even_through_direct_path() -> None:
    real_root_value = os.environ.get("SLM_TEST_REAL_DATA_ROOT")
    assert real_root_value, "real data-root guard was not installed"
    probe = Path(real_root_value) / "__slm_pytest_guard_probe__"
    with pytest.raises(PermissionError, match="live SLM state"):
        probe.open("w")


@pytest.mark.parametrize("port", [8765, 8767])
def test_live_public_daemon_ports_are_denied(port: int) -> None:
    sock = socket.socket()
    try:
        with pytest.raises(PermissionError, match="live SLM daemon port"):
            sock.connect_ex(("127.0.0.1", port))
    finally:
        sock.close()


def test_child_python_inherits_the_isolated_namespace() -> None:
    code = (
        "import os; from pathlib import Path; "
        "print(Path.home().resolve()); print(Path(os.environ['SLM_DATA_DIR']).resolve()); "
        "print(os.environ['SLM_DAEMON_PORT'])"
    )
    output = subprocess.check_output([sys.executable, "-c", code], text=True)
    home, data_dir, port = output.splitlines()
    real_root = Path(os.environ["SLM_TEST_REAL_DATA_ROOT"])
    assert Path(home).is_relative_to(_COLLECTION_TEST_ROOT)
    assert Path(data_dir) != real_root
    assert not Path(data_dir).is_relative_to(real_root)
    assert int(port) not in {8765, 8767}


def test_pytest_harness_cancels_test_owned_timer_threads() -> None:
    source = (Path(__file__).resolve().parents[1] / "conftest.py").read_text(
        encoding="utf-8"
    )
    assert "def _cancel_test_timer_threads" in source
    assert "isinstance(thread, threading.Timer)" in source
    assert "thread.cancel()" in source
    assert "thread.join(" in source


def test_pytest_harness_explicitly_closes_its_collection_namespace() -> None:
    """Repeated focused exits must not defer temp-root cleanup to ``__del__``."""
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONDEVMODE"] = "1"
    env["PYTHONWARNINGS"] = "always::ResourceWarning"

    for _attempt in range(2):
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "dev",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_resource_lifecycle.py",
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "ResourceWarning" not in result.stderr, result.stderr
        assert "PytestUnraisableExceptionWarning" not in result.stderr, result.stderr


def test_optional_native_backend_runs_in_a_dedicated_process_lane() -> None:
    root = Path(__file__).resolve().parents[2]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    vector_tests = (root / "tests/vector/test_lancedb_backend.py").read_text(
        encoding="utf-8"
    )
    workflow = (root / ".github/workflows/test.yml").read_text(encoding="utf-8")

    assert "not native" in pyproject
    assert "native: marks optional native-backend tests" in pyproject
    assert "pytest.mark.native" in vector_tests
    assert "from superlocalmemory.vector.lancedb_backend import" not in vector_tests.split(
        "@pytest.fixture", maxsplit=1
    )[0]
    assert "native-lancedb" in workflow
    assert "python -m pytest -q -m native tests/vector" in workflow
