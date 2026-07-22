"""Shared fixtures for the AutoGen adapter tests.

The framework-free store core (``_v3_autogen_store``) is loaded by file path
so its tests run without ``autogen-agentchat`` installed. Tests that exercise
the ``Memory`` ABC wrapper use ``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_CORE_PATH = (
    Path(__file__).resolve().parent.parent
    / "autogen_superlocalmemory"
    / "_v3_autogen_store.py"
)


def _load_core_module():
    spec = importlib.util.spec_from_file_location("_slm_autogen_core", _CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def autogen_store(tmp_path, monkeypatch):
    """A V3AutogenStore rooted at an isolated temp database."""
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_core_module()
    store = module.V3AutogenStore(tmp_path / "autogen.db")
    try:
        yield store
    finally:
        store.close()
