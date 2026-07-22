"""Shared fixtures for the Google ADK adapter tests.

The framework-free ADK store core (``_v3_adk_store``) is loaded directly by
file path so its tests run even when ``google-adk`` is not installed.  Tests
that exercise the ``SuperLocalMemoryService`` wrapper use ``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_STORE_PATH = (
    Path(__file__).resolve().parent.parent
    / "google_adk_superlocalmemory"
    / "_v3_adk_store.py"
)


def _load_store_module():
    spec = importlib.util.spec_from_file_location("_slm_adk_store_core", _STORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adk_store(tmp_path, monkeypatch):
    """A V3ADKStore rooted at an isolated temp database."""
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_store_module()
    store = module.V3ADKStore(tmp_path / "adk.db")
    try:
        yield store
    finally:
        store.close()
