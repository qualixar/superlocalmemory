"""Shared fixtures for the LangGraph adapter tests.

The framework-free key-value core (``_v3_kv_store``) is loaded directly by file
path so its tests run even when ``langgraph`` is not installed — importing the
package normally would pull in the ``BaseStore`` wrapper and require the
framework. Tests that exercise the wrapper itself use ``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_KV_PATH = Path(__file__).resolve().parent.parent / "langgraph_superlocalmemory" / "_v3_kv_store.py"


def _load_kv_module():
    spec = importlib.util.spec_from_file_location("_slm_lg_kv_core", _KV_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kv_store(tmp_path, monkeypatch):
    """A V3KVStore rooted at an isolated temp database."""
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_kv_module()
    store = module.V3KVStore(tmp_path / "kv.db")
    try:
        yield store
    finally:
        store.close()
