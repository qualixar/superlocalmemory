"""Shared fixtures for the CrewAI adapter tests.

The framework-free store core (``_v3_crewai_store``) is loaded by file path so
its tests run without ``crewai`` installed. Tests that exercise the
``StorageBackend`` wrapper use ``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_CORE_PATH = (
    Path(__file__).resolve().parent.parent
    / "crewai_superlocalmemory"
    / "_v3_crewai_store.py"
)


def _load_core_module():
    spec = importlib.util.spec_from_file_location("_slm_crewai_core", _CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def crewai_store(tmp_path, monkeypatch):
    """A V3CrewAIStore rooted at an isolated temp database."""
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_core_module()
    store = module.V3CrewAIStore(tmp_path / "crewai.db")
    try:
        yield store
    finally:
        store.close()
