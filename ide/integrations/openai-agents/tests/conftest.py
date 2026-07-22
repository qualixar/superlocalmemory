"""Shared fixtures for the OpenAI Agents SDK adapter tests.

The framework-free session store core (``_v3_session_store``) is loaded
directly by file path so its tests run even when ``openai-agents`` is not
installed.  Tests that exercise the ``SLMSession`` wrapper use
``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_STORE_PATH = (
    Path(__file__).resolve().parent.parent
    / "openai_agents_superlocalmemory"
    / "_v3_session_store.py"
)


def _load_store_module():
    spec = importlib.util.spec_from_file_location("_slm_oa_session_core", _STORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def session_store(tmp_path, monkeypatch):
    """A V3SessionStore rooted at an isolated temp database."""
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_store_module()
    store = module.V3SessionStore(tmp_path / "session.db")
    try:
        yield store
    finally:
        store.close()
