"""Shared fixtures for the Agent Framework adapter tests.

The framework-free message core (``_v3_message_store``) is loaded by file path
so its tests run without ``agent-framework-core`` installed. Provider tests use
``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_MSG_PATH = (
    Path(__file__).resolve().parent.parent
    / "agent_framework_superlocalmemory"
    / "_v3_message_store.py"
)


def _load_message_module():
    spec = importlib.util.spec_from_file_location("_slm_af_msg_core", _MSG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def message_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_message_module()
    store = module.V3MessageStore(tmp_path / "m.db")
    try:
        yield store
    finally:
        store.close()
