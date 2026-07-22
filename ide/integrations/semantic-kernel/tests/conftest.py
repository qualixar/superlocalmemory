"""Shared fixtures for the Semantic Kernel adapter tests.

The framework-free record core (``_v3_record_store``) is loaded by file path so
its tests run without ``semantic-kernel`` installed. Tests that exercise the
``VectorStoreCollection`` wrapper use ``importorskip``.
"""

import importlib.util
from pathlib import Path

import pytest

_REC_PATH = (
    Path(__file__).resolve().parent.parent
    / "semantic_kernel_superlocalmemory"
    / "_v3_record_store.py"
)


def _load_record_module():
    spec = importlib.util.spec_from_file_location("_slm_sk_rec_core", _REC_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def record_store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_TEST_ISOLATION", "1")
    module = _load_record_module()
    store = module.V3RecordStore(tmp_path / "rec.db")
    try:
        yield store
    finally:
        store.close()
