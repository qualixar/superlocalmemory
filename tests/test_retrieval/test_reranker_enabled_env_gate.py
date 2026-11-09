# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The real ``init_retrieval`` wiring gate (not just the config method in
isolation) must honour SLM_RERANKER_ENABLED, the lite bot-host profile's
switch for turning the reranker off on a RAM-constrained shared host.

``init_reranker`` itself is patched to a cheap stub — the subprocess it would
spawn is a real ~100-200MB PyTorch/ONNX load, which has no place in a unit
test — but the ROUTING decision under test (whether ``init_retrieval`` calls
it at all) is never mocked, matching the pattern in
tests/test_retrieval/test_remote_reranker_wiring.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine_wiring import init_retrieval
from superlocalmemory.core.modes import Mode
from tests.test_retrieval.cross_scope_fixture import build_store


@pytest.fixture()
def wired_config_and_store(tmp_path):
    store = build_store(tmp_path / "w.db")
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.retrieval.use_cross_encoder = True  # the persisted choice: reranker ON
    embedder = SimpleNamespace(embed=lambda text: [0.0] * 8, is_ready=True)
    return config, store, embedder


def test_env_override_stops_the_reranker_from_being_wired(wired_config_and_store, monkeypatch):
    config, store, embedder = wired_config_and_store
    monkeypatch.setenv("SLM_RERANKER_ENABLED", "false")
    with patch("superlocalmemory.core.engine_wiring.init_reranker") as mock_init:
        engine = init_retrieval(config, store.db, embedder, None, None, vector_store=None)
        try:
            mock_init.assert_not_called()
            assert engine._reranker is None
        finally:
            engine.close()


def test_without_the_override_the_persisted_choice_still_wires_it(wired_config_and_store, monkeypatch):
    """Control: with no env override, `use_cross_encoder=True` still reaches
    init_reranker exactly as before — this change must not flip the default."""
    config, store, embedder = wired_config_and_store
    monkeypatch.delenv("SLM_RERANKER_ENABLED", raising=False)
    # warmup_sync is called by a background thread init_retrieval starts —
    # give the stub one so that thread doesn't raise (and warn) in the suite.
    stub_reranker = SimpleNamespace(warmup_sync=lambda timeout=180: True)
    with patch("superlocalmemory.core.engine_wiring.init_reranker", return_value=stub_reranker) as mock_init:
        engine = init_retrieval(config, store.db, embedder, None, None, vector_store=None)
        try:
            mock_init.assert_called_once()
            assert engine._reranker is stub_reranker
        finally:
            engine.close()


def test_env_override_true_cannot_re_enable_a_persisted_false(wired_config_and_store, monkeypatch):
    """The override can turn it off; it can also turn it on over a persisted
    False — both directions are intentional ("opt in via env")."""
    config, store, embedder = wired_config_and_store
    config.retrieval.use_cross_encoder = False
    monkeypatch.setenv("SLM_RERANKER_ENABLED", "true")
    stub_reranker = SimpleNamespace(warmup_sync=lambda timeout=180: True)
    with patch("superlocalmemory.core.engine_wiring.init_reranker", return_value=stub_reranker) as mock_init:
        engine = init_retrieval(config, store.db, embedder, None, None, vector_store=None)
        try:
            mock_init.assert_called_once()
            assert engine._reranker is stub_reranker
        finally:
            engine.close()
