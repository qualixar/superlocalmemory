# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every way to move text to another model passes one guard.

The first 4.2.2 guard covered the switch route, the upgrade plan and the
installer's request. Two more doors reached ReindexRunner.request_switch around
it: the dashboard Settings save (and the config API behind it), and a config
file naming another model, adopted at daemon start. So the guard lives in
request_switch itself: whoever asks (dashboard, CLI, MCP, installer, config
file), moving text to a managed model not approved for text is refused, unless
the explicit override is passed. A rollback is never refused: it goes back.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.core import embedding_reindex as er
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.storage import embedding_spaces as sp
from tests.test_storage.test_embedding_reindex_units import _target, store  # noqa: F401

GEMMA = EmbeddingConfig(provider="slm-media", model_name="google/embeddinggemma-2", dimension=768)


def _jobs(db_path) -> int:
    conn = sp.connect(db_path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {sp.JOBS}").fetchone()[0])
    finally:
        conn.close()


def test_the_runner_refuses_an_unapproved_managed_model(store):
    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    with pytest.raises(er.Refused, match="not calibrated"):
        runner.request_switch(GEMMA)
    assert _jobs(db_path) == 0


def test_the_explicit_override_queues_it(store):
    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    job = runner.request_switch(GEMMA, allow_unapproved=True)
    assert job["state"] == "queued" and _jobs(db_path) == 1


def test_a_model_the_person_configures_is_not_the_guards_business(store):
    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    assert runner.request_switch(_target())["state"] == "queued"


def test_a_config_file_naming_gemma_is_not_adopted(store):
    from superlocalmemory.core.embedding_live import PENDING_ATTR

    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    config = SimpleNamespace(**{PENDING_ATTR: GEMMA})
    assert runner.adopt_pending(config) is None
    assert _jobs(db_path) == 0
    assert runner.notice and "not calibrated" in runner.notice


def test_the_dashboard_settings_save_gets_the_reason(store):
    from superlocalmemory.server.routes.embedding_reindex import queue_if_new_space

    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(embedding_reindex=runner)))
    live = EmbeddingConfig(provider="sentence-transformers", model_name="old-model", dimension=8)
    response = queue_if_new_space(request, live, GEMMA)
    assert response is not None and response.status_code == 409
    assert _jobs(db_path) == 0


# -- the same fact on every surface: dashboard card, slm doctor, MCP health ------

def test_agents_see_the_same_advice_the_dashboard_and_doctor_show():
    from superlocalmemory.core.support_fit_guard import answer_check_status, live_advice

    on_gemma = answer_check_status(GEMMA)
    assert on_gemma["status"] == "warn" and on_gemma["advice"] == live_advice(GEMMA)
    assert on_gemma["recipe"] == "answer-support-v2" and on_gemma["approved_for_text"] is False
    nomic = EmbeddingConfig(provider="sentence-transformers",
                            model_name="nomic-ai/nomic-embed-text-v1.5", dimension=768)
    assert answer_check_status(nomic) == {
        "status": "ok", "model": "nomic-ai/nomic-embed-text-v1.5",
        "recipe": "answer-support-v2", "approved_for_text": True, "advice": ""}
