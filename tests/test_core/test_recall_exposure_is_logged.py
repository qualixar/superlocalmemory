# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Every recall records which memories it returned; SLM's warm-up does not.

The record is queued on the shared background writer, so recall itself stays
write-free (the 3.8.6 contract); these tests run the queued job inline.

3.8.6 dropped the write and only a slow, never-run test noticed: on the
author's store the last recall row in ``fact_access_log`` is dated 26 Jul
2026. Retention's access clock and the "active decisions" Core Memory block
read that table. This file runs in the default suite.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from superlocalmemory.core import recall_pipeline
from superlocalmemory.core.answer_check_scope import is_a_read, not_a_read


class _Log:
    def __init__(self, boom: bool = False):
        self.calls, self.boom = [], boom

    def store_access_batch(self, **kwargs):
        if self.boom:
            raise RuntimeError("disk full")
        self.calls.append(kwargs)
        return len(kwargs["fact_ids"])


@pytest.fixture(autouse=True)
def _inline_background(monkeypatch):
    monkeypatch.setattr(
        "superlocalmemory.storage.deferred_writes.submit_background", lambda fn: fn(),
    )


def _results(*ids):
    return [SimpleNamespace(fact=SimpleNamespace(fact_id=i)) for i in ids]


def test_the_returned_memories_are_logged_as_recalled():
    log = _Log()
    recall_pipeline._log_recall_exposure(log, _results("a", "b"), "default")
    assert log.calls == [{"fact_ids": ["a", "b"], "profile_id": "default", "access_type": "recall"}]


def test_warm_up_recalls_are_not_reads():
    log = _Log()
    with not_a_read():
        assert not is_a_read()
        recall_pipeline._log_recall_exposure(log, _results("a"), "default")
    assert log.calls == [] and is_a_read()


def test_a_failed_log_never_fails_the_recall_and_empty_answers_log_nothing():
    recall_pipeline._log_recall_exposure(_Log(boom=True), _results("a"), "default")
    log = _Log()
    recall_pipeline._log_recall_exposure(log, [], "default")
    recall_pipeline._log_recall_exposure(None, _results("a"), "default")
    assert log.calls == []


def test_the_write_is_queued_not_run_on_the_recall_thread(monkeypatch):
    queued, log = [], _Log()
    monkeypatch.setattr(
        "superlocalmemory.storage.deferred_writes.submit_background", queued.append,
    )
    recall_pipeline._log_recall_exposure(log, _results("a"), "default")
    assert log.calls == [] and len(queued) == 1
    queued[0]()
    assert log.calls[0]["fact_ids"] == ["a"]


def test_the_pipeline_logs_and_the_warm_up_is_marked():
    from superlocalmemory.server import recall_warmup

    assert "_log_recall_exposure(access_log, response.results, profile_id)" in inspect.getsource(
        recall_pipeline)
    assert "not_a_read()" in inspect.getsource(recall_warmup.run_warmup_recalls)
