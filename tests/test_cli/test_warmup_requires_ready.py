# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm warmup` must not say PASS while the daemon reports itself not ready.

Audit 4.1.20 M1: /health said ``engine: initialized`` with ``ready: false``
and ``embedding_warm: false``, and warmup printed PASS. The first recall after
that PASS then paid the whole model load.
"""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

import pytest

from superlocalmemory.cli import commands
from superlocalmemory.cli.warmup_readiness import (
    describe_not_ready,
    is_fully_warm,
    wait_until_warm,
)

_STARTING = {"status": "ok", "engine": "initialized", "ready": False,
             "embedding_warm": False, "runtime_state": "warming"}
_READY = {"status": "ok", "engine": "initialized", "ready": True,
          "embedding_warm": True, "runtime_state": "serving_full"}


def _run(responses, timeout):
    it = iter(responses)
    with patch("superlocalmemory.cli.daemon.is_daemon_running", return_value=True), \
         patch("superlocalmemory.cli.daemon.daemon_request",
               side_effect=lambda *a, **k: next(it)), \
         patch("superlocalmemory.cli.warmup_readiness.time.sleep"), \
         patch("superlocalmemory.core.embeddings.EmbeddingService") as svc_cls:
        commands.cmd_warmup(Namespace(timeout=timeout))
    svc_cls.assert_not_called()


def test_engine_initialized_but_not_ready_is_not_pass(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _run([_STARTING], timeout=0)
    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "[PASS]" not in out
    assert "not ready yet" in out
    assert "loading the embedding model" in out
    assert "slm warmup --timeout" in out


def test_waits_for_a_starting_daemon_then_passes(capsys) -> None:
    _run([_STARTING, _STARTING, _READY], timeout=60)
    out = capsys.readouterr().out
    assert "waiting up to 60 s" in out
    assert "[PASS]" in out


def test_ready_but_model_cold_is_not_pass(capsys) -> None:
    cold = dict(_READY, embedding_warm=False)
    with pytest.raises(SystemExit):
        _run([cold], timeout=0)
    assert "[PASS]" not in capsys.readouterr().out


def test_daemon_that_stops_answering_fails(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        _run([_STARTING, None], timeout=60)
    assert exc.value.code == 1
    assert "stopped answering" in capsys.readouterr().out


def test_deadline_is_honoured() -> None:
    now = [0.0]
    verdict = wait_until_warm(
        lambda: _STARTING, 5.0,
        sleep=lambda s: now.__setitem__(0, now[0] + s),
        clock=lambda: now[0],
    )
    assert not verdict.ready and verdict.reachable
    assert 5.0 <= verdict.waited_seconds <= 6.0


def test_readiness_helpers() -> None:
    assert is_fully_warm(_READY)
    assert not is_fully_warm(_STARTING)
    assert not is_fully_warm(None)
    assert "semantic recall is not healthy" in describe_not_ready(
        dict(_READY, ready=False, runtime_state="serving_degraded"))


def test_a_genuinely_still_loading_model_keeps_the_old_wording() -> None:
    """No recorded failure yet (readiness missing, or the field is None):
    reads exactly as it did before this reason existed."""
    assert describe_not_ready(_STARTING) == "loading the embedding model"
    stuck_but_unrecorded = dict(
        _STARTING, readiness={"embedding_warmup_error": None},
    )
    assert describe_not_ready(stuck_but_unrecorded) == "loading the embedding model"


def test_a_failed_warmup_attempt_says_why_instead_of_implying_progress() -> None:
    """4.1.22: a first run offline with no cached embedding model never gets
    past "warming" on its own -- the recorded reason must say so, not read
    identically to a model that is merely a few seconds from ready."""
    stuck = dict(_STARTING, readiness={
        "embedding_warmup_error": "LocalEntryNotFoundError: outgoing traffic "
                                  "has been disabled",
    })
    out = describe_not_ready(stuck)
    assert "last attempt failed" in out
    assert "outgoing traffic has been disabled" in out


def test_a_recorded_failure_also_surfaces_outside_the_warming_state() -> None:
    """not_ready (engine/migrations not up yet, embedding never warmed
    either) gets the same reason instead of the bare "not loaded yet"."""
    not_ready = dict(
        _STARTING, ready=False, embedding_warm=False, runtime_state="not_ready",
        readiness={"embedding_warmup_error": "download failed"},
    )
    out = describe_not_ready(not_ready)
    assert "not loaded yet" in out
    assert "download failed" in out
