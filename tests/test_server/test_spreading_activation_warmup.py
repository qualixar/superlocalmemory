# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""F2: daemon startup must warm the spreading-activation channel.

The ``--fast`` warmup recalls deliberately skip spreading activation (and the
Mode-C remote agentic verification), which left the first FULL user recall
paying the cold graph-load cost. ``_warm_spreading_activation`` warms that
channel directly — local graph work only, no remote/LLM call.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

from superlocalmemory.server.unified_daemon import _warm_spreading_activation


class _Embedder:
    def embed(self, text: str):
        return [0.1, 0.2, 0.3]


class _Retrieval:
    def __init__(self, sa, embedder) -> None:
        self._spreading_activation = sa
        self._embedder = embedder


class _Engine:
    def __init__(self, retr, profile_id: str = "team-a") -> None:
        self._retrieval_engine = retr
        self.profile_id = profile_id


class _Runtime:
    """Fake profile runtime whose operation_nowait() yields a snapshot."""

    def __init__(self, snap: object | None) -> None:
        self._snap = snap
        self.calls = 0

    @contextlib.contextmanager
    def operation_nowait(self):
        self.calls += 1
        yield self._snap


def test_warmup_exercises_spreading_activation_for_active_profile() -> None:
    sa = MagicMock()
    engine = _Engine(_Retrieval(sa, _Embedder()), profile_id="team-a")
    runtime = _Runtime(snap=object())

    ran = _warm_spreading_activation(engine, runtime)

    assert ran is True
    sa.search.assert_called_once()
    _, kwargs = sa.search.call_args
    assert kwargs.get("profile_id") == "team-a"
    assert runtime.calls == 1  # warmup takes its own short lease


def test_warmup_skips_when_profile_transition_preempts() -> None:
    sa = MagicMock()
    engine = _Engine(_Retrieval(sa, _Embedder()))
    runtime = _Runtime(snap=None)  # a transition is draining → no snapshot

    ran = _warm_spreading_activation(engine, runtime)

    assert ran is False
    sa.search.assert_not_called()


def test_warmup_is_fail_soft_when_channel_absent() -> None:
    engine = _Engine(retr=None)
    assert _warm_spreading_activation(engine, runtime=None) is False
