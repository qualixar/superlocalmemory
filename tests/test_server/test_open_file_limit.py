"""The daemon raises a low open-file soft limit (launchd's 256) at start."""

from __future__ import annotations

import pytest

# POSIX only: Windows has no resource module and no open-file soft limit to raise.
resource = pytest.importorskip("resource")

from superlocalmemory.server import open_files


def test_low_soft_limit_is_raised_within_the_hard_limit(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda _k: (256, 10240))
    monkeypatch.setattr(resource, "setrlimit", lambda _k, v: calls.append(v))
    assert open_files.raise_open_file_limit() == (256, 8192)
    assert calls == [(8192, 10240)]


def test_hard_limit_below_target_caps_it(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(resource, "getrlimit", lambda _k: (256, 1024))
    monkeypatch.setattr(resource, "setrlimit", lambda _k, v: calls.append(v))
    assert open_files.raise_open_file_limit() == (256, 1024)


def test_a_high_limit_is_never_lowered(monkeypatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _k: (1048576, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda *_a: (_ for _ in ()).throw(AssertionError))
    assert open_files.raise_open_file_limit() == (1048576, 1048576)


def test_refusal_does_not_raise(monkeypatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _k: (256, 10240))

    def refuse(*_a):
        raise ValueError("not allowed")

    monkeypatch.setattr(resource, "setrlimit", refuse)
    assert open_files.raise_open_file_limit() is None
