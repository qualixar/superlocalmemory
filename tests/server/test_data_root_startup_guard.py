"""Daemon startup must fail before publishing identity for conflicting roots."""

import pytest


def test_start_server_checks_root_conflict_before_identity(monkeypatch) -> None:
    from superlocalmemory.server import unified_daemon

    class ExpectedConflict(RuntimeError):
        pass

    published = False

    def reject_conflict() -> None:
        raise ExpectedConflict("two populated roots")

    def record_publish(*_args, **_kwargs) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(
        unified_daemon,
        "assert_no_durable_root_conflict",
        reject_conflict,
        raising=False,
    )
    monkeypatch.setattr(unified_daemon, "_publish_process_descriptor", record_publish)

    with pytest.raises(ExpectedConflict):
        unified_daemon.start_server(port=19131)

    assert published is False
