# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The maintenance step signs in the background; the daemon's start reads the key once."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
from superlocalmemory.trust import signatures
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import SIGNING, save


def _sched(db):
    sched = SimpleNamespace(_running=True, _db=db, steps=[])
    sched._record_step = lambda name, ok, err="": sched.steps.append((name, ok, err))
    return sched


def _count(db):
    return db.execute("SELECT COUNT(*) AS n FROM fact_signatures")[0]["n"]


def test_the_scheduler_step_signs_what_a_hook_saved(db, monkeypatch, signer) -> None:
    save(db, "Saved by a hook with no key.")
    save(db, "Another one, also with no key.")
    sched = _sched(db)
    MaintenanceScheduler._signing_pass(sched)
    assert _count(db) == 0 and sched.steps == [("signing pass", True, "")]  # no key, flag off
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    MaintenanceScheduler._signing_pass(sched)
    assert _count(db) == 2
    sched._running = False
    MaintenanceScheduler._signing_pass(sched)
    assert len(sched.steps) == 2  # a stopped scheduler does nothing


def test_a_failing_step_is_recorded_not_raised(db, monkeypatch) -> None:
    monkeypatch.setattr(signatures, "maintenance_step",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    sched = _sched(db)
    MaintenanceScheduler._signing_pass(sched)
    assert sched.steps == [("signing pass", False, "disk")]


def test_the_step_runs_in_the_cycle_after_the_trust_pass() -> None:
    source = inspect.getsource(MaintenanceScheduler)
    assert source.index("self._trust_pass()") < source.index("self._signing_pass()") \
        < source.index("self._slot_pass()")


def test_the_daemons_start_reads_the_key_once_and_not_on_the_event_loop() -> None:
    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon.lifespan)
    assert "asyncio.to_thread(_signing_key.start_for_daemon)" in source
    # one read: nothing else in the daemon module asks the keychain
    module = inspect.getsource(unified_daemon)
    assert module.count("_signing_key.start_for_daemon") == 1 and "KeychainKeyStore" not in module


def test_no_hook_or_recall_module_reads_the_keychain() -> None:
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
    offenders = []
    for folder in ("hooks", "retrieval", "mcp"):
        for path in (root / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "signing_key.load" in text or "KeychainKeyStore" in text or "start_for_daemon" in text:
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
