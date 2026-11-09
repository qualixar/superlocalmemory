# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The config.json lock lives in infra so core can take it (4.2.14).

core/config.py may not import the server layer (.importlinter ratchet), and
SLMConfig.save() must take the very lock `slm config set` (server/
config_file.update_config) takes, or one writer silently undoes the other.
"""

from __future__ import annotations

import threading
from pathlib import Path


def test_server_module_uses_the_infra_lock() -> None:
    from superlocalmemory.infra import config_lock
    from superlocalmemory.server import config_file

    # backup_bundle imports the old private name; it must stay the same lock.
    assert config_file._file_lock is config_lock.config_file_lock


def test_lock_file_sits_next_to_the_target(tmp_path: Path) -> None:
    from superlocalmemory.infra.config_lock import config_file_lock

    target = tmp_path / "config.json"
    with config_file_lock(target):
        assert (tmp_path / ".config.json.lock").exists()


def test_the_same_thread_may_take_it_again(tmp_path: Path) -> None:
    # A nested take on a second open file would block forever on flock
    # (BSD/macOS and Linux alike): the lock is re-entrant per thread.
    from superlocalmemory.infra.config_lock import config_file_lock

    target = tmp_path / "config.json"
    done = threading.Event()

    def nested() -> None:
        with config_file_lock(target):
            with config_file_lock(target):
                pass
        done.set()

    worker = threading.Thread(target=nested, daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert done.is_set(), "nested take of the config lock deadlocked"


def test_another_thread_waits_for_the_holder(tmp_path: Path) -> None:
    from superlocalmemory.infra.config_lock import config_file_lock

    target = tmp_path / "config.json"
    entered = threading.Event()

    def other() -> None:
        with config_file_lock(target):
            entered.set()

    with config_file_lock(target):
        worker = threading.Thread(target=other, daemon=True)
        worker.start()
        assert not entered.wait(0.3), "a second thread entered a held lock"
    worker.join(timeout=10)
    assert entered.is_set()
