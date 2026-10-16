# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm ops resolve` writes memory.db through the canonical ``memory_write``.

It opened its own connection with sqlite3's default 5 s wait and no process
write lock, so in the daemon (dashboard Ops Health) it raced the memory writer
and failed with "database is locked" whenever a write batch ran longer than 5 s.
Through ``memory_write`` it queues behind in-process writers instead.
"""

from __future__ import annotations

import sqlite3
import threading
import time

from superlocalmemory.core.ops_remediation import _action_cancel
from superlocalmemory.storage.write_lock import get_write_lock
from tests.test_resilience.test_ops_resolve import _apply_migrations, _seed_dead_letter


def test_cancel_waits_for_the_process_write_lock(tmp_path) -> None:
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    _apply_migrations(conn)
    _seed_dead_letter(conn)
    conn.close()

    held = threading.Event()

    def hold() -> None:
        with get_write_lock(db):
            held.set()
            time.sleep(0.4)

    t = threading.Thread(target=hold)
    t.start()
    held.wait(2)
    start = time.monotonic()
    result = _action_cancel(db, "dlq-resolve-1")
    waited = time.monotonic() - start
    t.join()
    assert result["success"] is True
    assert waited >= 0.3, f"cancel did not wait for the write lock ({waited:.2f}s)"
