# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""The legacy offline spool may clean receipts, never raw evidence."""

from __future__ import annotations

import sqlite3

from superlocalmemory.cli.pending_store import cleanup_stale, store_pending


def test_cleanup_stale_never_deletes_unprocessed_rows(tmp_path) -> None:
    pending_id = store_pending("pending evidence", base_dir=tmp_path)
    failed_id = store_pending("failed evidence", base_dir=tmp_path)
    conn = sqlite3.connect(tmp_path / "pending.db")
    try:
        conn.execute(
            "UPDATE pending_memories SET created_at='2000-01-01T00:00:00'"
        )
        conn.execute(
            "UPDATE pending_memories SET status='failed', retry_count=99 "
            "WHERE id=?",
            (failed_id,),
        )
        conn.commit()
    finally:
        conn.close()

    cleanup_stale(base_dir=tmp_path)

    conn = sqlite3.connect(tmp_path / "pending.db")
    try:
        rows = conn.execute(
            "SELECT id, content, status FROM pending_memories ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        (pending_id, "pending evidence", "pending"),
        (failed_id, "failed evidence", "pending"),
    ]
