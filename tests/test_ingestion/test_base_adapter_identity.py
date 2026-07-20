# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Adapters must use the exact-instance daemon capability client."""

from unittest.mock import MagicMock


def test_adapter_ingest_uses_owned_daemon_request(monkeypatch) -> None:
    from superlocalmemory.ingestion.base_adapter import BaseAdapter, IngestItem

    daemon_request = MagicMock(return_value={"ingested": True})
    monkeypatch.setattr(
        "superlocalmemory.cli.daemon.daemon_request",
        daemon_request,
    )
    adapter = BaseAdapter()
    item = IngestItem(
        content="Dana approved the recovery plan.",
        dedup_key="mail-1",
        metadata={"mailbox": "reliability"},
    )

    assert adapter._ingest(item) is True
    daemon_request.assert_called_once_with(
        "POST",
        "/ingest",
        {
            "content": item.content,
            "source_type": "unknown",
            "dedup_key": item.dedup_key,
            "metadata": item.metadata,
        },
    )
