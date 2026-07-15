# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Keep the legacy pending database out of public write paths."""

from pathlib import Path


def test_shipped_runtime_does_not_call_legacy_pending_store() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
    allowed_owner = root / "cli" / "pending_store.py"
    offenders: list[str] = []

    for path in root.rglob("*.py"):
        if path == allowed_owner:
            continue
        source = path.read_text(encoding="utf-8")
        if "store_pending(" in source:
            offenders.append(str(path.relative_to(root)))

    assert offenders == [], (
        "Public writes must use canonical ingestion before persistence; "
        f"legacy pending-store callers: {offenders}"
    )
