# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""What a source skipped, quarantined, left in the cloud or failed on."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from superlocalmemory.sources.store import SourceStore
from superlocalmemory.sources.watcher import is_watching


@dataclass(frozen=True)
class SourceReport:
    source_id: str
    state: str
    counts: dict[str, int] = field(default_factory=dict)
    skipped_by_rule: dict[str, int] = field(default_factory=dict)
    quarantined: list[dict[str, str]] = field(default_factory=list)
    cloud_only: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    last_scan_at: str | None = None
    paused_reason: str | None = None
    capped: bool = False
    offline_reason: str | None = None
    #: 1 while file changes are noticed at once, 0 when only the 15-minute scan looks.
    watch: int = 0


def build_report(store: SourceStore, source: dict[str, Any]) -> SourceReport:
    sid = source["source_id"]
    try:
        stats = json.loads(source.get("last_scan_stats_json") or "{}")
    except ValueError:
        stats = {}
    rows = store.files(sid, ["quarantined", "cloud_placeholder", "error", "skipped"])
    pick = lambda state: [r for r in rows if r["state"] == state]  # noqa: E731
    return SourceReport(
        source_id=sid, state=source["state"], counts=store.counts(sid),
        skipped_by_rule=dict(stats.get("skipped") or {}),
        quarantined=[{"relpath": r["relpath"], "reason": r["reason"] or ""} for r in pick("quarantined")],
        cloud_only=[r["relpath"] for r in pick("cloud_placeholder")],
        errors=[{"relpath": r["relpath"], "reason": r["reason"] or ""} for r in pick("error")],
        last_scan_at=source.get("last_scan_at"), paused_reason=stats.get("paused_reason"),
        capped=bool(stats.get("capped")),
        offline_reason=(stats.get("offline_reason") or None) if source["state"] == "offline" else None,
        watch=1 if is_watching(sid) else 0)
