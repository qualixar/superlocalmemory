# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The preview a person sees before a folder is connected. Reads the folder; writes nothing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from superlocalmemory.sources import ingest, walk
from superlocalmemory.sources.ignore import IgnoreRules, kind_of
from superlocalmemory.sources.safe_read import read_bounded

TEXT_SECONDS = 0.005
PDF_PAGE_SECONDS = 1.0
IMAGE_SECONDS = 1.0
_PDF_BYTES_PER_PAGE = 75_000
ESTIMATE_NOTE = "Estimate only: about 5 ms per note, about 1 s per PDF page and per picture."


@dataclass(frozen=True)
class SourcePreview:
    source_id: str
    root: str
    kind: str
    files_by_type: dict[str, int]
    skipped_by_rule: dict[str, int]
    est_bytes: int
    quarantined_count: int
    est_seconds: float
    estimate_note: str = ESTIMATE_NOTE
    capped: bool = False
    warnings: list[str] = field(default_factory=list)


def _estimate(entries: list[walk.Entry]) -> float:
    total = 0.0
    for e in entries:
        kind = kind_of(e.relpath)
        if kind == "text":
            total += TEXT_SECONDS
        elif kind == "pdf":
            total += max(1, e.size // _PDF_BYTES_PER_PAGE) * PDF_PAGE_SECONDS
        else:
            total += IMAGE_SECONDS
    return round(total, 2)


def _quarantined(root: Path, entries: list[walk.Entry]) -> int:
    count = 0
    for e in entries:
        if kind_of(e.relpath) != "text" or e.placeholder:
            continue
        try:
            head = read_bounded(root / e.relpath, ingest.SCREEN_BYTES, e.file_id)
        except OSError:
            continue
        count += 1 if ingest.screen(head) else 0
    return count


def build_preview(source_id: str, root: Path, kind: str, include_types: tuple[str, ...]) -> SourcePreview:
    result = walk.walk_tree(root, IgnoreRules(root, include_types))
    by_type: dict[str, int] = {}
    for e in result.entries:
        ext = Path(e.relpath).suffix.lower()
        by_type[ext] = by_type.get(ext, 0) + 1
    warnings = []
    if result.capped:
        warnings.append(f"This folder is over the file limit; only the first {walk.MAX_FILES:,} "
                        "files will be indexed.")
    return SourcePreview(
        source_id=source_id, root=str(root), kind=kind, files_by_type=by_type,
        skipped_by_rule=dict(result.skipped), est_bytes=sum(e.size for e in result.entries),
        quarantined_count=_quarantined(root, result.entries), est_seconds=_estimate(result.entries),
        capped=result.capped, warnings=warnings)
