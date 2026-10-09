# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Read-only folder sources mirrored into memory. Off until the user confirms a folder."""

from __future__ import annotations

from superlocalmemory.sources.api import (
    HintsNotAvailable,
    SourceInfo,
    SourceRefused,
    add_source,
    confirm_source,
    hint,
    list_sources,
    release_file,
    remove_source,
    rescan,
    source_report,
)
from superlocalmemory.sources.host import SourceHost, configure
from superlocalmemory.sources.ignore import DEFAULT_TYPES
from superlocalmemory.sources.preview import SourcePreview
from superlocalmemory.sources.report import SourceReport

__all__ = [
    "DEFAULT_TYPES", "HintsNotAvailable", "SourceHost", "SourceInfo", "SourcePreview", "SourceRefused",
    "SourceReport", "add_source", "configure", "confirm_source", "hint", "list_sources",
    "release_file", "remove_source", "rescan", "source_report",
]
