# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Deterministic facets pulled from saved text: links, paths, dates, versions, code names, tags."""

from .extractors import (
    EXTRACTED_KEY,
    Extracted,
    add_extracted,
    extract_deterministic,
)

__all__ = ["EXTRACTED_KEY", "Extracted", "add_extracted", "extract_deterministic"]
