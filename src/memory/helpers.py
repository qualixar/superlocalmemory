#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Helper utilities for MemoryStoreV2.

This module contains standalone utility functions extracted from memory_store_v2.py
to reduce file size and improve maintainability.
"""


def format_content(content: str, full: bool = False, threshold: int = 5000, preview_len: int = 2000) -> str:
    """
    Smart content formatting with optional truncation.

    Args:
        content: Content to format
        full: If True, always show full content
        threshold: Max length before truncation (default 5000)
        preview_len: Preview length when truncating (default 2000)

    Returns:
        Formatted content string
    """
    if full or len(content) < threshold:
        return content
    else:
        return f"{content[:preview_len]}..."
