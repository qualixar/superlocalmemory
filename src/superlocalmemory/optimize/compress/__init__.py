# compress/__init__.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""SLM v3.6 Optimize — Compression module.

Public surface:
    CompressRouter  — implements CompressHook Protocol from lifecycle.py
    CCRStore        — CCR original storage and retrieval
    CacheAligner    — volatile-token detection for system prompt prefix stability
"""

from superlocalmemory.optimize.compress.router import CompressRouter, CompressTextResult
from superlocalmemory.optimize.compress.ccr import CCRStore
from superlocalmemory.optimize.compress.align import CacheAligner, AlignResult

__all__ = ["CompressRouter", "CompressTextResult", "CCRStore", "CacheAligner", "AlignResult"]
