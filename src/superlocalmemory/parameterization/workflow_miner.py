# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Workflow miner re-export for the parameterization namespace.

The canonical implementation lives in
:mod:`superlocalmemory.learning.workflows`. This thin module re-exports
``WorkflowMiner`` so that code in the parameterization subsystem can import
from the same namespace as the pattern extractor's dependency declaration.
"""

from __future__ import annotations

from superlocalmemory.learning.workflows import WorkflowMiner

__all__ = ["WorkflowMiner"]
