"""Cross-project preference aggregation for the parameterization subsystem.

The aggregator implementation lives in
:mod:`superlocalmemory.learning.cross_project`. It is re-exported here so the
parameterization namespace — where the pattern extractor consumes it — resolves
to the same class instead of a missing module.
"""
from __future__ import annotations

from superlocalmemory.learning.cross_project import CrossProjectAggregator

__all__ = ["CrossProjectAggregator"]
