# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Reading a JSON value from a TEXT column that may hold bytes.

SQLite stores whatever a writer binds: a ``bytes`` value lands in a TEXT column
as a BLOB and comes back to Python as ``bytes``. ``json.loads(bytes)`` decodes
it strictly, so a single such value that is not UTF-8 (in ``entities_json``,
``langevin_position``, a scene's ``entity_ids_json`` ...) raised
``'utf-8' codec can't decode byte ...`` in every enrichment that read the row,
and each of those saves was dead-lettered after its automatic attempts.

:func:`load_json_column` reads a BLOB that holds UTF-8 JSON as usual. One that
is not text has no recoverable value: it is read as the column's empty value,
reported once per column, and the row is left as it is (nothing is rewritten
or invented). Text values behave exactly as ``json.loads`` always did.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_REPORTED: set[str] = set()
_LOCK = threading.Lock()


def _report(column: str, row: object) -> None:
    with _LOCK:
        if column in _REPORTED:
            return
        _REPORTED.add(column)
    logger.warning(
        "%s holds bytes that are not UTF-8 text (row %s); read as empty so saves keep "
        "working. Find such rows with: SELECT rowid FROM <table> WHERE typeof(<column>)"
        "='blob'", column, row if row is not None else "?",
    )


def load_json_column(raw: Any, default: Any, *, column: str, row: object = None) -> Any:
    """The JSON value of ``raw``, or ``default`` when it is absent or not text."""
    if raw is None or raw == "" or raw == b"":
        return default
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            _report(column, row)
            return default
    return json.loads(raw)


__all__ = ["load_json_column"]
