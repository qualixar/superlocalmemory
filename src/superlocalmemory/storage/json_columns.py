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
is not text has no recoverable value: it is read as the column's empty value
and reported once per column. Reading never changes the row. A writer that
later replaces the value must not build the new one from that empty read: it
asks :func:`is_damaged` (or uses :func:`load_json_column_checked`), rebuilds
the value from its source of truth, and keeps the original bytes (a version
row for entity profiles and scenes, the change ledger for fact columns).
Text values behave exactly as ``json.loads`` always did.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: The JSON text columns whose readers use :func:`load_json_column`, as
#: ``(table, column)``. ``slm doctor`` counts binary values in exactly these.
TOLERATED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("atomic_facts", "entities_json"),
    ("atomic_facts", "canonical_entities_json"),
    ("atomic_facts", "source_turn_ids_json"),
    ("atomic_facts", "langevin_position"),
    ("atomic_facts", "shared_with"),
    ("memory_scenes", "fact_ids_json"),
    ("memory_scenes", "entity_ids_json"),
    ("entity_profiles", "fact_ids_json"),
    ("memories", "metadata_json"),
    ("memories", "shared_with"),
)

_REPORTED: set[str] = set()
_LOCK = threading.Lock()


def _report(column: str, row: object) -> None:
    with _LOCK:
        if column in _REPORTED:
            return
        _REPORTED.add(column)
    logger.warning(
        "%s holds bytes that are not UTF-8 text (row %s); read as empty so saves keep "
        "working. `slm doctor` counts such values per column", column,
        row if row is not None else "?",
    )


def is_damaged(raw: Any) -> bool:
    """True for a stored value that is bytes and not UTF-8 text (so it holds no JSON)."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        return False
    try:
        bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def text_value(raw: Any) -> Any:
    """``raw`` as JSON-serializable text: bytes that are text become ``str``, bytes that are not become None.

    For a column that is returned as it is stored (a response, an export row),
    where a BLOB would fail the serializer.
    """
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            return bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            return None
    return raw


def load_json_column_checked(raw: Any, default: Any, *, column: str,
                             row: object = None) -> tuple[Any, bool]:
    """``(value, damaged)``: like :func:`load_json_column`, and whether the value was damaged.

    A writer that rebuilds a value reads it through this: ``damaged`` means the
    returned ``default`` stands for bytes that were lost, not for "nothing was
    stored", so the writer rebuilds from the source instead of from empty.
    """
    if is_damaged(raw):
        _report(column, row)
        return default, True
    return load_json_column(raw, default, column=column, row=row), False


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


__all__ = ["TOLERATED_COLUMNS", "is_damaged", "load_json_column", "load_json_column_checked", "text_value"]
