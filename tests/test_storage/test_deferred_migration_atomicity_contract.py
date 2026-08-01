# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com
"""Deferred migrations must be safe under best-effort DDL application.

The runner applies a bare (non-self-wrapping) DDL script in autocommit mode, so
each statement commits independently and a failing trailing statement — e.g. a
performance index over a table that may not exist yet — does not roll back the
essential DDL before it. A migration relies on this only if its ``verify()``
checks the essential schema, never a statement that could be the one that fails.

This guard codifies that contract: a multi-statement bare-DDL deferred migration
must either self-wrap a transaction, ship a custom ``apply()``, or keep its
``verify()`` free of index checks. It exists so a future migration cannot
silently reintroduce a partial-apply-marked-complete hazard.
"""
from __future__ import annotations

import inspect
import re

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS

_OPENS_TXN = re.compile(r"(?im)^\s*BEGIN(\s+(IMMEDIATE|DEFERRED|EXCLUSIVE|TRANSACTION))?\s*;")
_INDEX_REF = re.compile(r"idx_|INDEX|\bindex\b")


def _statements(ddl: str) -> list[str]:
    return [s for s in (part.strip() for part in ddl.split(";")) if s and not s.startswith("--")]


def test_deferred_migrations_are_atomicity_safe() -> None:
    violations = []
    for migration in DEFERRED_MIGRATIONS:
        ddl = migration.ddl or ""
        module = _MODULES.get(migration.name)
        has_apply = module is not None and callable(getattr(module, "apply", None))
        self_wraps = bool(_OPENS_TXN.search(ddl))
        multi_statement = len(_statements(ddl)) > 1

        # Single-statement, self-wrapping, or custom-apply migrations are safe
        # regardless of what verify() inspects.
        if not multi_statement or self_wraps or has_apply:
            continue

        verify_fn = getattr(module, "verify", None) if module is not None else None
        verify_src = inspect.getsource(verify_fn) if verify_fn is not None else ""
        if _INDEX_REF.search(verify_src):
            violations.append(migration.name)

    assert not violations, (
        "multi-statement bare-DDL deferred migrations whose verify() checks an "
        f"index are unsafe under best-effort application: {violations}. Give the "
        "migration a self-wrapped transaction or a custom apply(), or make its "
        "verify() check only essential (non-index) schema."
    )
