# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""P1-6 (dedup-complete-03 prevention): CI tripwire forbidding new raw
``INSERT INTO atomic_facts`` writers.

Every fact write must go through ``storage.database.store_fact`` so the
content-idempotency invariant (v3.6.4) applies. A new unrouted INSERT is how
the original duplicate/contamination got in. This guard fails CI if a base
``atomic_facts`` INSERT appears outside the reviewed allowlist.
"""

from __future__ import annotations

import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"

# Files reviewed and permitted to contain a base-table INSERT. Each was
# verified to either BE the idempotent writer or route through it:
#   - storage/database.py        : store_fact() — the canonical idempotent writer
#   - core/fact_consolidator.py  : routed via dedup-or-insert (P0-3, v3.6.4)
#   - server/routes/data_io.py   : bulk import endpoint (separate review)
#   - storage/v2_migrator.py     : one-time v2→v3 migration
_ALLOWLIST = frozenset({
    "storage/database.py",
    "core/fact_consolidator.py",
    "server/routes/data_io.py",
    "storage/v2_migrator.py",
})

# Matches a base-table insert. \b after 'atomic_facts' means 'atomic_facts_fts'
# (the FTS5 shadow table / triggers) is NOT matched — those are not fact rows.
_INSERT_RE = re.compile(r"INSERT\s+(?:OR\s+[A-Z]+\s+)?INTO\s+atomic_facts\b", re.IGNORECASE)


def _offenders(root: pathlib.Path, allowlist) -> list[str]:
    out: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowlist:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _INSERT_RE.search(text):
            out.append(rel)
    return sorted(out)


def test_no_unrouted_atomic_facts_inserts() -> None:
    offenders = _offenders(_SRC, _ALLOWLIST)
    assert not offenders, (
        "New raw 'INSERT INTO atomic_facts' outside the reviewed allowlist. "
        "Route fact writes through storage.database.store_fact (idempotent) "
        f"or add the file to the allowlist after review: {offenders}"
    )


def test_guard_detects_a_synthetic_violation(tmp_path) -> None:
    # Prove the tripwire actually fires on an unrouted insert.
    (tmp_path / "rogue.py").write_text(
        "db.execute('INSERT INTO atomic_facts (fact_id) VALUES (1)')\n"
    )
    assert _offenders(tmp_path, frozenset()) == ["rogue.py"]


def test_guard_ignores_fts_shadow_table(tmp_path) -> None:
    # The FTS5 shadow table must NOT be flagged (it is not a fact-row write).
    (tmp_path / "fts.py").write_text(
        "c.execute('INSERT INTO atomic_facts_fts (rowid, content) VALUES (1, x)')\n"
    )
    assert _offenders(tmp_path, frozenset()) == []
