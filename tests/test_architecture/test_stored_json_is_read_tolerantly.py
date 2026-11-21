# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Ratchet: a stored JSON column is never decoded strictly.

A TEXT column that holds a BLOB of bytes which are not UTF-8 makes
``json.loads(value)`` raise ``'utf-8' codec can't decode ...``. Every reader of
the columns in ``storage/json_columns.TOLERATED_COLUMNS`` goes through
``load_json_column`` instead, so one damaged value cannot fail a save, a
recall, a listing or an export. This scan fails when a ``json.loads`` (or
``.decode``) is applied to an expression that names one of those columns.

The allowlist holds only places that cannot see a database value; each entry
says why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import superlocalmemory
from superlocalmemory.storage.json_columns import TOLERATED_COLUMNS

SRC = Path(superlocalmemory.__file__).parent
NAMES = sorted({column for _table, column in TOLERATED_COLUMNS})
PATTERN = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(NAMES) + r")(?![A-Za-z0-9_])")
ALLOWED = {
    # (file relative to the package, the decoded expression): why it cannot see a stored value
    ("core/judge_ladder.py", "r.fact_ids_json"):
        "a receipt row built in this process a line earlier; it is written, not read from the store",
}


def _is_decode(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "loads":
        return True
    if isinstance(func, ast.Name) and func.id == "loads":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "decode"


def _offenders() -> list[str]:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "storage/json_columns.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_decode(node):
                continue
            targets = list(node.args)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "decode":
                targets.append(node.func.value)
            for arg in targets:
                text = ast.unparse(arg)
                if PATTERN.search(text) and (rel, text) not in ALLOWED:
                    found.append(f"{rel}:{node.lineno}: {ast.unparse(node)[:100]}")
    return found


def test_no_stored_json_column_is_decoded_strictly() -> None:
    assert _offenders() == [], (
        "decode these with storage.json_columns.load_json_column (or load_json_column_checked "
        "when a write can follow) so a damaged value reads as empty instead of raising")


def test_the_allowlist_has_no_stale_entries() -> None:
    seen = set()
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and _is_decode(node):
                seen.update((rel, ast.unparse(a)) for a in node.args)
    assert set(ALLOWED) <= seen


def test_the_scan_sees_a_strict_read() -> None:
    tree = ast.parse('import json\nx = json.loads(row["entities_json"] or "[]")\n')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _is_decode(n)]
    assert calls and PATTERN.search(ast.unparse(calls[0].args[0]))
    assert not PATTERN.search("raw_metadata_json")
