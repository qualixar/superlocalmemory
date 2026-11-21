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

The scan also follows one step of data flow: inside a function, a local name
assigned from an expression that names a tolerated column (``raw = d.get("x_json")
or "[]"``, ``for raw in row["x_json"]``-style tuple targets are not followed) is
tainted, and decoding a tainted name (itself, or in ``x or default``) is flagged
unless the call sits in a ``try`` whose handler catches ``ValueError``,
``UnicodeDecodeError``, ``Exception`` or everything (that skips the damaged value,
which is tolerant). ``json.JSONDecodeError`` alone does not count: it is a
``ValueError`` but a ``UnicodeDecodeError`` is not a ``JSONDecodeError``. Limits: only plain
``Name = <expr>`` / ``Name: T = <expr>`` assignments inside the same function are
followed; a value passed in as a parameter, returned from a helper, or copied
through a second name is not. The ratchet narrows the gap, it does not close it.

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
    ("learning/reward.py", "_facts_json"):
        "pending_outcomes.fact_ids_json: a table whose rows this scan does not tolerate (not a fact column)",
    ("storage/_restore_delta.py", "raw"):
        "a row of a restore delta file (JSON lines), not a database value",
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


def _decode_targets(node: ast.Call) -> list[ast.expr]:
    targets = list(node.args)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "decode":
        targets.append(node.func.value)
    return targets


def _tainted_names(scope: ast.AST) -> set[str]:
    """Names assigned, in this function, from an expression that names a tolerated column."""
    tainted: set[str] = set()
    for node in ast.walk(scope):
        value = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value, names = node.value, [node.target.id]
        if value is not None and names and PATTERN.search(ast.unparse(value)):
            tainted.update(names)
    return tainted


_SAFE = {"ValueError", "UnicodeDecodeError", "Exception", "BaseException"}


def _handler_is_safe(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(ast.unparse(t).split(".")[-1] in _SAFE for t in types)


def _guarded(tree: ast.AST) -> set[int]:
    """ids of the calls inside a ``try`` body whose handlers catch a damaged-bytes decode error."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_handler_is_safe(h) for h in node.handlers):
            for stmt in node.body:
                out.update(id(n) for n in ast.walk(stmt) if isinstance(n, ast.Call))
    return out


def _direct_names(arg: ast.expr) -> set[str]:
    """The names an argument IS (``raw``, ``raw or "[]"``), not names used inside a subscript."""
    if isinstance(arg, ast.Name):
        return {arg.id}
    if isinstance(arg, ast.BoolOp):
        return set().union(*(_direct_names(v) for v in arg.values))
    if isinstance(arg, ast.IfExp):
        return _direct_names(arg.body) | _direct_names(arg.orelse)
    return set()


def offenders_in(source: str, rel: str) -> list[str]:
    found: list[str] = []
    tree = ast.parse(source)
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    taint_of: dict[int, set[str]] = {}
    for scope in scopes:
        names = _tainted_names(scope)
        for node in ast.walk(scope):
            if isinstance(node, ast.Call):
                taint_of.setdefault(id(node), set()).update(names)
    guarded = _guarded(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_decode(node):
            continue
        tainted = taint_of.get(id(node), set())
        for arg in _decode_targets(node):
            text = ast.unparse(arg)
            flows = bool(_direct_names(arg) & tainted) and id(node) not in guarded
            if (PATTERN.search(text) or flows) and (rel, text) not in ALLOWED:
                found.append(f"{rel}:{node.lineno}: {ast.unparse(node)[:100]}")
    return found


def _offenders() -> list[str]:
    found: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "storage/json_columns.py":
            continue
        found.extend(offenders_in(path.read_text(encoding="utf-8"), rel))
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


def test_the_scan_follows_a_value_through_a_local_name() -> None:
    old = (
        "import json\n"
        "def f(d):\n"
        "    raw = d.get('canonical_entities_json') or '[]'\n"
        "    try:\n"
        "        return json.loads(raw)\n"
        "    except (json.JSONDecodeError, TypeError):\n"
        "        return []\n")
    assert offenders_in(old, "x.py")
    unrelated = "import json\ndef f(d):\n    raw = d.get('title')\n    return json.loads(raw)\n"
    assert offenders_in(unrelated, "x.py") == []


# The two readers cognitive_consolidator.py had before the fix (4.2.33), verbatim in
# shape: a local copied from a tolerated column, decoded under JSONDecodeError only.
_OLD_CONSOLIDATOR = """
import json


class CognitiveConsolidator:
    def _candidates(self, rows):
        candidates = []
        for row in rows or []:
            d = dict(row)
            raw_entities = d.get("canonical_entities_json") or "[]"
            try:
                d["canonical_entities"] = json.loads(raw_entities)
            except (json.JSONDecodeError, TypeError):
                d["canonical_entities"] = []
            candidates.append(d)
        return candidates

    def _facts(self, rows):
        facts = []
        for r in rows or []:
            d = dict(r)
            raw = d.get("canonical_entities_json") or "[]"
            try:
                d["canonical_entities"] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d["canonical_entities"] = []
            facts.append(d)
        return facts
"""


def test_the_old_consolidator_would_have_been_caught() -> None:
    assert len(offenders_in(_OLD_CONSOLIDATOR, "encoding/cognitive_consolidator.py")) == 2


def test_a_decode_that_catches_value_error_is_tolerant_but_json_decode_error_alone_is_not() -> None:
    tmpl = ("import json\ndef f(d):\n    raw = d.get('entities_json')\n    try:\n"
            "        return json.loads(raw)\n    except {exc}:\n        return []\n")
    assert offenders_in(tmpl.format(exc="(ValueError, TypeError)"), "x.py") == []
    assert offenders_in(tmpl.format(exc="(json.JSONDecodeError, TypeError)"), "x.py")
