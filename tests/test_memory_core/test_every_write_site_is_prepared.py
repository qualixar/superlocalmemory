# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Every place that builds a durable write request must prepare its text.

A new ``IngestionRequest(...)`` or ``RememberRequest(...)`` site fails here until
its author either calls the shared save step or records why it is exempt. The
scan resolves import aliases, matches attribute calls, and flags any other use
of the request classes as a value (``partial``, ``replace``) or a rebuild from a
stored payload.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
_CLASSES = {"IngestionRequest", "RememberRequest"}
_REBUILD = {"from_payload", "from_dict"}
_PREPARERS = {"prepare_for_save", "prepare_user_text", "_prepared_prebuilt"}

_REPLAY = "exempt: replay of an already admitted request"
_STORED = "exempt: repair of text that is already stored"
_ROW = "exempt: reads a stored journal row"

_SITES: dict[tuple[str, str], str] = {
    ("server/unified_daemon.py", "ObserveBuffer.enqueue"): "prepared",
    ("server/unified_daemon.py", "_register_daemon_routes.remember"): "prepared",
    ("server/routes/ingest.py", "ingest"): "prepared",
    ("server/routes/data_io.py", "import_memories"): "prepared",
    ("daemon/materializer.py", "legacy_item"): "prepared",
    ("core/engine_ingestion.py", "canonical_store"): "prepared",
    ("core/engine_ingestion.py", "canonical_store_fact"): "prepared",
    ("core/remember_runtime.py", "CanonicalRememberRuntime._handle_admission"): _REPLAY,
    ("storage/own_fact_repair.py", "_queue_enrichment"): _STORED,
    ("storage/admission_journal.py", "AdmissionJournal.request_for"): _ROW,
}


@dataclass
class Finding:
    kind: str  # construct | reference | rebuild | replace
    qualname: str
    line: int
    call: ast.Call | None = None
    func: ast.AST | None = None
    notes: list[str] = field(default_factory=list)


def _aliases(tree: ast.AST) -> set[str]:
    names = set(_CLASSES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                if item.name in _CLASSES:
                    names.add(item.asname or item.name)
    return names


def _type_use_ids(tree: ast.AST) -> set[int]:
    """Node ids that are type positions: annotations, isinstance, subscripts."""
    ids: set[int] = set()

    def mark(node: ast.AST | None) -> None:
        if node is not None:
            ids.update(id(n) for n in ast.walk(node))

    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            mark(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            mark(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark(node.returns)
        elif isinstance(node, ast.Subscript):
            mark(node.slice)
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                mark(base)
        elif (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"isinstance", "issubclass"}
        ):
            for arg in node.args[1:]:
                mark(arg)
    return ids


def _is_class_ref(node: ast.AST, names: set[str]) -> bool:
    return (
        (isinstance(node, ast.Name) and node.id in names)
        or (isinstance(node, ast.Attribute) and node.attr in _CLASSES)
    )


def _is_replace_with_content(call: ast.Call) -> bool:
    fn = call.func
    named_replace = (isinstance(fn, ast.Name) and fn.id == "replace") or (
        isinstance(fn, ast.Attribute) and fn.attr == "replace"
        and isinstance(fn.value, ast.Name) and fn.value.id == "dataclasses"
    )
    return named_replace and any(kw.arg == "content" for kw in call.keywords)


def _classify_call(call: ast.Call, names: set[str]) -> str | None:
    fn = call.func
    if _is_class_ref(fn, names):
        return "construct"
    if isinstance(fn, ast.Attribute) and fn.attr in _REBUILD and _is_class_ref(fn.value, names):
        return "rebuild"
    return "replace" if _is_replace_with_content(call) else None


class _Scanner:
    def __init__(self, tree: ast.AST) -> None:
        self.names = _aliases(tree)
        self.type_ids = _type_use_ids(tree)
        self.findings: list[Finding] = []

    def visit(self, node: ast.AST, stack: list[str], func: ast.AST | None) -> None:
        qual = ".".join(stack) or "<module>"
        for child in ast.iter_child_nodes(node):
            sub, inner = stack, func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                sub = stack + [child.name]
                inner = func if isinstance(child, ast.ClassDef) else child
            self._note(child, qual, func)
            self.visit(child, sub, inner)

    def _note(self, child: ast.AST, qual: str, func: ast.AST | None) -> None:
        if isinstance(child, ast.Call):
            kind = _classify_call(child, self.names)
            if kind:
                self.findings.append(Finding(kind, qual, child.lineno, child, func))
        elif (
            isinstance(child, (ast.Name, ast.Attribute))
            and isinstance(getattr(child, "ctx", None), ast.Load)
            and _is_class_ref(child, self.names)
            and id(child) not in self.type_ids
        ):
            self.findings.append(Finding("reference", qual, child.lineno, None, func))


def scan_source(source: str) -> list[Finding]:
    tree = ast.parse(source)
    scanner = _Scanner(tree)
    scanner.visit(tree, [], None)
    # A call's func node is also seen as a reference; drop those.
    call_lines = {f.line for f in scanner.findings if f.kind in {"construct", "rebuild"}}
    return [
        f for f in scanner.findings
        if f.kind != "reference" or f.line not in call_lines
    ]


def _last_assigned_value(func: ast.AST | None, name: str, before: int) -> ast.AST | None:
    """The value of the last assignment to ``name`` above line ``before``."""
    best: ast.Assign | None = None
    for node in ast.walk(func) if func is not None else []:
        if not isinstance(node, ast.Assign) or node.lineno >= before:
            continue
        targets = [
            t for target in node.targets
            for t in (target.elts if isinstance(target, ast.Tuple) else [target])
        ]
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            if best is None or node.lineno > best.lineno:
                best = node
    return best.value if best is not None else None


def _from_preparer(value: ast.AST | None, func: ast.AST | None, before: int) -> bool:
    """True when ``value`` is, or is taken off, a prepare-step result."""
    if isinstance(value, ast.Call):
        fn = value.func
        return (isinstance(fn, ast.Name) and fn.id in _PREPARERS) or (
            isinstance(fn, ast.Attribute) and fn.attr in _PREPARERS
        )
    if isinstance(value, ast.Attribute) and value.attr == "text":
        return _from_preparer(value.value, func, before)
    if isinstance(value, ast.Name):
        return _from_preparer(_last_assigned_value(func, value.id, before), func, before)
    return False


def content_is_prepared(finding: Finding) -> bool:
    call = finding.call
    if call is None:
        return False
    for kw in call.keywords:
        if kw.arg == "content":
            return isinstance(kw.value, (ast.Attribute, ast.Name)) and _from_preparer(
                kw.value, finding.func, call.lineno,
            )
    return False


def violations(rel: str, source: str, sites: dict[tuple[str, str], str]) -> list[str]:
    problems: list[str] = []
    for f in scan_source(source):
        kind = sites.get((rel, f.qualname))
        where = f"{rel}:{f.line} ({f.qualname}, {f.kind})"
        if kind is None:
            problems.append(
                f"{where} uses a write request: prepare the text with "
                "prepare_for_save/prepare_user_text, then add the site to _SITES"
            )
        elif kind == "prepared":
            if f.kind != "construct" or not content_is_prepared(f):
                problems.append(
                    f"{where} must pass content= the prepared text (.text of "
                    "prepare_for_save/prepare_user_text)"
                )
    return problems


def test_every_write_site_is_mapped_and_prepared() -> None:
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not any(c in source for c in _CLASSES):
            continue
        rel = path.relative_to(_SRC).as_posix()
        problems += violations(rel, source, _SITES)
        seen |= {(rel, f.qualname) for f in scan_source(source)}
    problems += [f"stale site {s}" for s in _SITES if s not in seen]
    assert not problems, "\n".join(problems)


# --- the guard itself ---------------------------------------------------------

_SITE = {("m.py", "go"): "prepared"}


def _bad(source: str) -> list[str]:
    return violations("m.py", source, {})


def test_guard_catches_an_aliased_import() -> None:
    src = "from x import IngestionRequest as R\ndef go(t):\n    return R(content=t)\n"
    assert _bad(src)


def test_guard_catches_an_attribute_call() -> None:
    src = "import x\ndef go(t):\n    return x.RememberRequest(content=t)\n"
    assert _bad(src)


def test_guard_catches_partial_and_replace() -> None:
    partial = (
        "from functools import partial\nfrom x import IngestionRequest\n"
        "def go():\n    return partial(IngestionRequest, source_type='a')\n"
    )
    repl = (
        "from dataclasses import replace\nfrom x import IngestionRequest\n"
        "def go(r):\n    return replace(r, content='raw')\n"
    )
    assert _bad(partial) and _bad(repl)


def test_guard_catches_a_rebuild_from_a_payload_and_module_level_use() -> None:
    rebuild = "from x import RememberRequest\ndef go(p):\n    return RememberRequest.from_payload(p)\n"
    module = "from x import IngestionRequest\nREQ = IngestionRequest(content='raw')\n"
    assert _bad(rebuild) and _bad(module)


def test_guard_ignores_annotations_and_isinstance() -> None:
    src = (
        "from x import IngestionRequest\n"
        "def go(r: IngestionRequest) -> IngestionRequest | None:\n"
        "    return r if isinstance(r, IngestionRequest) else None\n"
    )
    assert _bad(src) == []


def test_guard_requires_content_to_come_from_the_prepare_step() -> None:
    raw = "from x import IngestionRequest\ndef go(t):\n    return IngestionRequest(content=t)\n"
    via_name = (
        "from x import IngestionRequest\n"
        "def go(t):\n    p = prepare_user_text(c, t)\n    return IngestionRequest(content=p.text)\n"
    )
    via_assign = (
        "from x import IngestionRequest\n"
        "def go(t):\n    t = prepare_for_save(t).text\n    return IngestionRequest(content=t)\n"
    )
    substring_only = (
        "from x import IngestionRequest\n"
        "def go(t):\n    # prepare_for_save\n    return IngestionRequest(content=t)\n"
    )
    assert violations("m.py", raw, _SITE)
    assert violations("m.py", via_name, _SITE) == []
    assert violations("m.py", via_assign, _SITE) == []
    assert violations("m.py", substring_only, _SITE)


def test_guard_rejects_text_taken_off_a_non_preparer() -> None:
    src = (
        "from x import IngestionRequest\n"
        "def go(msg):\n    return IngestionRequest(content=msg.text)\n"
    )
    assert violations("m.py", src, _SITE)


def test_guard_rejects_a_raw_reassignment_after_the_prepare_step() -> None:
    src = (
        "from x import IngestionRequest\n"
        "def go(t, raw):\n"
        "    t = prepare_for_save(t).text\n"
        "    t = raw\n"
        "    return IngestionRequest(content=t)\n"
    )
    assert violations("m.py", src, _SITE)


def test_guard_accepts_the_last_assignment_being_a_preparer() -> None:
    src = (
        "from x import IngestionRequest\n"
        "def go(t, raw):\n"
        "    t = raw\n"
        "    p = prepare_user_text(c, t)\n"
        "    t = p.text\n"
        "    return IngestionRequest(content=t)\n"
    )
    assert violations("m.py", src, _SITE) == []
