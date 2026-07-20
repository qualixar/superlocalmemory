"""Full-source guard against a second SuperLocalMemory data-root authority."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO / "src" / "superlocalmemory"
_ROOT_AUTHORITY = "src/superlocalmemory/infra/data_root.py"

# Entries here must be executable references to an SLM path that do not own or
# consume runtime state. State debt is never allowlisted. Tuple values are
# source line numbers and every entry requires a reviewable reason.
_INTENTIONAL_NON_STATE_ALLOWLIST: dict[
    str,
    dict[int, str],
] = {
    "src/superlocalmemory/cli/post_install.py": {
        55: "Display-only fallback for reporting the database location.",
    },
}


@dataclass(frozen=True, order=True)
class _Violation:
    path: str
    line: int
    column: int
    expression: str


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _assignment_aliases(tree: ast.AST) -> dict[str, ast.AST]:
    """Return only unambiguous aliases; repeated names are scope-dependent."""
    collected: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    collected.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                collected.setdefault(node.target.id, []).append(node.value)
    return {
        name: values[0]
        for name, values in collected.items()
        if len(values) == 1
    }


def _expanded_nodes(
    node: ast.AST,
    aliases: dict[str, ast.AST],
    expanding: frozenset[str] = frozenset(),
):
    yield node
    if isinstance(node, ast.Name) and node.id in aliases and node.id not in expanding:
        yield from _expanded_nodes(
            aliases[node.id],
            aliases,
            expanding | {node.id},
        )
        return
    for child in ast.iter_child_nodes(node):
        yield from _expanded_nodes(child, aliases, expanding)


def _string_values(
    node: ast.AST,
    aliases: dict[str, ast.AST],
) -> tuple[str, ...]:
    return tuple(
        child.value
        for child in _expanded_nodes(node, aliases)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    )


def _contains_slm_segment(
    node: ast.AST,
    aliases: dict[str, ast.AST],
) -> bool:
    for value in _string_values(node, aliases):
        normalized = value.replace("\\", "/")
        if ".superlocalmemory" in normalized.split("/"):
            return True
    return False


def _contains_home_anchor(
    node: ast.AST,
    aliases: dict[str, ast.AST],
) -> bool:
    for child in _expanded_nodes(node, aliases):
        if isinstance(child, ast.Call):
            name = _call_name(child.func)
            if name in {"Path.home", "pathlib.Path.home", "_Path.home"}:
                return True
            if name.endswith(("expanduser", "getenv", "environ.get")):
                values = _string_values(child, aliases)
                if any(value == "HOME" or value.startswith("~") for value in values):
                    return True
        if isinstance(child, ast.Subscript):
            if _call_name(child.value).endswith("environ"):
                if "HOME" in _string_values(child.slice, aliases):
                    return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            normalized = child.value.replace("\\", "/")
            if normalized == "~" or normalized.startswith("~/"):
                return True
    return False


def _constructs_home_slm_path(
    node: ast.AST,
    aliases: dict[str, ast.AST],
) -> bool:
    return _contains_slm_segment(node, aliases) and _contains_home_anchor(
        node,
        aliases,
    )


class _PathConstructionVisitor(ast.NodeVisitor):
    def __init__(
        self,
        relative_path: str,
        aliases: dict[str, ast.AST],
    ) -> None:
        self.relative_path = relative_path
        self.aliases = aliases
        self.violations: list[_Violation] = []
        self._seen: set[tuple[int, int]] = set()

    def _inspect(self, node: ast.AST) -> bool:
        if not _constructs_home_slm_path(node, self.aliases):
            return False
        location = (node.lineno, node.col_offset)
        if location not in self._seen:
            self._seen.add(location)
            self.violations.append(
                _Violation(
                    path=self.relative_path,
                    line=node.lineno,
                    column=node.col_offset,
                    expression=ast.unparse(node),
                )
            )
        return True

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if not self._inspect(node.value):
            self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is None or not self._inspect(node.value):
            self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        if node.value is None or not self._inspect(node.value):
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not self._inspect(node):
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if not self._inspect(node):
            self.generic_visit(node)


def _source_violations(
    relative_path: str,
    source: str,
    *,
    apply_allowlist: bool = True,
) -> list[_Violation]:
    tree = ast.parse(source, filename=relative_path)
    visitor = _PathConstructionVisitor(relative_path, _assignment_aliases(tree))
    visitor.visit(tree)
    if not apply_allowlist:
        return visitor.violations
    allowlisted_lines = _INTENTIONAL_NON_STATE_ALLOWLIST.get(relative_path, {})
    return [
        violation
        for violation in visitor.violations
        if violation.line not in allowlisted_lines
    ]


def _repository_violations() -> list[_Violation]:
    violations: list[_Violation] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(_REPO).as_posix()
        if relative == _ROOT_AUTHORITY:
            continue
        violations.extend(
            _source_violations(relative, path.read_text(encoding="utf-8"))
        )
    return sorted(violations)


def test_ast_detector_catches_supported_home_root_constructions() -> None:
    source = """
from pathlib import Path
import os

one = Path.home() / ".superlocalmemory" / "memory.db"
two = os.path.join(os.path.expanduser("~"), ".superlocalmemory", "cache.db")
three = Path("~/.superlocalmemory/config.json").expanduser()
SLM_DIR = ".superlocalmemory"
home = Path.home()
four = home / SLM_DIR / "graph.db"
"""

    violations = _source_violations("synthetic.py", source)

    assert {violation.line for violation in violations} == {5, 6, 7, 10}


def test_intentional_non_state_allowlist_is_reviewable_and_not_stale() -> None:
    stale: list[str] = []
    for relative_path, allowed in _INTENTIONAL_NON_STATE_ALLOWLIST.items():
        source = (_REPO / relative_path).read_text(encoding="utf-8")
        detected_lines = {
            violation.line
            for violation in _source_violations(
                relative_path,
                source,
                apply_allowlist=False,
            )
        }
        for line, reason in allowed.items():
            if not reason.strip() or line not in detected_lines:
                stale.append(f"{relative_path}:{line}: {reason!r}")

    assert not stale, "stale or unexplained path allowlist entries:\n" + "\n".join(stale)


def test_runtime_source_has_no_second_data_root_authority() -> None:
    violations = _repository_violations()
    rendered = "\n".join(
        f"{item.path}:{item.line}:{item.column + 1}: {item.expression}"
        for item in violations
    )
    assert not violations, "split-root constructions remain:\n" + rendered
