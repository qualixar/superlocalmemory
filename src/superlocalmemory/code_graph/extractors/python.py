# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""Python-specific AST extractor using tree-sitter S-expression queries.

Extracts: function_definition, class_definition, import_statement,
import_from_statement, call. Detects test functions and docstrings.

tree-sitter imports are lazy (HR-07) — only imported when this module
is actually used at parse time, never at package import time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from superlocalmemory.code_graph.extractors import BaseExtractor
from superlocalmemory.code_graph.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)
from superlocalmemory.storage.models import _new_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S-expression queries (compiled once per class, not per instance)
# ---------------------------------------------------------------------------

FUNC_QUERY_SRC = (
    "(function_definition"
    " name: (identifier) @func.name"
    " parameters: (parameters) @func.params"
    ") @func.def"
)

CLASS_QUERY_SRC = (
    "(class_definition"
    " name: (identifier) @class.name"
    ") @class.def"
)

IMPORT_QUERY_SRC = (
    "(import_statement) @import.stmt"
)

IMPORT_FROM_QUERY_SRC = (
    "(import_from_statement) @import.from_stmt"
)

CALL_QUERY_SRC = (
    "(call"
    " function: (identifier) @call.name"
    ") @call.expr"
)

CALL_METHOD_QUERY_SRC = (
    "(call"
    " function: (attribute"
    "  attribute: (identifier) @call.method)"
    ") @call.method_expr"
)


def _get_ts_modules() -> tuple[Any, Any, Any]:
    """Lazy import of tree-sitter modules."""
    try:
        import tree_sitter
        from tree_sitter_language_pack import get_language
    except ImportError as exc:
        raise ImportError(
            "tree-sitter and tree-sitter-language-pack are required for "
            "code graph parsing. Install with: "
            "pip install 'superlocalmemory[code-graph]'"
        ) from exc
    return tree_sitter, get_language, None


def _node_text(node: Any, source: bytes) -> str:
    """Extract text from a tree-sitter node."""
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _find_parent_class(node: Any) -> str | None:
    """Walk up the AST to find the enclosing class_definition name."""
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            name_node = current.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8", errors="replace")
        current = current.parent
    return None


def _extract_docstring(func_node: Any, source: bytes) -> str | None:
    """Extract the docstring from a function/class body."""
    body = func_node.child_by_field_name("body")
    if body is None or len(body.children) == 0:
        return None
    first_stmt = body.children[0]
    # tree-sitter Python may represent docstrings as:
    # 1. expression_statement > string (older grammars)
    # 2. string directly in block (newer grammars)
    string_node = None
    if first_stmt.type == "expression_statement":
        expr = first_stmt.children[0] if first_stmt.children else None
        if expr and expr.type == "string":
            string_node = expr
    elif first_stmt.type == "string":
        string_node = first_stmt

    if string_node is not None:
        raw = _node_text(string_node, source)
        # Strip triple-quote markers
        for quote in ('"""', "'''", '"', "'"):
            if raw.startswith(quote) and raw.endswith(quote):
                raw = raw[len(quote):-len(quote)]
                break
        return raw.strip()
    return None


def _extract_return_type(func_node: Any, source: bytes) -> str | None:
    """Extract return type annotation from function_definition."""
    rt = func_node.child_by_field_name("return_type")
    if rt:
        return _node_text(rt, source)
    return None


def _make_qualified_name(
    file_path: str, name: str, parent_name: str | None
) -> str:
    if parent_name:
        return f"{file_path}::{parent_name}.{name}"
    return f"{file_path}::{name}"


def _run_query(
    query_src: str, root_node: Any, language_name: str = "python"
) -> list[tuple[int, dict[str, list[Any]]]]:
    """Run a tree-sitter query and return matches."""
    ts, get_language, _ = _get_ts_modules()
    lang = get_language(language_name)
    query = ts.Query(lang, query_src)
    cursor = ts.QueryCursor(query)
    return cursor.matches(root_node)


class PythonExtractor(BaseExtractor):
    """Python-specific AST extractor using S-expression queries."""

    def extract_functions(self) -> list[GraphNode]:
        """Extract all function/method definitions."""
        matches = _run_query(FUNC_QUERY_SRC, self._root)
        nodes: list[GraphNode] = []

        for _pattern_idx, captures in matches:
            func_defs = captures.get("func.def", [])
            func_names = captures.get("func.name", [])
            func_params = captures.get("func.params", [])

            for i, func_def in enumerate(func_defs):
                name_node = func_names[i] if i < len(func_names) else None
                params_node = func_params[i] if i < len(func_params) else None
                if name_node is None:
                    continue

                name = _node_text(name_node, self._source)
                parent_name = _find_parent_class(func_def)
                kind = NodeKind.METHOD if parent_name else NodeKind.FUNCTION
                line_start = func_def.start_point[0]
                line_end = func_def.end_point[0]

                # Build signature
                params_text = _node_text(params_node, self._source) if params_node else "()"
                return_type = _extract_return_type(func_def, self._source)
                sig = f"def {name}{params_text}"
                if return_type:
                    sig += f" -> {return_type}"

                # Docstring
                docstring = _extract_docstring(func_def, self._source)

                # Decorators
                decorators: list[str] = []
                for child in func_def.children:
                    if child.type == "decorator":
                        decorators.append(_node_text(child, self._source))

                extra: dict[str, Any] = {}
                if decorators:
                    extra["decorators"] = decorators

                node = GraphNode(
                    node_id=_new_id(),
                    kind=kind,
                    name=name,
                    qualified_name=_make_qualified_name(
                        self._file_path, name, parent_name
                    ),
                    file_path=self._file_path,
                    line_start=line_start,
                    line_end=line_end,
                    language="python",
                    parent_name=parent_name,
                    signature=sig,
                    docstring=docstring,
                    extra_json=json.dumps(extra) if extra else "{}",
                )
                nodes.append(node)

        return nodes

    def extract_classes(self) -> list[GraphNode]:
        """Extract all class definitions, including INHERITS edges."""
        matches = _run_query(CLASS_QUERY_SRC, self._root)
        nodes: list[GraphNode] = []
        self._inherits_info: list[tuple[str, str, int]] = []  # (class_qname, base_name, line)

        for _pattern_idx, captures in matches:
            class_defs = captures.get("class.def", [])
            class_names = captures.get("class.name", [])

            for i, class_def in enumerate(class_defs):
                name_node = class_names[i] if i < len(class_names) else None
                if name_node is None:
                    continue

                name = _node_text(name_node, self._source)
                line_start = class_def.start_point[0]
                line_end = class_def.end_point[0]

                # Check for parent class in enclosing class
                parent_class = _find_parent_class(class_def)

                qualified_name = _make_qualified_name(
                    self._file_path, name, parent_class
                )

                # Extract superclasses
                for child in class_def.children:
                    if child.type == "argument_list":
                        for arg in child.named_children:
                            base_name = _node_text(arg, self._source)
                            self._inherits_info.append(
                                (qualified_name, base_name, line_start)
                            )

                # Docstring
                docstring = _extract_docstring(class_def, self._source)

                node = GraphNode(
                    node_id=_new_id(),
                    kind=NodeKind.CLASS,
                    name=name,
                    qualified_name=qualified_name,
                    file_path=self._file_path,
                    line_start=line_start,
                    line_end=line_end,
                    language="python",
                    parent_name=parent_class,
                    docstring=docstring,
                )
                nodes.append(node)

        return nodes

    def extract_imports(
        self,
    ) -> tuple[list[GraphEdge], dict[str, tuple[str, str]]]:
        """Extract import statements and build import_map."""
        edges: list[GraphEdge] = []
        import_map: dict[str, tuple[str, str]] = {}

        # Process "import X" statements
        matches = _run_query(IMPORT_QUERY_SRC, self._root)
        for _pat, captures in matches:
            for stmt in captures.get("import.stmt", []):
                for child in stmt.named_children:
                    if child.type == "dotted_name":
                        module = _node_text(child, self._source)
                        local_name = module.split(".")[-1]
                        import_map[local_name] = (module, local_name)
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            module = _node_text(name_node, self._source)
                            alias = (
                                _node_text(alias_node, self._source)
                                if alias_node
                                else module.split(".")[-1]
                            )
                            import_map[alias] = (module, module.split(".")[-1])

                edges.append(GraphEdge(
                    edge_id=_new_id(),
                    kind=EdgeKind.IMPORTS,
                    source_node_id="__unresolved__",
                    target_node_id="__unresolved__",
                    file_path=self._file_path,
                    line=stmt.start_point[0],
                ))

        # Process "from X import Y" statements
        from_matches = _run_query(IMPORT_FROM_QUERY_SRC, self._root)
        for _pat, captures in from_matches:
            for stmt in captures.get("import.from_stmt", []):
                module_name = ""
                # Find module name (dotted_name or relative_import)
                for child in stmt.children:
                    if child.type == "dotted_name":
                        if not module_name:
                            module_name = _node_text(child, self._source)
                            continue
                    elif child.type == "relative_import":
                        module_name = _node_text(child, self._source)
                        continue

                # Find imported names
                imported_names: list[tuple[str, str]] = []
                seen_import_keyword = False
                for child in stmt.children:
                    if child.type == "import":
                        seen_import_keyword = True
                        continue
                    if not seen_import_keyword:
                        continue
                    if child.type == "dotted_name":
                        imp_name = _node_text(child, self._source)
                        imported_names.append((imp_name, imp_name))
                    elif child.type == "aliased_import":
                        name_n = child.child_by_field_name("name")
                        alias_n = child.child_by_field_name("alias")
                        if name_n:
                            orig = _node_text(name_n, self._source)
                            alias = (
                                _node_text(alias_n, self._source)
                                if alias_n
                                else orig
                            )
                            imported_names.append((alias, orig))
                    elif child.type == "wildcard_import":
                        imported_names.append(("*", "*"))

                for local_name, orig_name in imported_names:
                    import_map[local_name] = (module_name, orig_name)

                edges.append(GraphEdge(
                    edge_id=_new_id(),
                    kind=EdgeKind.IMPORTS,
                    source_node_id="__unresolved__",
                    target_node_id="__unresolved__",
                    file_path=self._file_path,
                    line=stmt.start_point[0],
                ))

        return edges, import_map

    def extract_calls(
        self, import_map: dict[str, tuple[str, str]]
    ) -> list[GraphEdge]:
        """Extract function/method calls."""
        edges: list[GraphEdge] = []

        # Simple calls: foo()
        matches = _run_query(CALL_QUERY_SRC, self._root)
        for _pat, captures in matches:
            call_names = captures.get("call.name", [])
            for i, name_node in enumerate(call_names):
                call_name = _node_text(name_node, self._source)
                line = name_node.start_point[0]
                # Find enclosing function for source context
                confidence = 1.0 if call_name in import_map else self._config.heuristic_confidence
                edges.append(GraphEdge(
                    edge_id=_new_id(),
                    kind=EdgeKind.CALLS,
                    source_node_id="__unresolved__",
                    target_node_id=f"__call__{call_name}",
                    file_path=self._file_path,
                    line=line,
                    confidence=confidence,
                    extra_json=json.dumps({"call_name": call_name}),
                ))

        # Method calls: obj.method()
        method_matches = _run_query(CALL_METHOD_QUERY_SRC, self._root)
        for _pat, captures in method_matches:
            method_names = captures.get("call.method", [])
            for name_node in method_names:
                method_name = _node_text(name_node, self._source)
                line = name_node.start_point[0]
                edges.append(GraphEdge(
                    edge_id=_new_id(),
                    kind=EdgeKind.CALLS,
                    source_node_id="__unresolved__",
                    target_node_id=f"__call__{method_name}",
                    file_path=self._file_path,
                    line=line,
                    confidence=self._config.heuristic_confidence,
                    extra_json=json.dumps({"call_name": method_name, "is_method": True}),
                ))

        return edges
