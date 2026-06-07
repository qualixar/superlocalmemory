# compress/extractive_code.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# AST structural patterns adapted from:
#   headroom/compression/handlers/code_handler.py:66-138 (Apache-2.0)
#   Specifically: _STRUCTURAL_NODE_TYPES per-language dict, CodeLanguage enum
#   headroom/compression/handlers/code_handler.py:141-150 — _SIGNATURE_PATTERNS regex fallback
#   Attribution: See ATTRIBUTION.md.

"""CodeCompressor — AST-aware extractive code compressor.

Supported languages: Python, JavaScript, Go, Rust, Java, C++.
Path A (tree-sitter): used if tree-sitter-language-pack installed.
Path B (regex fallback): used otherwise. Tests pass on both paths.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("slm.optimize.compress.code")

_BODY_STUB_BY_LANG: dict[str, str] = {
    "python":     "    # [slm: body compressed — retrieve with ccr_id={ccr_id}]",
    "javascript": "    // [slm: body compressed — retrieve with ccr_id={ccr_id}]",
    "go":         "    // [slm: body compressed — retrieve with ccr_id={ccr_id}]",
    "rust":       "    // [slm: body compressed — retrieve with ccr_id={ccr_id}]",
    "java":       "    // [slm: body compressed — retrieve with ccr_id={ccr_id}]",
    "cpp":        "    // [slm: body compressed — retrieve with ccr_id={ccr_id}]",
}
_BODY_STUB_NO_CCR_BY_LANG: dict[str, str] = {
    "python":     "    # [slm: body compressed]",
    "javascript": "    // [slm: body compressed]",
    "go":         "    // [slm: body compressed]",
    "rust":       "    // [slm: body compressed]",
    "java":       "    // [slm: body compressed]",
    "cpp":        "    // [slm: body compressed]",
}

_MIN_BODY_LINES: int = 4


class CodeLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"


@dataclass(frozen=True)
class CodeSpan:
    start_line: int
    end_line: int
    role: str  # "import" | "signature" | "body" | "class_header" | "decorator"
    is_structural: bool


# ── Signature patterns for regex path (Path B) ────────────────────────────────

_SIGNATURE_PATTERNS: dict[str, list[str]] = {
    "python": [
        r"^\s*(async\s+)?def\s+\w+\s*\([^)]*\)\s*(->\s*[^:]+)?:",
        r"^\s*class\s+\w+(\([^)]*\))?:",
        r"^\s*import\s+",
        r"^\s*from\s+\w+\s+import",
        r"^\s*@\w+",
    ],
    "javascript": [
        r"^\s*(async\s+)?function\s+\w+\s*\([^)]*\)",
        r"^\s*class\s+\w+(\s+extends\s+\w+)?",
        r"^\s*import\s+",
        r"^\s*const\s+\w+\s*=\s*(async\s+)?\(",
        r"^\s*export\s+(default\s+|const\s+|class\s+|function\s+)",
    ],
    "go": [
        r"^\s*func\s+(\(\w+\s+\*?\w+\)\s*)?\w+\s*\(",
        r"^\s*import\s+",
        r"^\s*package\s+",
        r"^\s*type\s+\w+\s+(struct|interface)\s*\{",
    ],
    "rust": [
        r"^\s*(pub\s+)?(async\s+)?fn\s+\w+",
        r"^\s*use\s+",
        r"^\s*impl\s+",
        r"^\s*struct\s+",
        r"^\s*enum\s+",
        r"^\s*trait\s+",
    ],
    "java": [
        r"^\s*(public|private|protected)\s+(static\s+)?\w+\s+\w+\s*\(",
        r"^\s*import\s+",
        r"^\s*(public|private|protected)?\s*class\s+\w+",
    ],
    "cpp": [
        r"^\s*\w[\w\s\*&:<,>]*\s+\w+\s*\([^)]*\)\s*(const\s*)?\{",
        r"^\s*#include\s+",
        r"^\s*(class|struct)\s+\w+",
    ],
}


class CodeCompressor:
    """AST-aware extractive code compressor. Thread-safe (no mutable state)."""

    def compress(self, code: str, language: str, ccr_id: str = "") -> str:
        try:
            lang = CodeLanguage(language)
        except ValueError:
            logger.warning("CodeCompressor: unknown language %r — passthrough", language)
            return code

        try:
            if _tree_sitter_available():
                return self._compress_with_tree_sitter(code, lang, ccr_id)
            else:
                return self._compress_with_regex(code, lang, ccr_id)
        except Exception as exc:
            logger.warning("CodeCompressor.compress failed — passthrough: %s", exc)
            return code

    def _compress_with_tree_sitter(self, code: str, lang: CodeLanguage, ccr_id: str) -> str:
        from tree_sitter_language_pack import get_parser  # type: ignore[import]
        parser = get_parser(lang.value)
        tree = parser.parse(code.encode())
        lines = code.split("\n")
        spans = _collect_spans_tree_sitter(tree.root_node, lang)
        return _apply_spans(lines, spans, ccr_id, lang.value)

    def _compress_with_regex(self, code: str, lang: CodeLanguage, ccr_id: str) -> str:
        lines = code.split("\n")
        spans = _collect_spans_regex(lines, lang)
        return _apply_spans(lines, spans, ccr_id, lang.value)


# ── Span collection ───────────────────────────────────────────────────────────

def _collect_spans_tree_sitter(root_node: object, lang: CodeLanguage) -> list[CodeSpan]:
    _STRUCTURAL_TYPES: dict[str, set[str]] = {
        "python": {"import_statement", "import_from_statement", "function_definition",
                   "class_definition", "decorated_definition", "type_alias_statement"},
        "javascript": {"import_statement", "export_statement", "function_declaration",
                       "class_declaration", "method_definition", "arrow_function"},
        "go": {"import_declaration", "function_declaration", "method_declaration",
               "type_declaration", "interface_type"},
        "rust": {"use_declaration", "function_item", "impl_item", "struct_item",
                 "enum_item", "trait_item"},
        "java": {"import_declaration", "class_declaration", "method_declaration",
                 "interface_declaration", "annotation"},
        "cpp": {"function_definition", "preproc_include", "class_specifier"},
    }
    structural = _STRUCTURAL_TYPES.get(lang.value, set())
    spans: list[CodeSpan] = []

    def _walk(node: object, depth: int = 0) -> None:
        start_row: int = getattr(node, "start_point", (0, 0))[0]
        end_row: int = getattr(node, "end_point", (0, 0))[0]
        node_type: str = getattr(node, "type", "")
        is_struct = node_type in structural

        if is_struct and node_type in {
            "function_definition", "method_definition", "function_declaration",
            "function_item", "method_declaration",
        }:
            body_node = _find_child(node, "block") or _find_child(node, "body")
            if body_node is not None:
                body_start = getattr(body_node, "start_point", (0, 0))[0]
                sig_end = body_start - 1
                if sig_end > start_row:
                    spans.append(CodeSpan(start_row, sig_end, "signature", True))
                spans.append(CodeSpan(body_start, end_row, "body", False))
            else:
                spans.append(CodeSpan(start_row, end_row, "signature", True))
        elif is_struct and node_type in {"decorated_definition"}:
            def_node = _find_child(node, "function_definition") or _find_child(node, "class_definition")
            if def_node is not None:
                def_start = getattr(def_node, "start_point", (0, 0))[0]
                spans.append(CodeSpan(start_row, def_start - 1, "decorator", True))
                _walk(def_node, depth + 1)
            else:
                spans.append(CodeSpan(start_row, end_row, node_type, True))
        elif is_struct:
            spans.append(CodeSpan(start_row, end_row, "signature", True))
        else:
            children = getattr(node, "children", None)
            if children is not None:
                has_named = False
                for child in children:
                    if getattr(child, "is_named", False):
                        has_named = True
                        _walk(child, depth + 1)
                if not has_named:
                    spans.append(CodeSpan(start_row, end_row, "body", False))

    _walk(root_node)
    return spans


def _find_child(node: object, child_type: str) -> object | None:
    children = getattr(node, "children", None)
    if children is None:
        return None
    for child in children:
        if hasattr(child, "type") and child.type == child_type:
            return child
    return None


def _collect_spans_regex(lines: list[str], lang: CodeLanguage) -> list[CodeSpan]:
    patterns = _SIGNATURE_PATTERNS.get(lang.value, [])
    spans: list[CodeSpan] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        matched = False
        for pat in patterns:
            if re.search(pat, line):
                body_start = i + 1
                body_end = body_start
                while body_end < n and (
                    lines[body_end].startswith((" ", "\t", "", "#", "//", "/*", "*/", "{"))
                    or re.match(r"^\s*$", lines[body_end])
                ):
                    if re.match(r"^\s*($|#|//|/\*|\*/|\*|}|\)|\])", lines[body_end]):
                        body_end += 1
                    else:
                        break
                body_len = body_end - body_start
                if body_len >= _MIN_BODY_LINES and _looks_like_body(lines, body_start, body_end):
                    spans.append(CodeSpan(i, i, "signature", True))
                    spans.append(CodeSpan(body_start, body_end, "body", False))
                    i = body_end
                    matched = True
                    break
                else:
                    body_start = body_end
            # else: pattern didn't match
        if not matched:
            i += 1
    return spans


def _looks_like_body(lines: list[str], start: int, end: int) -> bool:
    body_lines = [l for l in lines[start:end] if l.strip() and not l.strip().startswith(("#", "//"))]
    return len(body_lines) >= _MIN_BODY_LINES


def _apply_spans(
    lines: list[str],
    spans: list[CodeSpan],
    ccr_id: str,
    lang: str = "python",
) -> str:
    if not spans:
        return "\n".join(lines)

    stub_tmpl = _BODY_STUB_BY_LANG.get(lang, _BODY_STUB_BY_LANG["python"])
    stub_no_ccr = _BODY_STUB_NO_CCR_BY_LANG.get(lang, _BODY_STUB_NO_CCR_BY_LANG["python"])
    stub = stub_tmpl.format(ccr_id=ccr_id) if ccr_id else stub_no_ccr

    sorted_spans = sorted(spans, key=lambda s: s.start_line)
    start_set = set()
    deduped: list[CodeSpan] = []
    for s in sorted_spans:
        if s.start_line not in start_set:
            deduped.append(s)
            start_set.add(s.start_line)

    span_by_start: dict[int, CodeSpan] = {s.start_line: s for s in deduped}

    result_lines: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        span = span_by_start.get(i)
        if span is not None and not span.is_structural:
            end = min(span.end_line, n - 1)
            body_len = end - i + 1
            if body_len >= _MIN_BODY_LINES:
                result_lines.append(stub)
                i = end + 1
            else:
                for j in range(i, end + 1):
                    result_lines.append(lines[j])
                i = end + 1
        elif span is not None and span.is_structural:
            end = min(span.end_line, n - 1)
            for j in range(i, end + 1):
                result_lines.append(lines[j])
            i = end + 1
        else:
            result_lines.append(lines[i])
            i += 1

    return "\n".join(result_lines)


def _tree_sitter_available() -> bool:
    try:
        import tree_sitter_language_pack  # noqa: F401
        return True
    except ImportError:
        return False
