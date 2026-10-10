# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""Multi-language tree-sitter parser with parallel execution.

Dispatches to language-specific extractors (Python, TypeScript).
Uses ProcessPoolExecutor for CPU-bound parallel file parsing.

tree-sitter imports are lazy (HR-07): only imported when parse_file
is called, never at module-level or package import time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from superlocalmemory.code_graph.config import CodeGraphConfig
from superlocalmemory.code_graph.models import (
    EdgeKind,
    FileRecord,
    GraphEdge,
    GraphNode,
    NodeKind,
    ParseResult,
)
from superlocalmemory.code_graph.resolver import ImportResolver
from superlocalmemory.storage.models import _new_id

logger = logging.getLogger(__name__)


class UnsupportedLanguageError(Exception):
    """Raised when a language is not supported."""


class ParseError(Exception):
    """Raised when tree-sitter parsing fails."""


def _is_test_file(file_path: str, config: CodeGraphConfig) -> bool:
    """Check if a file is a test file based on config patterns."""
    name = Path(file_path).name
    parts = Path(file_path).parts
    test_patterns = [
        "test_*.py", "*_test.py",
        "*.test.ts", "*.test.tsx", "*.spec.ts", "*.spec.tsx",
    ]
    for pattern in test_patterns:
        if fnmatch(name, pattern):
            return True
    # Check directory patterns
    test_dirs = {"tests", "test", "__tests__", "spec"}
    return bool(test_dirs.intersection(parts))


def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def _assign_edge_sources(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[GraphEdge]:
    """Replace ``__unresolved__`` source_node_ids on CALLS edges.

    Locates the innermost function/method whose line-range contains the call
    site.  Falls back to the file-level node when no function wraps the call
    (e.g. top-level script code).  IMPORTS and other non-CALLS edges whose
    source is still ``__unresolved__`` are dropped — they cannot be resolved
    without storing the module path in the edge, which is a separate defect.

    Returns a new list of edges.
    """
    # ── build per-file line-range index ────────────────────────────────
    # {file_path: [(line_start, line_end, node_id), ...]}
    # Sorted by range size ascending so the NARROWEST (innermost) match wins.
    file_ranges: dict[str, list[tuple[int, int, str]]] = {}
    file_node_ids: dict[str, str] = {}      # file_path → FILE node_id

    for node in nodes:
        if node.kind == NodeKind.FILE:
            file_node_ids[node.file_path] = node.node_id
        elif node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
            fp = node.file_path
            file_ranges.setdefault(fp, []).append(
                (node.line_start, node.line_end, node.node_id)
            )

    for fp in file_ranges:
        file_ranges[fp].sort(key=lambda t: t[1] - t[0])   # narrowest first

    resolved: list[GraphEdge] = []
    dropped = 0

    for edge in edges:
        if edge.source_node_id != "__unresolved__":
            resolved.append(edge)
            continue

        if edge.kind != EdgeKind.CALLS:
            # IMPORTS and other edges: source unknown — drop.
            # Root cause: extract_imports() does not store the module path in
            # extra_json, so targets are also unresolvable.  Tracked as a
            # separate pre-existing defect; fixing it here would require an
            # extractor change beyond this PR scope.
            dropped += 1
            continue

        # ── find innermost enclosing function ──────────────────────────
        enclosing_id: str | None = None
        for ls, le, nid in file_ranges.get(edge.file_path, []):
            if ls <= edge.line <= le:
                enclosing_id = nid
                break                      # narrowest first → stop at first hit

        # ── fallback: top-level call, use FILE node ─────────────────────
        if enclosing_id is None:
            enclosing_id = file_node_ids.get(edge.file_path)

        if enclosing_id is None:
            logger.debug(
                "_assign_edge_sources: no source node for call at %s:%d — dropping",
                edge.file_path, edge.line,
            )
            dropped += 1
            continue

        resolved.append(GraphEdge(
            edge_id=edge.edge_id,
            kind=edge.kind,
            source_node_id=enclosing_id,
            target_node_id=edge.target_node_id,
            file_path=edge.file_path,
            line=edge.line,
            confidence=edge.confidence,
            extra_json=edge.extra_json,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
        ))

    if dropped:
        logger.debug(
            "_assign_edge_sources: dropped %d edge(s) with unresolvable source",
            dropped,
        )

    return resolved


def _clean_and_resolve_edges(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    import_maps_by_file: dict[str, dict[str, tuple[str, str]]],
    repo_root: Path,
    config: CodeGraphConfig,
) -> list[GraphEdge]:
    """Full resolution pipeline applied after parse_all collects all nodes.

    1. Assign real source_node_ids to CALLS edges (line-range lookup).
    2. Drop IMPORTS edges whose endpoints are still ``__unresolved__``.
    3. Run ImportResolver.resolve_call_targets to replace ``__call__<name>``
       targets with real node_ids or drop the edge (external calls).
    4. Final defensive pass: drop any edge whose endpoints are still absent
       from the full node set (belt-and-braces, should be a no-op after 1-3).

    Returns a clean edge list safe to hand to GraphStore.
    """
    # Step 1: fix sources
    edges = _assign_edge_sources(nodes, edges)

    # Step 2: resolve targets
    resolver = ImportResolver(repo_root, config)
    edges = resolver.resolve_call_targets(nodes, edges, import_maps_by_file)

    # Step 3: final drop of any remaining stragglers
    valid_ids = {n.node_id for n in nodes}
    clean: list[GraphEdge] = []
    stray = 0
    for e in edges:
        if e.source_node_id in valid_ids and e.target_node_id in valid_ids:
            clean.append(e)
        else:
            stray += 1

    if stray:
        logger.debug(
            "_clean_and_resolve_edges: final pass dropped %d stray edge(s)",
            stray,
        )

    return clean


def _make_qualified_name(
    file_path: str, name: str, parent_name: str | None
) -> str:
    if parent_name:
        return f"{file_path}::{parent_name}.{name}"
    return f"{file_path}::{name}"


# ---------------------------------------------------------------------------
# Module-level parse function (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _parse_file_standalone(
    file_path_str: str,
    source_bytes: bytes,
    language: str,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    """Parse a single file. Module-level function for pickling.

    Returns a serializable dict with nodes, edges, errors.
    """
    try:
        # Lazy import (HR-07)
        from tree_sitter_language_pack import get_parser  # noqa: F811

        parser_instance = get_parser(language)
        tree = parser_instance.parse(source_bytes)
        root = tree.root_node

        # Create a minimal config for the extractor
        config = CodeGraphConfig(**{
            k: v for k, v in config_dict.items()
            if k in CodeGraphConfig.__dataclass_fields__
        })

        # Select extractor
        if language == "python":
            from superlocalmemory.code_graph.extractors.python import PythonExtractor
            extractor = PythonExtractor(root, source_bytes, file_path_str, config)
        elif language in ("typescript", "tsx", "javascript", "jsx"):
            from superlocalmemory.code_graph.extractors.typescript import TypeScriptExtractor
            extractor = TypeScriptExtractor(root, source_bytes, file_path_str, config)
        else:
            return {"nodes": [], "edges": [], "errors": [f"Unsupported language: {language}"]}

        nodes, edges, import_map = extractor.extract_with_import_map()

        return {
            "nodes": nodes,
            "edges": edges,
            "import_map": import_map,
            "errors": [],
        }
    except Exception as exc:
        return {
            "nodes": [],
            "edges": [],
            "errors": [str(exc)],
        }


class CodeParser:
    """Multi-language tree-sitter parser with parallel execution."""

    def __init__(self, config: CodeGraphConfig) -> None:
        """Store config. Does not import tree-sitter yet (lazy)."""
        self._config = config

    def discover_files(self, repo_root: Path) -> list[Path]:
        """Find all parseable files under repo_root.

        Returns relative paths sorted alphabetically.
        Raises FileNotFoundError if repo_root does not exist.
        """
        if not repo_root.exists():
            raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

        results: list[Path] = []
        exclude_dirs = self._config.exclude_dirs

        for dirpath, dirnames, filenames in os.walk(repo_root):
            # Prune excluded directories (modifying dirnames in-place)
            dirnames[:] = [
                d for d in dirnames
                if d not in exclude_dirs
                and not any(fnmatch(d, p) for p in exclude_dirs)
            ]

            for filename in filenames:
                # Check extension
                ext = Path(filename).suffix
                if ext not in self._config.extension_map:
                    continue

                # Check file size
                full_path = Path(dirpath) / filename
                try:
                    size = full_path.stat().st_size
                except OSError:
                    continue

                if size > self._config.max_file_size_bytes:
                    logger.warning(
                        "Skipping large file (%d bytes): %s", size, full_path
                    )
                    continue

                # Check exclude patterns
                rel = full_path.relative_to(repo_root)
                skip = False
                for pattern in self._config.exclude_patterns:
                    if fnmatch(rel.as_posix(), pattern) or fnmatch(filename, pattern):
                        skip = True
                        break
                if skip:
                    continue

                results.append(rel)

        return sorted(results)

    def parse_file(
        self,
        file_path: Path,
        source_bytes: bytes,
        language: str,
    ) -> tuple[list[GraphNode], list[GraphEdge], dict[str, tuple[str, str]]]:
        """Parse a single file and return extracted nodes and edges.

        Raises UnsupportedLanguageError if language is not supported.
        """
        supported = {"python", "typescript", "tsx", "javascript", "jsx"}
        if language not in supported:
            raise UnsupportedLanguageError(f"Unsupported language: {language}")

        try:
            from tree_sitter_language_pack import get_parser
        except ImportError as exc:
            raise ImportError(
                "tree-sitter required. Install: pip install 'superlocalmemory[code-graph]'"
            ) from exc

        parser = get_parser(language)
        tree = parser.parse(source_bytes)
        root = tree.root_node

        file_path_str = Path(file_path).as_posix()  # one form on every OS
        content_hash = _sha256(source_bytes)

        # Create File node
        file_node = GraphNode(
            node_id=_new_id(),
            kind=NodeKind.FILE,
            name=file_path.name,
            qualified_name=file_path_str,
            file_path=file_path_str,
            line_start=0,
            line_end=root.end_point[0],
            language=language,
            content_hash=content_hash,
        )

        # Select and run extractor
        if language == "python":
            from superlocalmemory.code_graph.extractors.python import PythonExtractor
            extractor = PythonExtractor(root, source_bytes, file_path_str, self._config)
        else:
            from superlocalmemory.code_graph.extractors.typescript import TypeScriptExtractor
            extractor = TypeScriptExtractor(root, source_bytes, file_path_str, self._config)

        extracted_nodes, extracted_edges, file_import_map = extractor.extract_with_import_map()

        # Check if test file
        is_test = _is_test_file(file_path_str, self._config)

        # Mark test functions
        if is_test:
            marked_nodes: list[GraphNode] = []
            for node in extracted_nodes:
                if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                    marked_nodes.append(GraphNode(
                        node_id=node.node_id,
                        kind=node.kind,
                        name=node.name,
                        qualified_name=node.qualified_name,
                        file_path=node.file_path,
                        line_start=node.line_start,
                        line_end=node.line_end,
                        language=node.language,
                        parent_name=node.parent_name,
                        signature=node.signature,
                        docstring=node.docstring,
                        is_test=True,
                        content_hash=node.content_hash,
                        extra_json=node.extra_json,
                    ))
                else:
                    marked_nodes.append(node)
            extracted_nodes = marked_nodes

        all_nodes = [file_node] + extracted_nodes

        # Generate CONTAINS edges
        contains_edges = self._generate_contains_edges(file_node, extracted_nodes)

        # Generate TESTED_BY edges
        tested_by_edges = self._generate_tested_by_edges(
            extracted_nodes, extracted_edges
        )

        all_edges = extracted_edges + contains_edges + tested_by_edges

        return all_nodes, all_edges, file_import_map

    def parse_all(
        self, repo_root: Path
    ) -> tuple[list[GraphNode], list[GraphEdge], list[FileRecord]]:
        """Parse entire project in parallel.

        Returns (all_nodes, all_edges, all_file_records).
        """
        files = self.discover_files(repo_root)
        if not files:
            return [], [], []

        all_nodes: list[GraphNode] = []
        all_edges: list[GraphEdge] = []
        all_file_records: list[FileRecord] = []
        # Keyed by relative file_path string; populated by the parallel path
        # (the sequential path goes through parse_file which does not expose
        # the per-file import map, so it stays empty — resolver Strategy 2/3
        # still works without it).
        import_maps_by_file: dict[str, dict[str, tuple[str, str]]] = {}

        # Read files and prepare tasks
        tasks: list[tuple[Path, bytes, str]] = []
        for rel_path in files:
            full_path = repo_root / rel_path
            try:
                source_bytes = full_path.read_bytes()
            except OSError as exc:
                logger.warning("Failed to read %s: %s", full_path, exc)
                continue

            ext = rel_path.suffix
            language = self._config.extension_map.get(ext)
            if language is None:
                continue

            tasks.append((rel_path, source_bytes, language))

        # Parse with ProcessPoolExecutor for parallel CPU-bound work
        # For small numbers of files, run sequentially to avoid overhead
        # Use the sequential (in-process) path for small repos or when the
        # caller explicitly requests single-worker execution.  parallel_workers=1
        # is the only way to guarantee in-process parsing (no subprocess spawn).
        if len(tasks) <= 2 or self._config.parallel_workers == 1:
            for rel_path, source_bytes, language in tasks:
                try:
                    nodes, edges, file_import_map = self.parse_file(
                        rel_path, source_bytes, language
                    )
                    import_maps_by_file[rel_path.as_posix()] = file_import_map
                    all_nodes.extend(nodes)
                    all_edges.extend(edges)
                    all_file_records.append(FileRecord(
                        file_path=rel_path.as_posix(),
                        content_hash=_sha256(source_bytes),
                        mtime=(repo_root / rel_path).stat().st_mtime,
                        language=language,
                        node_count=len(nodes),
                        edge_count=len(edges),
                        last_indexed=time.time(),
                    ))
                except Exception as exc:
                    logger.warning("Failed to parse %s: %s", rel_path, exc)
            # import_maps_by_file is now populated on the sequential path:
            # parse_file returns the per-file import map (Strategy 1 works
            # for small repos / parallel_workers=1 runs too).
        else:
            # Parallel execution
            parse_failures = 0   # count of futures that raised (worker crashes)
            config_dict = {
                field_name: getattr(self._config, field_name)
                for field_name in CodeGraphConfig.__dataclass_fields__
                if not isinstance(getattr(self._config, field_name), Path)
            }
            # Convert Path fields to strings
            config_dict["repo_root"] = str(self._config.repo_root)

            workers = min(self._config.parallel_workers, len(tasks))
            # spawn, not fork: a forked worker would inherit the daemon's instance-lock fd
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                future_map = {}
                for rel_path, source_bytes, language in tasks:
                    future = executor.submit(
                        _parse_file_standalone,
                        rel_path.as_posix(),
                        source_bytes,
                        language,
                        config_dict,
                    )
                    future_map[future] = (rel_path, source_bytes, language)

                for future in as_completed(future_map):
                    rel_path, source_bytes, language = future_map[future]
                    try:
                        result = future.result(timeout=self._config.parse_timeout_seconds)
                    except Exception as exc:
                        logger.warning("Parse failed for %s: %s", rel_path, exc)
                        parse_failures += 1
                        continue

                    if result["errors"]:
                        for err in result["errors"]:
                            logger.warning("Parse error in %s: %s", rel_path, err)
                        if not result["nodes"]:
                            continue

                    file_nodes = result["nodes"]
                    file_edges = result["edges"]
                    # Collect per-file import map for Strategy 1 resolution.
                    import_maps_by_file[rel_path.as_posix()] = result.get("import_map", {})

                    # Build the full parse result with file node and CONTAINS edges
                    file_path_str = rel_path.as_posix()
                    content_hash = _sha256(source_bytes)

                    file_node = GraphNode(
                        node_id=_new_id(),
                        kind=NodeKind.FILE,
                        name=rel_path.name,
                        qualified_name=file_path_str,
                        file_path=file_path_str,
                        line_start=0,
                        line_end=0,
                        language=language,
                        content_hash=content_hash,
                    )

                    is_test = _is_test_file(file_path_str, self._config)
                    if is_test:
                        marked: list[GraphNode] = []
                        for n in file_nodes:
                            if n.kind in (NodeKind.FUNCTION, NodeKind.METHOD):
                                marked.append(GraphNode(
                                    node_id=n.node_id, kind=n.kind, name=n.name,
                                    qualified_name=n.qualified_name,
                                    file_path=n.file_path,
                                    line_start=n.line_start, line_end=n.line_end,
                                    language=n.language, parent_name=n.parent_name,
                                    signature=n.signature, docstring=n.docstring,
                                    is_test=True, content_hash=n.content_hash,
                                    extra_json=n.extra_json,
                                ))
                            else:
                                marked.append(n)
                        file_nodes = marked

                    contains = self._generate_contains_edges(file_node, file_nodes)
                    tested_by = self._generate_tested_by_edges(file_nodes, file_edges)

                    final_nodes = [file_node] + file_nodes
                    final_edges = file_edges + contains + tested_by

                    all_nodes.extend(final_nodes)
                    all_edges.extend(final_edges)

                    try:
                        mtime = (repo_root / rel_path).stat().st_mtime
                    except OSError:
                        mtime = 0.0

                    all_file_records.append(FileRecord(
                        file_path=file_path_str,
                        content_hash=content_hash,
                        mtime=mtime,
                        language=language,
                        node_count=len(final_nodes),
                        edge_count=len(final_edges),
                        last_indexed=time.time(),
                    ))

            # ── Fail-fast: majority pool failure ──────────────────────────────
            # >50% failure rate means worker processes were killed (OOM, missing
            # deps in subprocess env) — NOT a legitimate empty repo.  Returning
            # silently would let build_code_graph report success with zero nodes,
            # which is worse than the original FK crash.
            # Threshold rationale: ≥50% is a clear systemic failure; it allows
            # for a tail of legitimately-malformed files without triggering on
            # ordinary parse errors.
            if parse_failures > 0 and len(tasks) > 0:
                failure_rate = parse_failures / len(tasks)
                if failure_rate > 0.50:
                    raise RuntimeError(
                        f"Parsing aborted: {parse_failures}/{len(tasks)} files "
                        f"failed ({failure_rate:.0%}). Likely cause: process pool "
                        f"worker crash or missing tree-sitter grammars in subprocess. "
                        f"Set CodeGraphConfig(parallel_workers=1) for in-process parsing."
                    )

        # ── Resolution pipeline ────────────────────────────────────────────────
        # Resolve placeholder edge endpoints before returning.  Ensures that
        # every edge exiting parse_all has both endpoints in the returned node
        # set, so GraphStore never encounters a FOREIGN KEY constraint failure.
        # This is Fix A: wiring resolver.py into the parse pipeline.
        all_edges = _clean_and_resolve_edges(
            all_nodes, all_edges, import_maps_by_file, repo_root, self._config
        )

        return all_nodes, all_edges, all_file_records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_contains_edges(
        file_node: GraphNode, extracted_nodes: list[GraphNode]
    ) -> list[GraphEdge]:
        """Generate CONTAINS edges: File -> top-level, parent -> child."""
        edges: list[GraphEdge] = []
        # Build name -> node_id map for parent lookup
        name_to_id: dict[str, str] = {}
        for node in extracted_nodes:
            name_to_id[node.name] = node.node_id

        for node in extracted_nodes:
            if node.parent_name is None:
                # Top-level: File contains this node
                edges.append(GraphEdge(
                    edge_id=_new_id(),
                    kind=EdgeKind.CONTAINS,
                    source_node_id=file_node.node_id,
                    target_node_id=node.node_id,
                    file_path=file_node.file_path,
                    line=node.line_start,
                ))
            else:
                # Child: parent contains this node
                parent_id = name_to_id.get(node.parent_name)
                if parent_id:
                    edges.append(GraphEdge(
                        edge_id=_new_id(),
                        kind=EdgeKind.CONTAINS,
                        source_node_id=parent_id,
                        target_node_id=node.node_id,
                        file_path=file_node.file_path,
                        line=node.line_start,
                    ))
                else:
                    # Fallback: File contains
                    edges.append(GraphEdge(
                        edge_id=_new_id(),
                        kind=EdgeKind.CONTAINS,
                        source_node_id=file_node.node_id,
                        target_node_id=node.node_id,
                        file_path=file_node.file_path,
                        line=node.line_start,
                    ))
        return edges

    @staticmethod
    def _generate_tested_by_edges(
        nodes: list[GraphNode], edges: list[GraphEdge]
    ) -> list[GraphEdge]:
        """Generate TESTED_BY edges: for each CALLS from test to non-test."""
        test_node_ids = {n.node_id for n in nodes if n.is_test}
        if not test_node_ids:
            return []

        tested_by: list[GraphEdge] = []
        for edge in edges:
            if edge.kind == EdgeKind.CALLS and edge.source_node_id in test_node_ids:
                # Only if target is not a test node
                if edge.target_node_id not in test_node_ids:
                    tested_by.append(GraphEdge(
                        edge_id=_new_id(),
                        kind=EdgeKind.TESTED_BY,
                        source_node_id=edge.target_node_id,
                        target_node_id=edge.source_node_id,
                        file_path=edge.file_path,
                        line=edge.line,
                    ))
        return tested_by
