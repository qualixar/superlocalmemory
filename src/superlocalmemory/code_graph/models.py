# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4 — CodeGraph Module

"""Data models for the CodeGraph module.

Frozen dataclasses + string enums. All immutable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from superlocalmemory.storage.models import _new_id

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeKind(str, Enum):
    """Kind of code entity in the graph."""
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE = "module"


class EdgeKind(str, Enum):
    """Kind of relationship between code entities."""
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    CONTAINS = "contains"
    TESTED_BY = "tested_by"
    DEPENDS_ON = "depends_on"


class LinkType(str, Enum):
    """Type of bridge link between code node and SLM memory."""
    MENTIONS = "mentions"
    DECISION_ABOUT = "decision_about"
    BUG_FIX = "bug_fix"
    REFACTOR = "refactor"
    DESIGN_RATIONALE = "design_rationale"


# ---------------------------------------------------------------------------
# Frozen Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphNode:
    """A code entity in the graph (function, class, file, etc.)."""
    node_id: str = field(default_factory=_new_id)
    kind: NodeKind = NodeKind.FUNCTION
    name: str = ""
    qualified_name: str = ""
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    language: str = ""
    parent_name: str | None = None
    signature: str | None = None
    docstring: str | None = None
    is_test: bool = False
    content_hash: str | None = None
    community_id: int | None = None
    extra_json: str = "{}"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GraphEdge:
    """A relationship between two code entities."""
    edge_id: str = field(default_factory=_new_id)
    kind: EdgeKind = EdgeKind.CALLS
    source_node_id: str = ""
    target_node_id: str = ""
    file_path: str = ""
    line: int = 0
    confidence: float = 1.0
    extra_json: str = "{}"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class FileRecord:
    """Tracking record for a parsed source file."""
    file_path: str = ""
    content_hash: str = ""
    mtime: float = 0.0
    language: str = ""
    node_count: int = 0
    edge_count: int = 0
    last_indexed: float = field(default_factory=time.time)


@dataclass(frozen=True)
class CodeMemoryLink:
    """Bridge link between a code graph node and an SLM memory fact."""
    link_id: str = field(default_factory=_new_id)
    code_node_id: str = ""
    slm_fact_id: str = ""
    slm_entity_id: str | None = None
    link_type: LinkType = LinkType.MENTIONS
    confidence: float = 0.8
    created_at: str = ""
    last_verified: str | None = None
    is_stale: bool = False


# ---------------------------------------------------------------------------
# Parse result containers (used by parser → database pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a single file."""
    file_path: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    file_record: FileRecord
    errors: tuple[str, ...] = ()
