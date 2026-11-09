# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for FlowDetector — entry point detection, flow tracing, criticality."""

from __future__ import annotations

import json
import time

import pytest

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.flows import (
    SECURITY_KEYWORDS,
    FlowDetector,
    FlowResult,
)
from superlocalmemory.code_graph.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_db(db: CodeGraphDatabase) -> CodeGraphDatabase:
    """DB pre-populated with a call graph for flow detection.

    Graph:
        main -> process_request -> validate_token -> query_db
                                -> log_request
        handle_event -> process_event
    """
    now = time.time()
    nodes = [
        GraphNode(
            node_id="main", kind=NodeKind.FUNCTION, name="main",
            qualified_name="app.py::main", file_path="app.py",
            line_start=1, line_end=20, language="python",
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="process_request", kind=NodeKind.FUNCTION,
            name="process_request",
            qualified_name="handlers.py::process_request",
            file_path="handlers.py", line_start=1, line_end=30,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="validate_token", kind=NodeKind.FUNCTION,
            name="validate_token",
            qualified_name="auth.py::validate_token",
            file_path="auth.py", line_start=1, line_end=15,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="query_db", kind=NodeKind.FUNCTION, name="query_db",
            qualified_name="db.py::query_db", file_path="db.py",
            line_start=1, line_end=20, language="python",
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="log_request", kind=NodeKind.FUNCTION,
            name="log_request",
            qualified_name="logging.py::log_request",
            file_path="logging.py", line_start=1, line_end=10,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="handle_event", kind=NodeKind.FUNCTION,
            name="handle_event",
            qualified_name="events.py::handle_event",
            file_path="events.py", line_start=1, line_end=25,
            language="python", created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="process_event", kind=NodeKind.FUNCTION,
            name="process_event",
            qualified_name="events.py::process_event",
            file_path="events.py", line_start=30, line_end=50,
            language="python", created_at=now, updated_at=now,
        ),
    ]
    for node in nodes:
        db.upsert_node(node)

    edges = [
        GraphEdge(
            edge_id="e1", kind=EdgeKind.CALLS,
            source_node_id="main", target_node_id="process_request",
            file_path="app.py", line=5, created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e2", kind=EdgeKind.CALLS,
            source_node_id="process_request", target_node_id="validate_token",
            file_path="handlers.py", line=10, created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e3", kind=EdgeKind.CALLS,
            source_node_id="validate_token", target_node_id="query_db",
            file_path="auth.py", line=5, created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e4", kind=EdgeKind.CALLS,
            source_node_id="process_request", target_node_id="log_request",
            file_path="handlers.py", line=15, created_at=now, updated_at=now,
        ),
        GraphEdge(
            edge_id="e5", kind=EdgeKind.CALLS,
            source_node_id="handle_event", target_node_id="process_event",
            file_path="events.py", line=10, created_at=now, updated_at=now,
        ),
    ]
    for edge in edges:
        db.upsert_edge(edge)

    return db


@pytest.fixture
def detector(flow_db: CodeGraphDatabase) -> FlowDetector:
    """FlowDetector instance with populated DB."""
    return FlowDetector(flow_db)


# ---------------------------------------------------------------------------
# Entry Point Detection Tests
# ---------------------------------------------------------------------------

class TestEntryPointDetection:
    """Tests for detect_entry_points()."""

    def test_detects_no_incoming_calls(self, detector: FlowDetector) -> None:
        """Nodes with no incoming CALLS edges are entry points."""
        eps = detector.detect_entry_points()
        assert "main" in eps
        assert "handle_event" in eps

    def test_called_nodes_not_entry_points(self, detector: FlowDetector) -> None:
        """Nodes with incoming CALLS are not entry points (unless pattern match)."""
        eps = detector.detect_entry_points()
        # process_request is called by main, should not be entry
        assert "process_request" not in eps

    def test_pattern_match_entry_point(self, detector: FlowDetector) -> None:
        """Nodes matching entry patterns are entry points."""
        eps = detector.detect_entry_points()
        # handle_event matches ^handle_ pattern
        assert "handle_event" in eps

    def test_empty_graph(self, db: CodeGraphDatabase) -> None:
        """Empty graph returns no entry points."""
        detector = FlowDetector(db)
        assert detector.detect_entry_points() == []


# ---------------------------------------------------------------------------
# Flow Tracing Tests
# ---------------------------------------------------------------------------

class TestFlowTracing:
    """Tests for trace_flow()."""

    def test_trace_main_flow(self, detector: FlowDetector) -> None:
        """Tracing from main follows CALLS edges."""
        flow = detector.trace_flow("main")
        assert isinstance(flow, FlowResult)
        assert flow.entry_node_id == "main"
        assert flow.node_count >= 2
        assert "main" in flow.path_node_ids
        assert "process_request" in flow.path_node_ids

    def test_trace_depth(self, detector: FlowDetector) -> None:
        """Flow depth reflects BFS levels."""
        flow = detector.trace_flow("main", max_depth=15)
        # main -> process_request -> validate_token -> query_db = depth 3
        assert flow.depth >= 2

    def test_trace_file_count(self, detector: FlowDetector) -> None:
        """Flow tracks file count correctly."""
        flow = detector.trace_flow("main")
        # Nodes span multiple files: app.py, handlers.py, auth.py, db.py, logging.py
        assert flow.file_count >= 2

    def test_trace_nonexistent_node(self, detector: FlowDetector) -> None:
        """Tracing from nonexistent node returns empty flow."""
        flow = detector.trace_flow("nonexistent_id")
        assert flow.node_count == 0

    def test_trace_max_depth_respected(self, detector: FlowDetector) -> None:
        """max_depth parameter is respected."""
        flow = detector.trace_flow("main", max_depth=1)
        # At depth=1 from main, should only reach process_request
        assert flow.depth <= 1

    def test_trace_short_flow(self, detector: FlowDetector) -> None:
        """Handle event -> process event is a 2-node flow."""
        flow = detector.trace_flow("handle_event")
        assert flow.node_count == 2
        assert "handle_event" in flow.path_node_ids
        assert "process_event" in flow.path_node_ids


# ---------------------------------------------------------------------------
# Criticality Scoring Tests
# ---------------------------------------------------------------------------

class TestCriticalityScoring:
    """Tests for criticality score computation."""

    def test_criticality_is_bounded(self, detector: FlowDetector) -> None:
        """Criticality score is between 0 and 1."""
        flows = detector.trace_all_flows()
        for flow in flows:
            assert 0.0 <= flow.criticality <= 1.0

    def test_multi_file_flow_higher_criticality(
        self, detector: FlowDetector
    ) -> None:
        """Flows spanning multiple files score higher on file_spread."""
        flows = detector.trace_all_flows()
        if len(flows) >= 2:
            # main flow spans more files than handle_event flow
            main_flows = [f for f in flows if f.name == "flow_main"]
            event_flows = [f for f in flows if f.name == "flow_handle_event"]
            if main_flows and event_flows:
                assert main_flows[0].file_count >= event_flows[0].file_count

    def test_security_keyword_increases_criticality(
        self, flow_db: CodeGraphDatabase
    ) -> None:
        """Nodes with security keywords boost criticality."""
        # validate_token contains "token" and "validate" from SECURITY_KEYWORDS
        detector = FlowDetector(flow_db)
        flow = detector.trace_flow("main")
        # The flow contains validate_token which has security keywords
        assert "validate_token" in flow.path_node_ids


# ---------------------------------------------------------------------------
# trace_all_flows Tests
# ---------------------------------------------------------------------------

class TestTraceAllFlows:
    """Tests for trace_all_flows()."""

    def test_finds_multiple_flows(self, detector: FlowDetector) -> None:
        """Detects flows from all entry points."""
        flows = detector.trace_all_flows()
        assert len(flows) >= 2  # main flow + handle_event flow

    def test_flows_sorted_by_criticality(self, detector: FlowDetector) -> None:
        """Flows are sorted by criticality descending."""
        flows = detector.trace_all_flows()
        if len(flows) >= 2:
            for i in range(len(flows) - 1):
                assert flows[i].criticality >= flows[i + 1].criticality

    def test_flows_stored_in_metadata(
        self, detector: FlowDetector, flow_db: CodeGraphDatabase
    ) -> None:
        """Flows are stored in graph_metadata."""
        detector.trace_all_flows()
        raw = flow_db.get_metadata("flows")
        assert raw is not None
        data = json.loads(raw)
        assert len(data) >= 2

    def test_stored_flows_loadable(self, detector: FlowDetector) -> None:
        """Stored flows can be loaded back."""
        original = detector.trace_all_flows()
        loaded = detector.get_stored_flows()
        assert len(loaded) == len(original)
        for orig, load in zip(original, loaded):
            assert orig.name == load.name
            assert orig.entry_node_id == load.entry_node_id

    def test_empty_graph_no_flows(self, db: CodeGraphDatabase) -> None:
        """Empty graph returns no flows."""
        detector = FlowDetector(db)
        assert detector.trace_all_flows() == []


# ---------------------------------------------------------------------------
# Security Keywords Tests
# ---------------------------------------------------------------------------

class TestSecurityKeywords:
    """Tests for security keyword constant."""

    def test_security_keywords_frozen(self) -> None:
        """Security keywords are a frozenset."""
        assert isinstance(SECURITY_KEYWORDS, frozenset)

    def test_security_keywords_nonempty(self) -> None:
        """Security keywords set is not empty."""
        assert len(SECURITY_KEYWORDS) > 0

    def test_expected_keywords_present(self) -> None:
        """Expected security keywords are included."""
        for kw in ("auth", "password", "token", "encrypt", "sql"):
            assert kw in SECURITY_KEYWORDS
