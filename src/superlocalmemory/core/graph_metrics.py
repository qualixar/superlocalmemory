# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Whole-graph structural metrics for ``fact_importance``.

WHAT THIS IS FOR
----------------
Recall multiplies a candidate's activation by ``min(1 + pagerank * 2, 2)`` at
every hop and biases it toward the communities its query seeds belong to. Both
numbers come from ``fact_importance``. A memory missing from that table is found
by the graph walk and then ranked as though it had no position in the graph at
all -- so the table's coverage is a recall-quality property, not a reporting
nicety.

WHY ONE WRITER
--------------
Two writers were filling this table with numbers that do not mean the same
thing. The whole-graph pass (``core/graph_analyzer.py``) computes PageRank over
every fact and edge. A second path compiled an entity's facts and, on finding no
PageRank for them, computed its own over a ``LIMIT 50`` slice of facts sharing
that one entity -- a near-clique of at most 50 nodes, which hands every member
roughly ``1/n``. Measured on the author's store: the whole-graph pass produced a
maximum of 0.008744 and a median of 0.000214, while the local pass wrote 0.1 --
**eleven times the largest real score, and roughly 470x the median**. Those facts
took the maximum hop boost the formula allows (1.2 against everyone else's
1.0004) for no reason other than having shared an entity with a few others, and
they were written with no community, so the community bias could not see them
either. The signal was not stale, it was wrong.

So: one function computes this table, and it computes it over the whole graph.

WHICH ENGINE, AND WHY IT IS NOT THE GRAPH ENGINE
-----------------------------------------------
Two adapters compute the same function of the same edge set. In-process wins on
measurement, so it is the default. Timed on a copy of the author's 208,151-edge
store, each algorithm alone, warmed:

    algorithm     in process   graph engine   winner
    PageRank          0.36 s        0.38 s    tie
    Louvain           1.98 s        5.17 s    in process, 2.6x
    whole pass        2.33 s        5.56 s    in process, 2.4x

An earlier reading of 1.88 s for ``nx.pagerank`` against 0.13 s was a cold scipy
import counted as compute -- the third time a first-call warm-up has been
misread as a backend difference in this release. Both engines are deterministic
across runs (checked, not assumed) and agree to Spearman rho 0.996, so the choice
is a cost decision and may be revisited by measurement, not by argument.

The graph engine still earns its keep, just not here: reading the adjacency for
one recall measured 395 ms against SQLite's 2,477 ms on the same store, 6.3x, and
the gap widens with edge count. Those two figures were measured by hand on a copy
of a 208,151-edge store and no test reproduces them -- see the table in
``graph/cozo_adjacency`` for what was run. That is a latency win on the recall path; this
pass is background work where 2 s versus 5 s buys nothing.

Its ``pagerank()``/``community_detect()`` helpers are NOT used and could not have
been: they take their node set from the ``entity`` relation and their edges from
``edge``, which hold canonical entity IDs and fact IDs respectively.
``pagerank()`` therefore indexes an entity-keyed dict with a fact ID, raises
KeyError, and returns ``{}`` -- verified against the real store, where it
returned nothing and ``community_detect`` returned 1,386 singleton communities of
entities and not one fact. The adapter here calls the native
``PageRank``/``CommunityDetectionLouvain`` fixed rules, which operate on whatever
relation they are handed.

WHAT COUNTS AS A NODE, AND AS AN EDGE
-------------------------------------
Nodes are the profile's *visible* facts, isolated ones included: a fact with no
edges still needs a row, or the ranker treats "no position in the graph" and "not
computed yet" as the same thing. Edges come from ``iter_logical_edges``, which
already excludes any edge with a withheld or soft-deleted endpoint -- the same
predicate the retrieval channel prunes its adjacency with. The previous
whole-graph pass read ``graph_edges`` and ``atomic_facts`` raw, so it ranked
1,299 unreturnable facts alongside real memories and diluted every real score.

PageRank runs on the edge-connected subgraph and the isolated facts are then
given the uniform teleport share ``(1 - damping) / N`` over the full node set,
with the connected scores scaled to leave room for it. That convention lives
here, in the port, so both engines produce numbers on one scale -- the ranker
reads absolute values, so an engine swap that shifted the scale would silently
re-tune every boost in the system.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from superlocalmemory.storage.logical_edges import iter_logical_edges

logger = logging.getLogger(__name__)


@contextmanager
def _short_connection(db: Any) -> Any:
    """A connection held for one read or one write, never across the compute.

    ``raw_connection`` takes the manager's write lock, so holding it around a
    two-to-ten-second graph computation would stall every store for that long.
    The pass therefore reads, releases, computes, and writes.

    The consolidation cycle passes a minimal proxy that owns a bare sqlite3
    connection and exposes ``execute`` only. Supporting it here is what keeps
    this the single writer of the table; the alternative was a second
    implementation for that one caller, which is how the two disagreeing
    PageRanks happened in the first place.
    """
    # Resolved on the TYPE, and the fallback is checked with isinstance, for the
    # reason written up in retrieval/scope_policy.py: a MagicMock fabricates any
    # attribute you ask for, so ``getattr(db, "raw_connection")`` returns a
    # callable whose context manager yields another MagicMock -- and iterating
    # that raises, which this module's error path then reports as "could not read
    # the graph" for a store that is perfectly readable. The suite passes exactly
    # such a mock, and it cost a debugging round to notice the pass had not run
    # rather than found nothing.
    raw = getattr(type(db), "raw_connection", None)
    if callable(raw):
        with raw(db) as conn:
            yield conn
        return
    conn = getattr(db, "_conn", None)
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError(
            f"{type(db).__name__} exposes no connection to compute metrics on"
        )
    yield conn
    conn.commit()

@contextmanager
def _read_connection(db: Any) -> Any:
    """A connection for a READ, which never takes the store's write lock.

    ``_short_connection`` goes through ``raw_connection``, which takes the
    process-wide write lock even to read. Reading every edge of a large store
    that way held it for 60 s on a 22k-fact store (4.1.21): saves and edits
    failed and enrichment lost its lease meanwhile. A ``DatabaseManager``
    store is read through a query-only snapshot instead; the consolidation
    proxy (a bare connection) keeps its own connection.
    """
    if callable(getattr(type(db), "raw_connection", None)) and getattr(db, "db_path", None):
        from superlocalmemory.storage.read_connection import read_only_snapshot

        with read_only_snapshot(db) as conn:
            yield conn
        return
    with _short_connection(db) as conn:
        yield conn


#: Rows written per transaction, so the write lock is released between them
#: and an interactive save never waits behind the whole table.
_WRITE_CHUNK = 500

#: PageRank damping. Matches the previous whole-graph pass so scores stay
#: comparable across the version that introduced this module.
DEFAULT_DAMPING = 0.85

#: Betweenness centrality is O(V*E). At 4k facts and 130k edges that is minutes,
#: and at the 12k/208k store it is hours -- which is the likeliest reason the
#: whole-graph pass had run exactly once in nine days. Nothing in retrieval reads
#: ``bridge_score``; only the dashboard displays it. So it is computed under a
#: node ceiling and reported as skipped above it, rather than being the reason
#: PageRank never lands.
BRIDGE_NODE_LIMIT = 1500

#: Louvain returns a hierarchy of partitions per node. Level 0 is the coarsest,
#: and on the author's store it yields 13 communities against the 11
#: meaningfully-sized ones the previous pass found -- so the community bias keeps
#: the granularity it was tuned against instead of being handed 210 fragments.
LOUVAIN_LEVEL = 0

#: A projection is only trustworthy if it agrees with the store. If Cozo's edge
#: count for this profile differs from SQLite's by more than this fraction, the
#: pass computes on SQLite instead and says so, rather than ranking the store's
#: memories from a graph the store does not have.
MAX_PROJECTION_DRIFT = 0.02


@dataclass(frozen=True)
class GraphMetricsReport:
    """What one pass actually did. No field here is inferred from another."""

    profile_id: str
    engine: str = "none"
    facts: int = 0
    edges: int = 0
    connected: int = 0
    isolated: int = 0
    communities: int = 0
    written: int = 0
    removed: int = 0
    bridges_computed: bool = False
    duration_ms: int = 0
    error: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        if self.error:
            return f"graph metrics for {self.profile_id} FAILED: {self.error}"
        return (
            f"graph metrics for {self.profile_id}: {self.written} facts "
            f"({self.connected} connected, {self.isolated} isolated), "
            f"{self.edges} edges, {self.communities} communities, "
            f"engine={self.engine}, {self.duration_ms} ms"
        )


# ----------------------------------------------------------------------
# Engines. Each returns metrics for the edge-connected nodes only; the
# isolated-node convention belongs to the caller, once, for both.
# ----------------------------------------------------------------------


def _cozo_metrics(
    backend: Any, profile_id: str, damping: float
) -> tuple[dict[str, float], dict[str, int]]:
    """PageRank and Louvain from Cozo's native fixed rules.

    Both queries filter on ``profile_id``: the projection holds every profile's
    edges in one relation, and an unfiltered rule would rank one profile's
    memories using another's graph.
    """
    client = backend._db  # the module-private client wrapper; no public accessor
    pr_rows = client.run(
        "rel[a, b, w] := *edge{from_id: a, to_id: b, weight: w, "
        "                      profile_id: $pid}\n"
        "?[node, score] <~ PageRank(rel[a, b, w], theta: $theta)",
        {"pid": profile_id, "theta": damping},
    )
    pagerank = {str(r[0]): float(r[1]) for r in pr_rows.values.tolist()}

    comm_rows = client.run(
        "rel[a, b, w] := *edge{from_id: a, to_id: b, weight: w, "
        "                      profile_id: $pid}\n"
        "?[grp, node] <~ CommunityDetectionLouvain(rel[a, b, w])",
        {"pid": profile_id},
    )
    communities: dict[str, int] = {}
    for grp, node in comm_rows.values.tolist():
        label = grp[LOUVAIN_LEVEL] if isinstance(grp, (list, tuple)) else grp
        communities[str(node)] = int(label)
    return pagerank, communities


def _networkx_metrics(
    edges: list[tuple[str, str, float]], damping: float
) -> tuple[dict[str, float], dict[str, int]]:
    """The same two metrics without a graph projection to read."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    digraph = nx.DiGraph()
    for source, target, weight in edges:
        if digraph.has_edge(source, target):
            if weight > digraph[source][target].get("weight", 0.0):
                digraph[source][target]["weight"] = weight
        else:
            digraph.add_edge(source, target, weight=weight)
    if digraph.number_of_nodes() == 0:
        return {}, {}

    pagerank = nx.pagerank(digraph, alpha=damping, weight="weight")
    communities: dict[str, int] = {}
    try:
        partitions = louvain_communities(
            digraph.to_undirected(), weight="weight", seed=42,
        )
        for label, members in enumerate(partitions):
            for node in members:
                communities[str(node)] = label
    except Exception as exc:  # noqa: BLE001 -- a partition is optional, a score is not
        logger.debug("Louvain partition unavailable: %s", exc)
    return pagerank, communities


def _bridge_scores(
    edges: list[tuple[str, str, float]], node_count: int
) -> dict[str, float] | None:
    """Sampled betweenness, or None when the graph is too big to afford it."""
    if node_count > BRIDGE_NODE_LIMIT or node_count <= 2:
        return None
    try:
        import networkx as nx

        graph = nx.DiGraph()
        for source, target, weight in edges:
            graph.add_edge(source, target, weight=weight)
        return nx.betweenness_centrality(graph, weight="weight", normalized=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Bridge scores unavailable: %s", exc)
        return None


# ----------------------------------------------------------------------
# The pass
# ----------------------------------------------------------------------


def _visible_fact_ids(conn: sqlite3.Connection, profile_id: str) -> list[str]:
    from superlocalmemory.storage.database import (
        visible_fact_clause_for_connection,
    )

    clause = visible_fact_clause_for_connection(conn)
    rows = conn.execute(
        f"SELECT fact_id FROM atomic_facts WHERE profile_id = ?{clause}",
        (profile_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _projection_usable(
    backend: Any, profile_id: str, sqlite_edges: int
) -> tuple[bool, str]:
    """Whether Cozo's edge set is close enough to the store's to rank from."""
    if backend is None:
        return False, "no graph projection"
    try:
        rows = backend._db.run(
            "?[count(a)] := *edge{from_id: a, profile_id: $pid}",
            {"pid": profile_id},
        )
        projected = int(rows.values.tolist()[0][0]) if len(rows) else 0
    except Exception as exc:  # noqa: BLE001
        return False, f"projection unreadable: {exc}"
    if sqlite_edges == 0:
        return projected == 0, "both empty"
    drift = abs(projected - sqlite_edges) / float(sqlite_edges)
    if drift > MAX_PROJECTION_DRIFT:
        return False, (
            f"projection drifted {drift:.1%} "
            f"({projected} projected vs {sqlite_edges} stored)"
        )
    return True, f"projection within {drift:.2%}"


def _projection_is_current(db: Any, profile_id: str) -> bool:
    """Whether every change SQLite has recorded has reached the projection.

    Two engines ranking two different graphs is worse than one engine ranking
    the right one, because the difference is invisible in the output: both
    return a full table of plausible scores. The queue is the only place that
    records a change the projection has not seen yet, so an outstanding row
    means the projection is a graph the store no longer has.

    Absent queue means an older store that never had one; there is nothing to
    be behind on, so the projection is as current as it can be.
    """
    try:
        from superlocalmemory.storage import projection_outbox

        if not projection_outbox.is_available(db):
            return True
        with _read_connection(db) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM projection_outbox WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
        return int(row[0] if row else 0) == 0
    except Exception as exc:  # noqa: BLE001 -- unreadable means do not trust it
        logger.debug("graph metrics: cannot check projection currency: %s", exc)
        return False


def read_graph(conn: sqlite3.Connection, profile_id: str) -> tuple[list[str], list[tuple[str, str, float]]]:
    """The profile's visible fact ids and logical edges, in deterministic order."""
    nodes = _visible_fact_ids(conn, profile_id)
    edges = [
        (str(source), str(target), float(weight))
        for source, target, _etype, weight, _pid
        in iter_logical_edges(conn, profile_id)
    ]
    return nodes, edges


def metrics_rows(profile_id: str, nodes: list[str], edges: list[tuple[str, str, float]],
                 damping: float, pagerank: dict[str, float],
                 communities: dict[str, int]) -> dict[str, Any]:
    """``fact_importance`` rows from the engine's output; identical wherever it runs."""
    notes: list[str] = []
    # An engine only sees nodes that appear in an edge. Everything else is a
    # visible fact with no graph position, and it gets the teleport share -- the
    # value PageRank would give a node nothing links to.
    node_set = set(nodes)
    total = len(node_set)
    base = (1.0 - damping) / float(total)
    connected = {fid for fid in pagerank if fid in node_set}
    isolated = node_set - connected
    # Leave room for the isolated mass so the whole table still sums to ~1 and
    # the ranker's absolute thresholds keep meaning what they meant.
    headroom = max(0.0, 1.0 - base * len(isolated))
    connected_mass = sum(pagerank[fid] for fid in connected) or 1.0

    degree: dict[str, int] = {}
    for source, target, _weight in edges:
        degree[source] = degree.get(source, 0) + 1
        degree[target] = degree.get(target, 0) + 1
    divisor = float(total - 1) if total > 1 else 1.0

    bridges = _bridge_scores(edges, total)
    if bridges is None:
        notes.append(f"bridge scores skipped above {BRIDGE_NODE_LIMIT} facts")

    rows: list[tuple[Any, ...]] = []
    for fact_id in nodes:
        if fact_id in connected:
            score = pagerank[fact_id] / connected_mass * headroom
        else:
            score = base
        community = communities.get(fact_id)
        rows.append((
            fact_id,
            profile_id,
            round(float(score), 9),
            int(community) if community is not None else None,
            round(degree.get(fact_id, 0) / divisor, 6),
            round(float((bridges or {}).get(fact_id, 0.0)), 6),
        ))

    return {
        "rows": rows, "facts": total, "edges": len(edges), "connected": len(connected),
        "isolated": len(isolated), "communities": len({c for c in communities.values()}),
        "bridges_computed": bridges is not None, "notes": notes,
    }


def compute_graph_metrics(
    db: Any,
    profile_id: str,
    *,
    backend: Any = None,
    damping: float = DEFAULT_DAMPING,
    prefer: str = "networkx",
    isolate: bool = False,
) -> GraphMetricsReport:
    """Recompute ``fact_importance`` for one profile. Returns what it did.

    ``backend`` is a live graph projection to compute on and ``prefer`` selects
    the engine; the projection is used only when both point at it AND it agrees
    with the store. Errors are returned in the report rather than swallowed: a
    pass that writes nothing and a store with no facts are different events, and
    the previous implementation reported both as ``node_count: 0``.
    """
    # One pass per profile and store at a time: a pass that read the graph
    # earlier must not finish after a newer one and delete the rows that pass
    # wrote for facts stored in between. A second pass waits, then reads afresh.
    with _pass_lock(db, profile_id):
        return _compute_graph_metrics(db, profile_id, backend=backend, damping=damping,
                                      prefer=prefer, isolate=isolate)


_PASS_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_PASS_LOCKS_GUARD = threading.Lock()


def _pass_lock(db: Any, profile_id: str) -> threading.Lock:
    path = getattr(db, "db_path", None) or getattr(db, "_db_path", None) or id(db)
    key = (str(path), str(profile_id))
    with _PASS_LOCKS_GUARD:
        return _PASS_LOCKS.setdefault(key, threading.Lock())


def _compute_graph_metrics(db: Any, profile_id: str, *, backend: Any, damping: float,
                           prefer: str, isolate: bool) -> GraphMetricsReport:
    started = time.monotonic()
    notes: list[str] = []
    if isolate and prefer != "cozo":
        # The whole computation in a separate low-priority process
        # (core/graph_metrics_process.py); this process only writes the rows.
        from superlocalmemory.core.graph_metrics_process import compute_rows_in_child

        return _store_computed(db, profile_id, started, notes,
                               compute_rows_in_child(db, profile_id, damping))
    try:
        with _read_connection(db) as conn:
            nodes, edges = read_graph(conn, profile_id)
    except Exception as exc:  # noqa: BLE001
        return GraphMetricsReport(
            profile_id=profile_id,
            error=f"could not read the graph: {exc}",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if not nodes:
        return GraphMetricsReport(
            profile_id=profile_id,
            engine="none",
            duration_ms=int((time.monotonic() - started) * 1000),
            notes=("no visible facts",),
        )

    usable, why = _projection_usable(backend, profile_id, len(edges))
    notes.append(why)
    if usable and not _projection_is_current(db, profile_id):
        usable = False
        why = "projection has unapplied changes"
        notes.append(why)
    engine = "cozo" if (usable and prefer == "cozo") else "networkx"
    try:
        # On ``engine``, not on ``usable``. Branching on ``usable`` ran the
        # projection whenever one was available, whatever the caller asked for,
        # and then reported the engine the caller had asked for -- so a store
        # with a projection was ranked by it while every log line said
        # otherwise, and the fallback below could never be reached.
        if engine == "cozo":
            pagerank, communities = _cozo_metrics(backend, profile_id, damping)
        else:
            pagerank, communities = _networkx_metrics(edges, damping)
    except Exception as exc:  # noqa: BLE001
        if engine == "cozo":
            notes.append(f"cozo failed, fell back: {exc}")
            engine = "networkx"
            try:
                pagerank, communities = _networkx_metrics(edges, damping)
            except Exception as inner:  # noqa: BLE001
                return GraphMetricsReport(
                    profile_id=profile_id, engine="networkx",
                    facts=len(nodes), edges=len(edges),
                    error=f"both engines failed: {exc} / {inner}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    notes=tuple(notes),
                )
        else:
            return GraphMetricsReport(
                profile_id=profile_id, engine=engine,
                facts=len(nodes), edges=len(edges),
                error=f"metrics failed: {exc}",
                duration_ms=int((time.monotonic() - started) * 1000),
                notes=tuple(notes),
            )

    computed = metrics_rows(profile_id, nodes, edges, damping, pagerank, communities)
    computed["engine"] = engine
    return _store_computed(db, profile_id, started, notes, computed)


def _store_computed(db: Any, profile_id: str, started: float, notes: list[str],
                    computed: dict[str, Any]) -> GraphMetricsReport:
    """Write rows computed here or in the child process; the report either way."""
    if computed.get("empty"):  # nothing visible: nothing written, as before
        return GraphMetricsReport(
            profile_id=profile_id, engine="none",
            duration_ms=int((time.monotonic() - started) * 1000),
            notes=("no visible facts",),
        )
    if computed.get("error"):
        return GraphMetricsReport(
            profile_id=profile_id, engine=computed.get("engine", "networkx"),
            facts=computed.get("facts", 0), edges=computed.get("edges", 0),
            error=computed["error"],
            duration_ms=int((time.monotonic() - started) * 1000),
            notes=tuple(notes) + tuple(computed.get("notes", ())),
        )
    notes = notes + list(computed.get("notes", ()))
    rows = computed["rows"]
    engine, total = computed["engine"], computed["facts"]
    edges_n, connected_n, isolated_n = computed["edges"], computed["connected"], computed["isolated"]
    try:
        removed = _write(db, profile_id, rows)
    except Exception as exc:  # noqa: BLE001
        return GraphMetricsReport(
            profile_id=profile_id, engine=engine, facts=total,
            edges=edges_n, connected=connected_n,
            isolated=isolated_n,
            error=f"metrics computed but not stored: {exc}",
            duration_ms=int((time.monotonic() - started) * 1000),
            notes=tuple(notes),
        )

    return GraphMetricsReport(
        profile_id=profile_id,
        engine=engine,
        facts=total,
        edges=edges_n,
        connected=connected_n,
        isolated=isolated_n,
        communities=computed["communities"],
        written=len(rows),
        removed=removed,
        bridges_computed=computed["bridges_computed"],
        duration_ms=int((time.monotonic() - started) * 1000),
        notes=tuple(notes),
    )


#: Longest one chunk waits for in-flight recalls, so steady recall traffic slows
#: the write-back but can never stop it.
_RECALL_YIELD_MAX_SECONDS = 30.0


def _yield_to_recalls() -> None:
    """Wait while a person's recall runs (bounded). Writing 22k rows beside a
    recall stretched it to 6.1 s on a 22k-fact store (sampled); the rows are a
    background projection and can land a moment later."""
    from superlocalmemory.core import recall_gate

    deadline = time.monotonic() + _RECALL_YIELD_MAX_SECONDS
    while recall_gate.in_flight() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)


def _write(db: Any, profile_id: str, rows: list[tuple[Any, ...]]) -> int:
    """Replace this profile's rows, a bounded chunk per transaction.

    Deleting what is no longer visible is what makes the table a projection
    rather than an accumulation: a fact that has since been withheld or erased
    must lose its row, or the ranker keeps scoring something recall will never
    return.

    One transaction for the whole table held the store's write lock for 2.3 s
    on a 22k-fact store - longer than an edit's 2 s budget, so edits failed
    whenever a pass landed. Each chunk now commits on its own. Rows are
    upserted, never deleted and re-inserted, so between chunks a recall sees
    some scores from this pass and some from the last one, never a profile
    with no metrics at all.
    """
    with _read_connection(db) as conn:
        existing = {
            str(r[0]) for r in conn.execute(
                "SELECT fact_id FROM fact_importance WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
        }
    stale = sorted(existing - {row[0] for row in rows})
    with _short_connection(db) as conn:
        _ensure_bridge_column(conn)
    for index in range(0, len(stale), 800):
        _yield_to_recalls()  # a delete takes the same write lock as an insert
        chunk = stale[index:index + 800]
        placeholders = ",".join("?" for _ in chunk)
        with _short_connection(db) as conn:
            conn.execute(
                f"DELETE FROM fact_importance WHERE profile_id = ? "
                f"AND fact_id IN ({placeholders})",
                (profile_id, *chunk),
            )
    for index in range(0, len(rows), _WRITE_CHUNK):
        _yield_to_recalls()
        with _short_connection(db) as conn:
            conn.executemany(
                "INSERT INTO fact_importance "
                "(fact_id, profile_id, pagerank_score, community_id, "
                " degree_centrality, bridge_score, computed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(fact_id) DO UPDATE SET "
                "  profile_id = excluded.profile_id, "
                "  pagerank_score = excluded.pagerank_score, "
                "  community_id = excluded.community_id, "
                "  degree_centrality = excluded.degree_centrality, "
                "  bridge_score = excluded.bridge_score, "
                "  computed_at = excluded.computed_at",
                rows[index:index + _WRITE_CHUNK],
            )
    return len(stale)


def _ensure_bridge_column(conn: sqlite3.Connection) -> None:
    """Idempotent: ``bridge_score`` arrived after the table did.

    A store created before it exists in the wild, so the insert below cannot
    assume the column. Adding it here rather than refusing keeps an upgrade from
    silently losing its metrics on first run.
    """
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(fact_importance)")}
        if "bridge_score" not in columns:
            conn.execute(
                "ALTER TABLE fact_importance ADD COLUMN bridge_score REAL DEFAULT 0.0"
            )
    except sqlite3.Error as exc:
        logger.debug("bridge_score column check failed: %s", exc)


def metrics_are_stale(db: Any, profile_id: str) -> tuple[bool, str]:
    """Whether this profile's metrics no longer describe its graph.

    Deliberately not a clock. "Recomputed 30 minutes ago" says nothing about
    whether the store changed, and the failure this guards against is a memory
    that has no row at all -- which the ranker cannot distinguish from a memory
    with no graph position. So the test is coverage: is any visible fact
    missing, or does the table describe facts that are gone.

    Cheap enough to run every cycle: two counting queries against an indexed
    column.
    """
    try:
        with _read_connection(db) as conn:
            from superlocalmemory.storage.database import (
                visible_fact_clause_for_connection,
            )

            clause = visible_fact_clause_for_connection(conn, prefix="f")
            missing = conn.execute(
                "SELECT COUNT(*) FROM atomic_facts f "
                "LEFT JOIN fact_importance fi ON fi.fact_id = f.fact_id "
                f"WHERE f.profile_id = ?{clause} AND fi.fact_id IS NULL",
                (profile_id,),
            ).fetchone()[0]
            if missing:
                return True, f"{missing} visible fact(s) have no metrics"
            surplus = conn.execute(
                "SELECT COUNT(*) FROM fact_importance fi "
                "LEFT JOIN atomic_facts f ON f.fact_id = fi.fact_id "
                f"WHERE fi.profile_id = ? AND (f.fact_id IS NULL OR NOT (1=1{clause}))",
                (profile_id,),
            ).fetchone()[0]
            if surplus:
                return True, f"{surplus} metric row(s) describe facts recall cannot return"
        return False, "metrics cover the visible graph"
    except Exception as exc:  # noqa: BLE001 -- a failed check must not skip the pass
        return True, f"staleness check failed ({exc}); recomputing"
