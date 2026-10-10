# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The worker that carries queued facts into CozoDB and LanceDB.

One writer, one direction: SQLite is canonical, the projections are derived, and
this is the only thing that writes them during normal operation. Making it the
sole writer is what removes the class of bug this replaced — a projection write
attempted inline on the store path, on whichever thread happened to be storing,
with its failure swallowed into a debug log.

WHAT IT PROJECTS IS WHAT RECALL CAN RETURN
------------------------------------------
Not "lifecycle in (active, warm)", which is what the inline sync used. That
predicate was wrong twice over: it dropped ``cold``, which is a live tier that
recall answers from, and it never withdrew a fact that had since been archived,
leaving forgotten memories offered as candidates.

The filter is ``visible_fact_clause()`` — the same predicate every read path
uses to decide whether a row may be shown as a memory at all. Deriving the
projection from that clause means the two cannot drift.

BUT VISIBILITY GOVERNS CANDIDACY, NOT ADJACENCY
-----------------------------------------------
The SQLite channel does not treat those as one thing, and neither may this. Its
entity map filters on visibility — a withheld row must never enter it, because
it carries its whole cluster's pooled entity list and out-ranks real memories.
Its edge walk filters on scope alone, with no visibility predicate, so it
traverses edges into withheld and archived facts and lets hydration drop them at
the end.

So a hidden fact loses its entity bridge and its vector and KEEPS its edges. An
earlier version of this module deleted the edges too, and on a real store that
left 32 visible facts with a smaller adjacency here than in SQLite — every
missing endpoint quarantined. A graph that answers differently from the walk it
replaces is not a projection of it.

FAILURE IS LOUD
---------------
A projection write that raises leaves its row queued with the attempt counted
and the error recorded. Nothing is dropped, and the queue depth is a health
metric, so a projection that has stopped keeping up is visible instead of
silent. That is the entire point of the mechanism and it is why nothing in this
module catches an exception and continues as though it had not happened.

AFTER A BULK IMPORT, THE QUEUE IS NOT CLEARED — IT IS DRAINED
-------------------------------------------------------------
A promotion builds the whole projection from SQLite in one pass, which satisfies
every row queued before it started. Deleting those rows on that basis would need
a watermark: a timestamp taken before the import read its snapshot, with
everything older discarded. That is one off-by-one away from throwing out a
projection nobody will ever write again, in the exact mechanism whose only job
is to make that impossible.

So nothing is discarded. The worker re-projects the backlog, which is idempotent
and lands on the same graph. It costs the import's work once more, in the
background, off the hot path — a price worth paying for a rule with no edge case
in it. Queue depth right after a promotion is therefore high and falling, which
is the truth; ``stalled`` is the number that indicates trouble.

A ROW THAT KEEPS FAILING MUST NOT BLOCK THE ONES BEHIND IT
----------------------------------------------------------
The queue is claimed in ``attempts, revision`` order, so a fact whose
projection is genuinely impossible — a malformed embedding, an id Cozo refuses
— sinks to the back after its first failure and healthy work continues past it.
Ordering by revision alone would let one poisoned row starve every fact behind
it, which is the failure mode that makes queues look like outages.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from superlocalmemory.storage import projection_outbox

logger = logging.getLogger(__name__)

#: Rows per pass. Large enough that a backlog clears in few passes, small enough
#: that a pass cannot hold the drain thread past a shutdown request for long.
DEFAULT_BATCH = 200

#: How long the worker waits before looking again when nothing woke it. A store
#: signals the worker directly, so this is only the safety net for an enqueue
#: that arrived from another process — a CLI write, or a second daemon.
IDLE_INTERVAL_SECONDS = 2.0

#: Attempts after which a row is reported at warning level rather than debug.
#: It keeps being retried; the change is that it stops being quiet about it.
LOUD_AFTER_ATTEMPTS = 3


@dataclass
class DrainResult:
    """What one pass did. Every field is a number an operator can act on."""

    projected: int = 0
    removed: int = 0
    failed: int = 0
    skipped: int = 0
    superseded: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def handled(self) -> int:
        return self.projected + self.removed + self.skipped + self.superseded

    def as_dict(self) -> dict[str, Any]:
        return {
            "projected": self.projected,
            "removed": self.removed,
            "failed": self.failed,
            "skipped": self.skipped,
            "superseded": self.superseded,
        }


class _VectorBatch:
    """Collects a drain pass's vector writes so it creates ONE version, not N.

    ``add_vectors`` issues a LanceDB ``merge_insert``, a whole-table operation
    that creates a new version of the store. It used to be called once per
    fact, so a store accumulated one version per memory with nothing removing
    them: 5,561 memories produced a 17 GB vector store holding 50,580 versions,
    and every vector operation then had to walk that history. GitHub #137.

    Splits on ``profile_id`` because ``add_vectors`` takes exactly one, and a
    batch that merged two tenants' vectors under one id would file a memory
    against the wrong owner.
    """

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[tuple[str, object, str, str]] = []

    @property
    def pending(self) -> int:
        return len(self._items)

    def add(self, fact_id: str, embedding: object, tier: str,
            profile_id: str) -> None:
        self._items.append((fact_id, embedding, tier, profile_id))

    def flush(self, vector: "Any | None") -> None:
        """Write everything collected. Clears the buffer even with no backend.

        Raises whatever the backend raises: the caller must not clear an outbox
        row for a vector that did not land.
        """
        items, self._items = self._items, []
        if not items or vector is None:
            return
        grouped: dict[str, tuple[list, list, list]] = {}
        for fact_id, embedding, tier, profile_id in items:
            ids, embeddings, tiers = grouped.setdefault(
                profile_id, ([], [], []),
            )
            ids.append(fact_id)
            embeddings.append(embedding)
            tiers.append(tier)
        for profile_id, (ids, embeddings, tiers) in grouped.items():
            vector.add_vectors(ids, embeddings, tiers, profile_id)


def _should_keep_draining(handled: int, failed: int, batch: int) -> bool:
    """Whether a drain pass earned another immediate pass.

    A full batch used to be enough, counting failures as activity. A batch of
    permanently failing rows satisfies that with ``handled == 0``, and the
    outbox retires nothing on failure -- ``claim_batch`` has no WHERE and
    ``record_failure`` only increments a counter -- so the same rows came back
    every pass, forever, with no wait between them. GitHub #137.

    Progress, not activity: at least one row must have cleared.
    """
    return handled > 0 and (handled + failed) >= batch


class ProjectionDrain:
    """Applies queued facts to the graph and vector projections.

    Takes accessors rather than backends so it always sees the current ones: a
    promotion or a rollback swaps them underneath a long-lived worker, and a
    reference captured at construction would keep writing into the store that
    was just replaced.
    """

    def __init__(
        self,
        db: Any,
        graph_backend: Callable[[], Any],
        vector_backend: Callable[[], Any],
    ) -> None:
        self._db = db
        self._graph = graph_backend
        self._vector = vector_backend
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pass_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Begin draining in the background. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="slm-projection-drain", daemon=True,
        )
        self._thread.start()
        logger.info("projection drain started")
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to finish its pass and exit."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def has_target(self) -> bool:
        """Whether a graph or vector projection is open to drain into.

        When neither is (Local Core, the default) a pass touches no row, so a
        running worker is not the same as a queue that is being drained.
        """
        return self._graph() is not None or self._vector() is not None

    def notify(self) -> None:
        """Tell the worker there is something to do.

        Called after a store commits. Cheap enough to call on every write, and
        it is what keeps the gap between "remembered" and "in the graph" at
        milliseconds instead of the idle interval.
        """
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=IDLE_INTERVAL_SECONDS)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                # Keep going while a pass is filling its batch: a backlog
                # should drain continuously rather than one batch per tick.
                while not self._stop.is_set():
                    result = self.drain_once()
                    if not _should_keep_draining(
                        result.handled, result.failed, DEFAULT_BATCH,
                    ):
                        break
            except Exception as exc:  # pragma: no cover - worker must not die
                # A worker that exits on an unexpected error would leave the
                # queue growing with nothing draining it and no thread left to
                # report why. Log it and stay alive; the rows are still queued.
                logger.error("projection drain pass failed: %s", exc, exc_info=True)
        logger.info("projection drain stopped")

    # ------------------------------------------------------------------
    # One pass
    # ------------------------------------------------------------------

    def drain_once(self, limit: int = DEFAULT_BATCH) -> DrainResult:
        """Apply up to ``limit`` queued facts. Safe to call directly.

        Returns without touching a row when no projection is open. The rows are
        the pending work for whenever one is, and discarding them would be
        throwing away the only record of which facts still need projecting.
        """
        result = DrainResult()
        graph, vector = self._graph(), self._vector()
        if graph is None and vector is None:
            return result

        with self._pass_lock:
            batch = _VectorBatch()
            staged: list[tuple[dict[str, Any], str]] = []
            for row in projection_outbox.claim_batch(self._db, limit=limit):
                outcome = self._apply_row(row, graph, vector, result, batch)
                if outcome is not None:
                    staged.append((row, outcome))
            # The vectors must be durable BEFORE any outbox row is cleared.
            # Clearing first and writing after would lose a pass's embeddings
            # to a crash with nothing left queued to notice.
            try:
                batch.flush(vector)
            except Exception as exc:  # noqa: BLE001
                # The whole batch failed together, so it fails together. Graph
                # projection is idempotent, so the retry re-does it harmlessly.
                for row, _ in staged:
                    attempts = projection_outbox.record_failure(
                        self._db, row["fact_id"], str(exc),
                    )
                    result.failed += 1
                    result.errors.append(f"{row['fact_id'][:12]}: {exc}")
                    log = (logger.warning if attempts >= LOUD_AFTER_ATTEMPTS
                           else logger.debug)
                    log("vector batch failed for %s after %d attempt(s): %s",
                        row["fact_id"][:12], attempts, exc)
                staged = []
            for row, outcome in staged:
                self._resolve(row, outcome, result)
        return result

    def _apply_row(
        self, row: dict[str, Any], graph: Any, vector: Any,
        result: DrainResult, batch: "_VectorBatch",
    ) -> "str | None":
        """Project one fact. Returns its outcome, or None if it failed.

        The outcome is returned rather than resolved here: the pass's vector
        writes are still buffered at this point, and clearing an outbox row
        before its vector is durable would lose the embedding to a crash.
        """
        fact_id = row["fact_id"]
        try:
            if row["op"] == projection_outbox.OP_DELETE:
                self._remove(fact_id, graph, vector)
                return "removed"
            return self._project(fact_id, graph, vector, batch)
        except Exception as exc:
            attempts = projection_outbox.record_failure(self._db, fact_id, str(exc))
            result.failed += 1
            result.errors.append(f"{fact_id[:12]}: {exc}")
            log = logger.warning if attempts >= LOUD_AFTER_ATTEMPTS else logger.debug
            log(
                "projection failed for %s after %d attempt(s): %s",
                fact_id[:12], attempts, exc,
            )
            return None

    def _resolve(
        self, row: dict[str, Any], outcome: str, result: DrainResult,
    ) -> None:
        """Clear one outbox row now that its projections are durable."""
        if projection_outbox.resolve(self._db, row["fact_id"], row["revision"]):
            setattr(result, outcome, getattr(result, outcome) + 1)
        else:
            # The fact was written again while this projection was in flight,
            # so a newer intent is queued. Counting it as done would report
            # work that still has to happen.
            result.superseded += 1

    # ------------------------------------------------------------------
    # The projections themselves
    # ------------------------------------------------------------------

    def _project(self, fact_id: str, graph: Any, vector: Any,
                 batch: "_VectorBatch") -> str:
        """Bring one fact's projection up to date. Returns the outcome name."""
        state = self._visibility(fact_id)
        if state == "absent":
            # An entity id, or a fact hard-deleted since it was queued. There
            # is nothing to project and nothing to remove.
            return "skipped"
        if state == "hidden":
            # Archived or withheld. It must stop being offered as a candidate —
            # but its EDGES stay. The SQLite entity map filters on visibility
            # and its edge walk does not, so deleting a hidden fact's edges
            # would leave every visible fact that neighboured it with a smaller
            # adjacency here than in SQLite, and the two graphs would answer
            # differently. Measured on a real store before this was split: 32
            # visible facts had lost edges, every missing endpoint quarantined.
            self._remove(fact_id, graph, vector)
            return "removed"

        fact = self._db.get_fact(fact_id)
        if fact is None:
            return "skipped"
        if graph is not None:
            self._project_graph(fact, graph)
        if vector is not None:
            self._project_vector(fact, batch)
        return "projected"

    def _visibility(self, fact_id: str) -> str:
        """``visible``, ``hidden`` or ``absent`` for one id."""
        rows = self._db.execute(
            "SELECT 1 FROM atomic_facts WHERE fact_id = ?", (fact_id,),
        )
        if not rows:
            return "absent"
        visible = self._db.execute(
            "SELECT 1 FROM atomic_facts WHERE fact_id = ?"
            + self._db.visible_fact_clause(),
            (fact_id,),
        )
        return "visible" if visible else "hidden"

    def _project_graph(self, fact: Any, graph: Any) -> None:
        """Replace this fact's node, entity bridge and edges in the graph.

        ``remove_fact`` first, so a re-projection cannot leave an entity link
        or an edge that SQLite no longer has. Replace-then-write is what makes
        a replay idempotent.
        """
        profile_id = getattr(fact, "profile_id", "default") or "default"
        graph.remove_fact(fact.fact_id)
        entities = list(getattr(fact, "canonical_entities", []) or [])
        for entity_id in entities:
            rows = self._db.execute(
                "SELECT canonical_name, entity_type, fact_count FROM canonical_entities "
                "WHERE entity_id = ? AND profile_id = ?",
                (entity_id, profile_id),
            )
            if not rows:
                continue
            entity = dict(rows[0])
            graph.add_entity(
                entity_id,
                entity.get("canonical_name") or entity_id,
                entity.get("entity_type") or "concept",
                {"fact_count": int(entity.get("fact_count") or 0)},
                profile_id,
            )
        graph.add_fact_entities(fact.fact_id, entities, profile_id)
        for row in self._db.execute(
            "SELECT source_id, target_id, edge_type, weight FROM graph_edges "
            "WHERE profile_id = ? AND (source_id = ? OR target_id = ?)",
            (profile_id, fact.fact_id, fact.fact_id),
        ):
            edge = dict(row)
            graph.add_edge(
                edge["source_id"], edge["target_id"],
                edge.get("edge_type") or "related",
                float(edge.get("weight") or 1.0), profile_id=profile_id,
            )

    def _project_vector(self, fact: Any, batch: "_VectorBatch") -> None:
        """Buffer this fact's embedding for the pass's single vector write.

        A fact with no embedding yet is not an error: ingestion is
        queryable-first, so the vector arrives with enrichment and the update
        that writes it queues the fact again.

        Buffered rather than written: one ``add_vectors`` per fact is one
        LanceDB version per fact, which is how a 5,561-memory store came to
        hold 50,580 versions in 17 GB. GitHub #137.
        """
        embedding = getattr(fact, "embedding", None)
        if not embedding:
            return
        lifecycle = getattr(fact, "lifecycle", None)
        tier = getattr(lifecycle, "value", lifecycle) or "active"
        batch.add(
            fact.fact_id, embedding, tier,
            getattr(fact, "profile_id", "default") or "default",
        )

    def _remove(self, fact_id: str, graph: Any, vector: Any) -> None:
        """Take one fact out of both projections entirely, edges included.

        For a fact that is genuinely gone. Its edges go too, because a hard
        delete cascades them out of SQLite as well, so keeping them here would
        be the projection holding an adjacency the store no longer has.
        """
        if graph is not None:
            graph.remove_fact(fact_id)
        if vector is not None:
            vector.remove_vector(fact_id)

    def _withdraw_candidacy(self, fact_id: str, graph: Any, vector: Any) -> None:
        """Stop offering a fact, without changing the graph's shape.

        For a fact that still exists but may no longer be returned. It leaves
        the entity bridge and the vector — the two things that put a fact into a
        result set — and leaves its edges alone, because those are still in
        ``graph_edges`` and the walk this projection replaces still follows them.
        """
        if graph is not None:
            withdraw = getattr(graph, "remove_fact_candidacy", None)
            # A backend without the narrower call is better served by the full
            # removal than by silently leaving a withheld fact recallable.
            (withdraw or graph.remove_fact)(fact_id)
        if vector is not None:
            vector.remove_vector(fact_id)
