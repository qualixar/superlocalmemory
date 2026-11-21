# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Cognitive Consolidation Quantization (CCQ) — Phase E.
Sleep-time consolidation that mirrors hippocampal-neocortical transfer:
  1. IDENTIFY: warm/cold facts below retention threshold
  2. CLUSTER: group by entity overlap + temporal proximity (Union-Find)
  3. EXTRACT GIST: rules (Mode A) or LLM (Mode B/C) summary
  4. COMPRESS: source embeddings -> PolarQuant 2-bit
  5. STORE: gist block at float32 + archive source facts
  6. AUDIT: complete audit trail
Hard rules:
  - Already-consolidated facts NEVER re-consolidated (HR-01)
  - Minimum cluster size 3 (HR-02)
  - Gist must cover 50% shared entities (HR-03)
  - Source facts soft-archived, NEVER deleted (HR-04)
  - Gist embedding always float32 (HR-05)
  - Parameterized SQL only (HR-06)
  - Per-cluster error isolation (HR-07)
  - Idempotent (HR-08)
  - PolarQuant optional (HR-10)
  - Audit trail mandatory (HR-11)

"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from superlocalmemory.encoding.ccq_dates import _parse_date, _temporal_midpoint
from superlocalmemory.storage.json_columns import load_json_column
from superlocalmemory.storage.models import _new_id

if TYPE_CHECKING:
    from superlocalmemory.core.config import CCQConfig
    from superlocalmemory.storage.database import DatabaseManager
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Anything that produces an embedding vector from text."""

    def encode(self, text: str) -> list[float]: ...


class LLM(Protocol):
    """Anything that can generate text from a prompt."""

    def generate(self, prompt: str, system: str = "") -> str: ...

    def is_available(self) -> bool: ...


# ---------------------------------------------------------------------------
# Data classes (frozen, immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsolidationCluster:
    """A group of related fading memories identified for consolidation."""

    cluster_id: str
    fact_ids: tuple[str, ...]
    shared_entities: tuple[str, ...]
    temporal_centroid: str
    avg_retention: float
    fact_count: int


@dataclass(frozen=True)
class GistResult:
    """Output of the gist extraction step."""

    gist_text: str
    key_entities: tuple[str, ...]
    extraction_mode: str       # 'rules' or 'llm'
    representative_fact_id: str


@dataclass(frozen=True)
class CCQPipelineResult:
    """Full result of one CCQ pipeline execution."""

    clusters_processed: int
    blocks_created: int
    facts_archived: int
    total_bytes_before: int
    total_bytes_after: int
    compression_ratio: float
    audit_entries: tuple[str, ...]
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Union-Find helper
# ---------------------------------------------------------------------------


class _UnionFind:
    """Minimal Union-Find for entity-based clustering."""

    __slots__ = ("_parent", "_rank")

    def __init__(self, elements: list[str]) -> None:
        self._parent: dict[str, str] = {e: e for e in elements}
        self._rank: dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for element in self._parent:
            groups[self.find(element)].append(element)
        return dict(groups)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


# _parse_date / _temporal_midpoint live in encoding/ccq_dates.py and are
# imported at the top of this module (tests import them from here).


# ---------------------------------------------------------------------------
# CognitiveConsolidator
# ---------------------------------------------------------------------------


def _config_for_this_store() -> "CCQConfig":
    """The settings this store is actually running under.

    A caller that supplies only a database used to get the bare defaults, which
    say a language model may be used to summarise. Three of the four places
    that build this pass only a database, so on a store configured to use no
    language model at all, the refusal below rested on nobody having handed one
    in -- true today, and not a property of the mode. Reading the store's own
    settings makes it a property of the mode.

    Falls back to the defaults if the settings cannot be read, because a
    consolidation pass that cannot start is worse than one that starts with the
    documented defaults.
    """
    from superlocalmemory.core.config import CCQConfig as _CCQConfig

    try:
        from superlocalmemory.core.config import SLMConfig

        resolved = getattr(SLMConfig.load(), "ccq", None)
        if isinstance(resolved, _CCQConfig):
            return resolved
    except Exception as exc:  # noqa: BLE001
        logger.debug("consolidator: using default settings (%s)", exc)
    return _CCQConfig()


class CognitiveConsolidator:
    """CCQ engine: sleep-time consolidation with quantization.

    Executes 6-step pipeline: identify -> cluster -> gist -> compress
    -> store -> audit. Each cluster failure is isolated (HR-07).
    """

    __slots__ = ("_db", "_embedder", "_llm", "_config")

    def __init__(
        self,
        db: DatabaseManager,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        config: CCQConfig | None = None,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._config = config if config is not None else _config_for_this_store()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_pipeline(
        self, profile_id: str, dry_run: bool = False,
    ) -> CCQPipelineResult:
        """Execute the full 6-step CCQ pipeline.

        Per-cluster error isolation: one cluster failure does NOT
        abort the pipeline (HR-07).

        Args:
            profile_id: Target profile.
            dry_run: If True, identify clusters but don't apply changes.
        """
        # Step 1: Identify candidates
        candidates = self._step1_identify(profile_id)
        if not candidates:
            return self._empty_result()

        # Step 2: Cluster by entity overlap + temporal proximity
        clusters = self._step2_cluster(candidates, profile_id)
        if not clusters:
            return self._empty_result()

        if dry_run:
            return CCQPipelineResult(
                clusters_processed=len(clusters),
                blocks_created=0,
                facts_archived=len(candidates),
                total_bytes_before=0,
                total_bytes_after=0,
                compression_ratio=0.0,
                audit_entries=(),
                errors=(),
            )

        # Process each cluster
        blocks_created = 0
        facts_archived = 0
        bytes_before = 0
        bytes_after = 0
        audit_ids: list[str] = []
        errors: list[str] = []

        for cluster in clusters:
            try:
                # Step 3: Extract gist
                gist = self._step3_extract_gist(cluster, profile_id)

                # Step 4: Compress source embeddings
                cb, ca = self._step4_compress_embeddings(cluster, profile_id)

                # Step 5: Store block + archive source facts
                block_id = self._step5_store_block(cluster, gist, profile_id)

                # Step 6: Audit trail
                audit_id = self._step6_audit(
                    cluster, gist, cb, ca, block_id, profile_id,
                )

                blocks_created += 1
                facts_archived += cluster.fact_count
                bytes_before += cb
                bytes_after += ca
                audit_ids.append(audit_id)

            except Exception as exc:
                logger.warning(
                    "CCQ cluster %s failed (non-fatal): %s",
                    cluster.cluster_id, exc,
                )
                errors.append(
                    f"cluster={cluster.cluster_id}: {exc!s}",
                )

        compression_ratio = (
            bytes_before / bytes_after if bytes_after > 0 else 0.0
        )

        return CCQPipelineResult(
            clusters_processed=len(clusters),
            blocks_created=blocks_created,
            facts_archived=facts_archived,
            total_bytes_before=bytes_before,
            total_bytes_after=bytes_after,
            compression_ratio=round(compression_ratio, 2),
            audit_entries=tuple(audit_ids),
            errors=tuple(errors),
        )

    # ------------------------------------------------------------------
    # Step 1: Identify candidates
    # ------------------------------------------------------------------

    def _step1_identify(self, profile_id: str) -> list[dict]:
        """Identify warm/cold facts not yet consolidated.

        Excludes:
          - Active/archive/forgotten lifecycle zones
          - Facts above retention threshold
          - Facts already in ccq_consolidated_blocks (HR-01, HR-08)
        """
        # Only facts the store still believes are consolidated: a fact a
        # correction retired must never be copied into a fresh, current gist
        # (the retired rule came back under a new id).
        current = self._db.current_fact_clause("f")
        rows = self._db.execute(
            """
            SELECT f.fact_id, f.content, f.canonical_entities_json,
                   f.created_at, f.observation_date, f.importance,
                   f.confidence,
                   r.retention_score, r.lifecycle_zone, r.memory_strength
            FROM atomic_facts f
            INNER JOIN fact_retention r
                ON f.fact_id = r.fact_id AND r.profile_id = ?
            WHERE f.profile_id = ?
              AND r.lifecycle_zone IN ('warm', 'cold')
              AND r.retention_score < ?
              AND f.lifecycle != 'forgotten'
              -- Withheld rows are not candidates, and this is not tidiness.
              -- 304 of them remain warm/cold in atomic_facts after 4.0.10
              -- withholds them, and each still carries its cluster's POOLED
              -- canonical_entities_json. CCQ clusters on entity overlap, so one
              -- withheld summary naming ten entities joins a cluster of real
              -- memories, contributes model prose to the gist, and then this
              -- pass archives every source in the cluster -- including the real
              -- memories, at scores M043's restore would not bring back.
              -- Exactly the damage this release exists to stop, on a path the
              -- release did not otherwise touch.
              AND COALESCE(f.quarantined, 0) = 0
              {current}
              AND f.fact_id NOT IN (
                  SELECT je.value
                  FROM ccq_consolidated_blocks ccb,
                       json_each(ccb.source_fact_ids) je
                  WHERE ccb.profile_id = ?
              )
            ORDER BY r.retention_score ASC
            LIMIT ?
            """.replace("{current}", current),
            (
                profile_id,
                profile_id,
                self._config.retention_threshold,
                profile_id,
                self._config.max_candidates_per_run,
            ),
        )

        candidates: list[dict] = []
        for row in rows or []:
            d = dict(row)
            try:
                d["canonical_entities"] = load_json_column(
                    d.get("canonical_entities_json"), [],
                    column="atomic_facts.canonical_entities_json", row=d.get("fact_id"))
            except (ValueError, TypeError):
                d["canonical_entities"] = []
            candidates.append(d)

        return candidates

    # ------------------------------------------------------------------
    # Step 2: Cluster by entity overlap + temporal proximity
    # ------------------------------------------------------------------

    def _step2_cluster(
        self,
        candidates: list[dict],
        profile_id: str,
    ) -> list[ConsolidationCluster]:
        """Group candidates via Union-Find (entity overlap) + temporal sub-clustering."""
        if len(candidates) < self._config.min_cluster_size:
            return []

        fact_ids = [c["fact_id"] for c in candidates]
        fact_map = {c["fact_id"]: c for c in candidates}

        # Build entity-to-fact index
        entity_index: dict[str, list[str]] = defaultdict(list)
        for c in candidates:
            for entity in c.get("canonical_entities", []):
                entity_index[entity].append(c["fact_id"])

        # Union-Find by entity overlap
        uf = _UnionFind(fact_ids)
        for i, fid_a in enumerate(fact_ids):
            entities_a = set(fact_map[fid_a].get("canonical_entities", []))
            for fid_b in fact_ids[i + 1:]:
                entities_b = set(
                    fact_map[fid_b].get("canonical_entities", []),
                )
                if len(entities_a & entities_b) >= self._config.min_entity_overlap:
                    uf.union(fid_a, fid_b)

        # Extract components and sub-cluster temporally
        clusters: list[ConsolidationCluster] = []
        for _root, group_ids in uf.components().items():
            if len(group_ids) < self._config.min_cluster_size:
                continue
            sub_clusters = self._temporal_subcluster(group_ids, fact_map)
            for sc_ids in sub_clusters:
                if len(sc_ids) < self._config.min_cluster_size:
                    continue
                # Cap cluster size (HR: prevents huge gists)
                sc_ids = sc_ids[: self._config.max_cluster_size]

                # Compute shared entities (appear in ALL facts)
                entity_sets = [
                    set(fact_map[fid].get("canonical_entities", []))
                    for fid in sc_ids
                ]
                shared = (
                    set.intersection(*entity_sets) if entity_sets else set()
                )

                # Compute temporal centroid
                dates = [
                    _parse_date(
                        fact_map[fid].get("observation_date")
                        or fact_map[fid].get("created_at"),
                    )
                    for fid in sc_ids
                ]
                valid_dates = [d for d in dates if d is not None]
                centroid = _temporal_midpoint(valid_dates)

                # Average retention
                avg_ret = sum(
                    fact_map[fid].get("retention_score", 0.0)
                    for fid in sc_ids
                ) / len(sc_ids)

                clusters.append(
                    ConsolidationCluster(
                        cluster_id=_new_id(),
                        fact_ids=tuple(sc_ids),
                        shared_entities=tuple(sorted(shared)),
                        temporal_centroid=centroid,
                        avg_retention=round(avg_ret, 4),
                        fact_count=len(sc_ids),
                    ),
                )

        return clusters

    def _temporal_subcluster(
        self,
        fact_ids: list[str],
        fact_map: dict[str, dict],
    ) -> list[list[str]]:
        """Split an entity group into temporal sub-clusters within window."""
        dated: list[tuple[str, datetime | None]] = []
        for fid in fact_ids:
            raw = (
                fact_map[fid].get("observation_date")
                or fact_map[fid].get("created_at")
            )
            dated.append((fid, _parse_date(raw)))

        # A fact whose date is unknown cannot be shown to fall inside the
        # window, so it joins nothing (it used to join whatever
        # cluster was open, which merged facts months apart).
        dated = [(fid, dt) for fid, dt in dated if dt is not None]
        dated.sort(key=lambda t: t[1])

        sub_clusters: list[list[str]] = [[]]
        prev_dt: datetime | None = None
        window_seconds = self._config.temporal_window_days * 86400

        for fid, dt in dated:
            if (
                prev_dt is not None
                and (dt - prev_dt).total_seconds() > window_seconds
            ):
                sub_clusters.append([])
            sub_clusters[-1].append(fid)
            prev_dt = dt

        return [sc for sc in sub_clusters if sc]

    # ------------------------------------------------------------------
    # Step 3: Extract gist
    # ------------------------------------------------------------------

    def _step3_extract_gist(
        self,
        cluster: ConsolidationCluster,
        profile_id: str,
    ) -> GistResult:
        """Extract a single gist from the cluster.

        Mode B (LLM) attempted first if available; falls back to Mode A (rules).
        """
        # Fetch fact content
        placeholders = ",".join("?" for _ in cluster.fact_ids)
        rows = self._db.execute(
            f"SELECT fact_id, content, importance, confidence, "
            f"       canonical_entities_json "
            f"FROM atomic_facts "
            f"WHERE fact_id IN ({placeholders}) AND profile_id = ? "
            # Belt and braces with the identify query above: a cluster assembled
            # before a row was withheld must not contribute its text to a gist.
            f"  AND COALESCE(quarantined, 0) = 0",
            (*cluster.fact_ids, profile_id),
        )

        facts: list[dict] = []
        for r in rows or []:
            d = dict(r)
            try:
                d["canonical_entities"] = load_json_column(
                    d.get("canonical_entities_json"), [],
                    column="atomic_facts.canonical_entities_json", row=d.get("fact_id"))
            except (ValueError, TypeError):
                d["canonical_entities"] = []
            facts.append(d)

        if not facts:
            return GistResult(
                gist_text="[empty cluster]",
                key_entities=(),
                extraction_mode="rules",
                representative_fact_id="",
            )

        # Configuration first, then availability. Mode A sets use_llm_gist=False,
        # so this refuses because the mode says to — not because an LLM happened
        # not to be passed in. The previous order tested availability first,
        # which meant a caller that supplied a model to a Mode A engine would
        # have got an LLM call on a mode whose whole promise is that nothing
        # does.
        if (
            self._config.use_llm_gist
            and self._llm is not None
        ):
            try:
                gist = self._extract_gist_llm(
                    facts, cluster.shared_entities,
                )
                if gist is not None:
                    return gist
            except Exception as exc:
                logger.warning("LLM gist failed, falling back to rules: %s", exc)

        # Mode A: rules-based
        return self._extract_gist_mode_a(facts, cluster.shared_entities)

    def _extract_gist_llm(
        self,
        facts: list[dict],
        shared_entities: tuple[str, ...],
    ) -> GistResult | None:
        """LLM-based gist extraction (Mode B). Returns None if validation fails."""
        if self._llm is None:
            return None

        fact_lines = "\n".join(
            f"{i + 1}. {f['content']}" for i, f in enumerate(facts)
        )
        entity_str = ", ".join(shared_entities)

        prompt = (
            f"Summarize these {len(facts)} related memories into one "
            f"concise factual statement.\n"
            f"Preserve all key entities: {entity_str}.\n\n"
            f"Memories:\n{fact_lines}\n\n"
            f"Consolidated statement:"
        )

        response = self._llm.generate(
            prompt,
            system="You are a precise memory consolidator.",
        )

        # Validate entity coverage (HR-03)
        if shared_entities:
            mentioned = sum(
                1 for e in shared_entities
                if e.lower() in response.lower()
            )
            coverage = mentioned / len(shared_entities)
            if coverage < self._config.min_entity_coverage:
                logger.info(
                    "LLM gist entity coverage %.2f < %.2f, falling back",
                    coverage, self._config.min_entity_coverage,
                )
                return None

        # Truncate if needed
        gist_text = response
        if len(gist_text) > self._config.max_gist_chars:
            gist_text = gist_text[: self._config.max_gist_chars - 3] + "..."

        return GistResult(
            gist_text=gist_text,
            key_entities=shared_entities,
            extraction_mode="llm",
            representative_fact_id="",
        )

    def _extract_gist_mode_a(
        self,
        facts: list[dict],
        shared_entities: tuple[str, ...],
    ) -> GistResult:
        """Rules-based gist: representative fact + entity summary."""
        # Find representative (highest importance * confidence)
        best = max(
            facts,
            key=lambda f: f.get("importance", 0) * f.get("confidence", 0),
        )

        # Entity frequency across cluster
        entity_freq: dict[str, int] = defaultdict(int)
        for f in facts:
            for e in f.get("canonical_entities", []):
                entity_freq[e] += 1
        top_entities = sorted(
            entity_freq, key=lambda k: entity_freq[k], reverse=True,
        )[:5]

        entity_summary = ", ".join(top_entities) if top_entities else ""
        gist = f"{best['content']} [Entities: {entity_summary}]"

        if len(gist) > self._config.max_gist_chars:
            gist = gist[: self._config.max_gist_chars - 3] + "..."

        return GistResult(
            gist_text=gist,
            key_entities=tuple(top_entities),
            extraction_mode="rules",
            representative_fact_id=best["fact_id"],
        )

    # ------------------------------------------------------------------
    # Step 4: Compress source embeddings
    # ------------------------------------------------------------------

    def _step4_compress_embeddings(
        self,
        cluster: ConsolidationCluster,
        profile_id: str,
    ) -> tuple[int, int]:
        """Compress source fact embeddings. Returns (bytes_before, bytes_after).

        PolarQuant is optional (HR-10). If not available, marks as pending.
        """
        if not self._config.compress_embeddings:
            return (0, 0)

        total_before = 0
        total_after = 0

        for fact_id in cluster.fact_ids:
            bb, ba = self._compress_single_embedding(fact_id, profile_id)
            total_before += bb
            total_after += ba

        return (total_before, total_after)

    def _compress_single_embedding(
        self,
        fact_id: str,
        profile_id: str,
    ) -> tuple[int, int]:
        """Measure one source embedding. Returns (bytes_before, bytes_after).

        This step does NOT quantize, and since 4.1.25 it no longer says it
        does. It used to import PolarQuantEncoder, never call
        it, write ``quantization_level='polar2', bit_width=2`` and report an
        estimated 2-bit size: the stored embedding kept every byte (identical
        SHA-256), and the fake ``bit_width=2`` made the EAP scheduler — which
        does real quantization through ``QuantizedStore`` — treat the fact as
        already compressed and skip it.

        Physical quantization of a consolidated (archived) source belongs to
        the EAP lifecycle scheduler, which reads the retention zone this
        pipeline sets. So this measures the float32 payload and reports it
        unchanged: bytes_after == bytes_before, a compression ratio of 1.0.
        """
        meta = self._db.execute(
            "SELECT vec_rowid, dimension FROM embedding_metadata "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        if not meta:
            return (0, 0)
        dim = dict(meta[0]).get("dimension", 768)
        eq_meta = self._db.execute(
            "SELECT quantization_level FROM embedding_quantization_metadata "
            "WHERE fact_id = ? AND profile_id = ?",
            (fact_id, profile_id),
        )
        if eq_meta:
            level = dict(eq_meta[0]).get("quantization_level", "float32")
            if level in ("polar2", "polar4", "deleted"):
                return (0, 0)  # quantized by EAP already; nothing to measure
        bytes_now = int(dim) * 4  # float32, as stored
        return (bytes_now, bytes_now)

    # ------------------------------------------------------------------
    # Step 5: Store block + archive source facts
    # ------------------------------------------------------------------

    def _step5_store_block(
        self,
        cluster: ConsolidationCluster,
        gist: GistResult,
        profile_id: str,
    ) -> str:
        """Atomically persist the replacement before soft-archiving its sources."""
        from superlocalmemory.encoding.gist_store import store_block
        return store_block(self._db, self._embedder, cluster, gist, profile_id, _new_id,
                           lambda ids: self._still_current_confidence(ids, profile_id))

    def _still_current_confidence(self, fact_ids: tuple[str, ...],
                                  profile_id: str) -> float:
        """Min confidence of the sources; raises if any is no longer current."""
        marks = ",".join("?" for _ in fact_ids)
        rows = self._db.execute(
            f"SELECT confidence FROM atomic_facts f WHERE f.profile_id = ? "
            f"AND f.fact_id IN ({marks}){self._db.current_fact_clause('f')}",
            (profile_id, *fact_ids),
        )
        if len(rows or []) != len(fact_ids):
            raise RuntimeError(
                "a source was retired or removed after selection; "
                "cluster abandoned")
        return min(float(dict(r).get("confidence") or 0.0) for r in rows)

    # ------------------------------------------------------------------
    # Step 6: Audit trail
    # ------------------------------------------------------------------

    def _step6_audit(
        self,
        cluster: ConsolidationCluster,
        gist: GistResult,
        bytes_before: int,
        bytes_after: int,
        block_id: str,
        profile_id: str,
    ) -> str:
        """Record audit trail for this consolidation (HR-11)."""
        audit_id = _new_id()
        compression_ratio = (
            bytes_before / bytes_after if bytes_after > 0 else 0.0
        )

        self._db.store_ccq_audit({
            "audit_id": audit_id,
            "profile_id": profile_id,
            "cluster_id": cluster.cluster_id,
            "block_id": block_id,
            "fact_ids": json.dumps(list(cluster.fact_ids)),
            "fact_count": cluster.fact_count,
            "gist_text": gist.gist_text,
            "extraction_mode": gist.extraction_mode,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "compression_ratio": round(compression_ratio, 2),
            "shared_entities": json.dumps(list(cluster.shared_entities)),
        })

        return audit_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> CCQPipelineResult:
        """Return a zero-work pipeline result."""
        return CCQPipelineResult(
            clusters_processed=0,
            blocks_created=0,
            facts_archived=0,
            total_bytes_before=0,
            total_bytes_after=0,
            compression_ratio=0.0,
            audit_entries=(),
            errors=(),
        )
