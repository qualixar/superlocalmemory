# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3

"""Benchmark data generator for Phase 1 performance tests.

Generates deterministic synthetic facts with embeddings for
reproducible latency benchmarks. Uses numpy RNG with fixed seeds.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: MIT
"""

from __future__ import annotations

import numpy as np

from superlocalmemory.storage.models import (
    AtomicFact, FactType, MemoryRecord,
)


_FACT_TYPES = [FactType.SEMANTIC, FactType.EPISODIC, FactType.OPINION, FactType.TEMPORAL]


def generate_synthetic_facts(
    count: int = 10_000,
    dimension: int = 768,
    profile_id: str = "bench_profile",
    seed: int = 42,
) -> list[AtomicFact]:
    """Generate N synthetic facts with random L2-normalized embeddings.

    All facts have:
    - Deterministic embeddings (seeded RNG for reproducibility)
    - L2-normalized embedding vectors
    - Unique fact_ids: "bench_fact_{i:06d}"
    - Unique memory_ids: "bench_mem_{i:06d}"
    - Varied fact_types (cyclic: semantic, episodic, opinion, temporal)
    - Varied access_counts (0-99, cyclic)
    - Confidence in [0.5, 1.0], importance in [0.3, 1.0]

    Args:
        count: Number of facts to generate (default 10K).
        dimension: Embedding dimension (default 768).
        profile_id: Profile for all facts.
        seed: RNG seed for reproducibility.

    Returns:
        List of AtomicFact with populated embeddings.
    """
    rng = np.random.RandomState(seed)
    facts: list[AtomicFact] = []

    for i in range(count):
        # Generate L2-normalized embedding
        vec = rng.randn(dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embedding = vec.tolist()

        fact_type = _FACT_TYPES[i % len(_FACT_TYPES)]
        access_count = i % 100
        confidence = 0.5 + (i % 50) / 100.0  # [0.5, 1.0)
        importance = 0.3 + (i % 70) / 100.0  # [0.3, 1.0)

        fact = AtomicFact(
            fact_id=f"bench_fact_{i:06d}",
            memory_id=f"bench_mem_{i:06d}",
            profile_id=profile_id,
            content=f"Synthetic fact number {i} for benchmarking",
            fact_type=fact_type,
            embedding=embedding,
            access_count=access_count,
            confidence=min(1.0, confidence),
            importance=min(1.0, importance),
        )
        facts.append(fact)

    return facts


def generate_query_vectors(
    count: int = 100,
    dimension: int = 768,
    seed: int = 99,
) -> list[list[float]]:
    """Generate N random L2-normalized query vectors for benchmarks.

    Args:
        count: Number of query vectors (default 100).
        dimension: Embedding dimension (default 768).
        seed: RNG seed (different from facts for realistic queries).

    Returns:
        List of L2-normalized query vectors as float lists.
    """
    rng = np.random.RandomState(seed)
    queries: list[list[float]] = []

    for _ in range(count):
        vec = rng.randn(dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        queries.append(vec.tolist())

    return queries


def generate_memory_records(
    count: int = 10_000,
    profile_id: str = "bench_profile",
) -> list[MemoryRecord]:
    """Generate parent memory records for FK satisfaction.

    Must be stored before the corresponding facts (FK constraint).
    Memory IDs match fact memory_ids: "bench_mem_{i:06d}".
    """
    records: list[MemoryRecord] = []

    for i in range(count):
        record = MemoryRecord(
            memory_id=f"bench_mem_{i:06d}",
            profile_id=profile_id,
            content=f"Benchmark memory record {i}",
            session_id=f"bench_session_{i // 100:04d}",
        )
        records.append(record)

    return records
