#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""
Sampling utilities for synthetic bootstrap.

Functions for diverse sampling and record aggregation.
"""

from typing import Dict, List


def diverse_sample(
    records: List[dict],
    target: int,
) -> List[dict]:
    """
    Sample records while maintaining source diversity.

    Takes proportional samples from each source strategy to ensure
    the training data isn't dominated by one strategy.

    Args:
        records: List of training records with 'source' field.
        target: Target number of samples to return.

    Returns:
        Sampled list of records (at most target items).
    """
    if len(records) <= target:
        return records

    # Group by source
    by_source: Dict[str, List[dict]] = {}
    for r in records:
        src = r.get('source', 'unknown')
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(r)

    # Proportional allocation
    n_sources = len(by_source)
    if n_sources == 0:
        return records[:target]

    per_source = max(1, target // n_sources)
    sampled = []

    for source, source_records in by_source.items():
        # Take up to per_source from each, or all if fewer
        take = min(len(source_records), per_source)
        sampled.extend(source_records[:take])

    # If under target, fill from remaining
    if len(sampled) < target:
        used_ids = {(r['query_hash'], r['memory_id']) for r in sampled}
        for r in records:
            if len(sampled) >= target:
                break
            key = (r['query_hash'], r['memory_id'])
            if key not in used_ids:
                sampled.append(r)
                used_ids.add(key)

    return sampled[:target]


def count_sources(records: List[dict]) -> Dict[str, int]:
    """
    Count records by source strategy.

    Args:
        records: List of training records with 'source' field.

    Returns:
        Dict mapping source name to count.
    """
    counts: Dict[str, int] = {}
    for r in records:
        src = r.get('source', 'unknown')
        counts[src] = counts.get(src, 0) + 1
    return counts
