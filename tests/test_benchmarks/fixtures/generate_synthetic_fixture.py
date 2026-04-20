# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Track C.2 (LLD-14 §2)

"""Synthetic fixture generator for the evo-memory benchmark.

NOT executed in CI. Run once, check the output JSONL + SHA-256 sidecar
into the repo, and then the harness verifies the hash on every run
(LLD-14 §2.2).

Invocation:

    python tests/test_benchmarks/fixtures/generate_synthetic_fixture.py

Produces next to this script:

- ``evo_memory_synthetic_v1.jsonl`` — ~500 seeds + 30 days of activity
  and test queries, interleaved in chronological order.
- ``evo_memory_synthetic_v1.sha256`` — one-line hex digest, verified at
  harness start-up.

The fixture is the contract. Any non-backward-compat change bumps the
version and adds a new ``_v2.jsonl`` beside this one — never mutate v1.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

# Constants — all baked from LLD-14 §2.2. Changing any of these requires
# a version bump (v1 → v2).
SEED = 12345
N_SEEDS = 500
N_TOPICS = 10
FACTS_PER_TOPIC = 50
N_DAYS = 30
TEST_QUERIES_PER_DAY = 50
ACTIVITY_QUERIES_MIN = 3
ACTIVITY_QUERIES_MAX = 5

TOPICS = [
    "tooling_prefs", "legal_ip", "product_roadmap", "research_notes",
    "conference_deadlines", "system_architecture", "memory_decisions",
    "financial_ops", "hiring_pipeline", "community_feedback",
]

ENTITIES_PER_TOPIC = [
    ("varun", "dark_mode", "terminal", "zsh", "iterm"),
    ("agentassert", "accenture", "patent", "ip", "april_2026"),
    ("slm", "v3421", "mesh", "launch", "pypi"),
    ("paper3", "arxiv", "neurips", "locomo", "mrr"),
    ("iclr", "deadline", "submission", "review", "camera_ready"),
    ("engine", "recall", "embedding", "rerank", "hnsw"),
    ("consolidation", "dedup", "archive", "reward", "lineage"),
    ("cashflow", "invoice", "runway", "burn", "grant"),
    ("candidate", "interview", "offer", "ats", "referral"),
    ("user_feedback", "bug_report", "feature_request", "discord", "issue"),
]

# Fact templates per topic — keep them short and unique so hashed
# embeddings can distinguish them.
FACT_TEMPLATES = [
    "Varun prefers {a} terminals on {b}",
    "AgentAssert {a} decision recorded on {b}",
    "SuperLocalMemory v3.4.22 ships {a} {b}",
    "Research note {a} published at {b}",
    "Conference {a} deadline on {b}",
    "Memory engine {a} wired to {b}",
    "Consolidation step {a} gated by {b}",
    "Finance operation {a} approved by {b}",
    "Hiring record {a} advanced to {b}",
    "Community thread {a} escalated via {b}",
]


def _generate(out_path: Path) -> bytes:
    rng = random.Random(SEED)

    rows: list[dict] = []

    # Day 0 seeds — balanced across topics, deterministic ordering.
    for idx in range(N_SEEDS):
        topic_i = idx % N_TOPICS
        topic = TOPICS[topic_i]
        template = FACT_TEMPLATES[topic_i]
        entities = list(ENTITIES_PER_TOPIC[topic_i])
        a = entities[idx % len(entities)]
        b = entities[(idx + 1) % len(entities)]
        fact = template.format(a=a, b=b) + f" #{idx}"
        rows.append({
            "type": "seed",
            "day": 0,
            "idx": idx,
            "fact": fact,
            "topic": topic,
            "entities": [a, b],
        })

    # Days 1..N_DAYS — activity queries (with outcome labels) + test queries.
    next_idx = N_SEEDS
    for day in range(1, N_DAYS + 1):
        n_activity = rng.randint(ACTIVITY_QUERIES_MIN, ACTIVITY_QUERIES_MAX)
        for _ in range(n_activity):
            topic_i = rng.randrange(N_TOPICS)
            topic = TOPICS[topic_i]
            # 70% pick a single seed from this topic (truth), 20% cross,
            # 10% multi-hop (two seeds same topic).
            r = rng.random()
            seed_indices = [i for i in range(N_SEEDS) if i % N_TOPICS == topic_i]
            if r < 0.70:
                relevant = [rng.choice(seed_indices)]
                reward = 0.9  # "cite" → strong positive
            elif r < 0.90:
                other_topic = (topic_i + rng.randrange(1, N_TOPICS)) % N_TOPICS
                other_seeds = [
                    i for i in range(N_SEEDS) if i % N_TOPICS == other_topic
                ]
                relevant = [rng.choice(other_seeds)]
                reward = 0.0  # "requery" → negative
            else:
                picks = rng.sample(seed_indices, min(2, len(seed_indices)))
                relevant = sorted(picks)
                reward = 0.75  # "edit + dwell" → positive

            rows.append({
                "type": "query",
                "day": day,
                "idx": next_idx,
                "query": f"activity-query day{day} topic={topic}",
                "relevant_seed_idxs": relevant,
                "topic": topic,
                "is_test": False,
                "reward": reward,
            })
            next_idx += 1

        # Test queries — side-effect-free, scored for MRR/Recall.
        for _ in range(TEST_QUERIES_PER_DAY):
            topic_i = rng.randrange(N_TOPICS)
            topic = TOPICS[topic_i]
            seed_indices = [
                i for i in range(N_SEEDS) if i % N_TOPICS == topic_i
            ]
            r = rng.random()
            if r < 0.70:
                relevant = [rng.choice(seed_indices)]
            elif r < 0.90:
                other_topic = (
                    (topic_i + rng.randrange(1, N_TOPICS)) % N_TOPICS
                )
                other_seeds = [
                    i for i in range(N_SEEDS) if i % N_TOPICS == other_topic
                ]
                relevant = [rng.choice(other_seeds)]
            else:
                picks = rng.sample(seed_indices, min(2, len(seed_indices)))
                relevant = sorted(picks)

            rows.append({
                "type": "query",
                "day": day,
                "idx": next_idx,
                "query": f"test-query day{day} topic={topic} "
                         f"probe={next_idx}",
                "relevant_seed_idxs": relevant,
                "topic": topic,
                "is_test": True,
            })
            next_idx += 1

    # Stable serialisation — one row per line, sorted keys for
    # byte-reproducibility across json's dict-ordering quirks.
    payload = "\n".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows
    ).encode("utf-8") + b"\n"
    out_path.write_bytes(payload)
    return payload


def main() -> None:
    here = Path(__file__).resolve().parent
    jsonl_path = here / "evo_memory_synthetic_v1.jsonl"
    sha_path = here / "evo_memory_synthetic_v1.sha256"
    payload = _generate(jsonl_path)
    digest = hashlib.sha256(payload).hexdigest()
    sha_path.write_text(digest + "\n", encoding="utf-8")
    print(f"Wrote {jsonl_path.name}: {len(payload)} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
