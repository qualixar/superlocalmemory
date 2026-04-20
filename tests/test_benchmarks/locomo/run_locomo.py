# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21

"""Public LoCoMo runner — S9-STAT-04.

Usage::

    python -m tests.test_benchmarks.locomo.run_locomo \
        --dataset /path/to/locomo.json \
        --mode a \
        --out result.json

The runner:
  1. Loads the LoCoMo dataset (user-provided path; LoCoMo is MIT and
     distributed from the upstream paper's repo).
  2. For each conversation, seeds SLM with the conversation turns as
     atomic facts under a temporary profile.
  3. For each question, calls SLM's ``recall`` API and captures the
     top-10 fact_ids.
  4. Scores MRR@10 against the gold fact_ids published by LoCoMo.
  5. Writes a JSON result artifact.

No internal prompts / judge models are embedded — this is the
reproducibility contract, not the full internal harness.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tests.test_benchmarks.locomo.score import mrr_at_k


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # LoCoMo ships as a list of conversation objects. Each conversation
    # has a ``conversation`` (list of turns) and ``qa`` (list of
    # questions with gold answers + gold fact_ids).
    if not isinstance(data, list):
        raise ValueError("expected top-level JSON array (LoCoMo schema)")
    return data


def _seed_profile_with_conversation(
    profile_id: str, conv: dict[str, Any],
) -> None:
    """Write each conversation turn as an atomic fact under the
    temporary profile. Uses SLM's Python API so the runner does not
    depend on the daemon being up.
    """
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig

    cfg = SLMConfig.load()
    cfg.profile_id = profile_id  # type: ignore[attr-defined]
    engine = MemoryEngine(cfg)
    engine.initialize()
    try:
        for turn in conv.get("conversation", []):
            text = turn.get("text") or turn.get("content") or ""
            if not text:
                continue
            try:
                engine.remember(content=text, tags="locomo")
            except Exception:
                # Skip malformed turns without poisoning the profile.
                continue
    finally:
        try:
            engine.close()
        except Exception:
            pass


def _recall_fact_ids(
    profile_id: str, query: str, limit: int = 10,
) -> list[str]:
    """Recall top-K fact_ids for ``query`` under ``profile_id``."""
    from superlocalmemory.core.engine import MemoryEngine
    from superlocalmemory.core.config import SLMConfig

    cfg = SLMConfig.load()
    cfg.profile_id = profile_id  # type: ignore[attr-defined]
    engine = MemoryEngine(cfg)
    engine.initialize()
    try:
        results = engine.recall(query=query, limit=limit)
    finally:
        try:
            engine.close()
        except Exception:
            pass
    out: list[str] = []
    for r in results or []:
        if isinstance(r, dict):
            fid = r.get("fact_id") or r.get("id") or ""
        else:
            fid = getattr(r, "fact_id", "") or getattr(r, "id", "")
        if fid:
            out.append(str(fid))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to LoCoMo JSON file")
    parser.add_argument("--mode", choices=("a", "b", "c"), default="a",
                        help="SLM mode: a = zero-LLM, b = local LLM, "
                             "c = cloud LLM")
    parser.add_argument("--out", type=Path, required=True,
                        help="Path for the result JSON")
    parser.add_argument("--limit", type=int, default=10,
                        help="Top-K for retrieval (default 10)")
    parser.add_argument("--max-conversations", type=int, default=0,
                        help="Optional cap for quick smoke runs; "
                             "0 means full dataset")
    args = parser.parse_args(argv)

    dataset = _load_dataset(args.dataset)
    if args.max_conversations:
        dataset = dataset[: args.max_conversations]

    per_conv_mrr: list[float] = []
    total_q = 0
    t0 = time.monotonic()
    for i, conv in enumerate(dataset):
        profile_id = f"locomo_bench_{i}"
        try:
            _seed_profile_with_conversation(profile_id, conv)
        except Exception as exc:
            print(f"[warn] conv {i} seed failed: {exc}", file=sys.stderr)
            continue
        conv_mrr = []
        for qa in conv.get("qa", []):
            question = qa.get("question") or qa.get("q") or ""
            gold_fact_ids = qa.get("gold_fact_ids") or qa.get("gold_ids") or []
            if not question or not gold_fact_ids:
                continue
            try:
                ranked = _recall_fact_ids(
                    profile_id, question, limit=args.limit,
                )
            except Exception as exc:
                print(f"[warn] recall failed q: {exc}", file=sys.stderr)
                continue
            mrr = mrr_at_k(ranked, gold_fact_ids, k=args.limit)
            conv_mrr.append(mrr)
            total_q += 1
        if conv_mrr:
            per_conv_mrr.append(sum(conv_mrr) / len(conv_mrr))

    elapsed = time.monotonic() - t0
    overall_mrr = (sum(per_conv_mrr) / len(per_conv_mrr)) if per_conv_mrr else 0.0

    result = {
        "version": "3.4.21",
        "mode": args.mode,
        "n_conversations": len(per_conv_mrr),
        "n_questions": total_q,
        "mrr_at_10": overall_mrr,
        "per_conversation_mrr": per_conv_mrr,
        "elapsed_sec": elapsed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[locomo] mode={args.mode} convs={len(per_conv_mrr)} "
        f"qs={total_q} mrr@10={overall_mrr:.3f} "
        f"elapsed={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
