"""Text-only check at scale: is a candidate's text recall within 3 points of S1?

The main golden set has 39 text test queries, too few to test a 3-point margin
(the 95% CI is about 26 points wide). This check uses every LoCoMo question with
evidence turns in the single-hop, multi-hop and temporal categories across all ten
conversations (about 1,440 questions). Each conversation is its own store: a
question only searches the turns of its own conversation, as one SLM profile would.
LoCoMo is downloaded by fetch_data.sh (CC BY-NC 4.0); nothing is committed.

Writes $SLM_BENCH_HOME/text_scale/<system>.json (per-query recall@5 and MRR@10)
and prints paired deltas against S1 with the one-model rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import ranking
import run_eval
import stats
from build_dataset import bench_home, dia_ids, turns_of

CATEGORIES = {1: "multi_hop", 2: "temporal", 4: "single_hop"}
# system -> (embedder, venv label); s1 = nomic as SLM ships it (= C1's text side)
SYSTEMS = {"s1": ("nomic", "slm"), "c2": ("eg2_text", "eg2"), "c3": ("qwen3vl", "eg2")}


def load(locomo: Path, max_convs: int | None = None) -> tuple[list[dict], list[dict]]:
    """Every turn of every conversation and every usable question."""
    docs, queries = [], []
    for sample in json.loads(locomo.read_text())[:max_convs]:
        sid, turns = sample["sample_id"], turns_of(sample)
        for dia, t in turns.items():
            docs.append({"id": f"{sid}:{dia}", "conv": sid, "text": f"[{t['date']}] {t['speaker']}: {t['text']}"})
        for i, qa in enumerate(sample["qa"]):
            ev = dia_ids(qa.get("evidence", []))
            if qa["category"] in CATEGORIES and ev and all(e in turns for e in ev):
                queries.append({"id": f"{sid}:{i}", "conv": sid, "text": qa["question"],
                                "category": CATEGORIES[qa["category"]], "rel": [f"{sid}:{e}" for e in ev]})
    return docs, queries


def _per_query(tops: list[list[str]], queries: list[dict]) -> dict:
    out = {}
    for q, top in zip(queries, tops):
        rel = set(q["rel"])
        rank = next((i for i, d in enumerate(top[:10]) if d in rel), None)
        out[q["id"]] = {"recall@5": len(rel & set(top[:5])) / len(rel),
                        "mrr@10": 0.0 if rank is None else 1.0 / (rank + 1), "category": q["category"]}
    return out


def _search(qv: np.ndarray, dv: np.ndarray, docs: list[dict], queries: list[dict]) -> list[list[str]]:
    """Cosine search restricted to the question's own conversation."""
    by_conv: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        by_conv.setdefault(d["conv"], []).append(i)
    tops = []
    for qi, q in enumerate(queries):
        idx = by_conv[q["conv"]]
        sims = dv[idx] @ qv[qi]
        tops.append([docs[idx[j]]["id"] for j in np.argsort(-sims, kind="stable")[:10]])
    return tops


def run_system(name: str, docs, queries, pythons: dict) -> tuple[dict, dict]:
    if name == "bm25":
        tops = []
        for conv in dict.fromkeys(q["conv"] for q in queries):
            cd = [d for d in docs if d["conv"] == conv]
            cq = [q for q in queries if q["conv"] == conv]
            ranked = ranking.bm25_rank([d["id"] for d in cd], [d["text"] for d in cd], [q["text"] for q in cq])
            tops += [[d for d, _ in r[:10]] for r in ranked]
        return _per_query(tops, queries), {}
    embedder, label = SYSTEMS[name]
    job = {"stages": [{"embedder": embedder, "queries": [q["text"] for q in queries],
                       "docs": [d["text"] for d in docs]}]}
    timings, peak, out = run_eval.spawn(job, pythons[label])
    tops = _search(np.load(out / "0_queries.npy"), np.load(out / "0_docs.npy"), docs, queries)
    return _per_query(tops, queries), run_eval.summarise(timings, peak)


def compare(base: dict, other: dict) -> dict:
    """Paired recall@5 delta (other minus base), overall and per category, plus the rule."""
    ids = sorted(set(base) & set(other))
    out = {"overall": stats.one_model_rule([base[i]["recall@5"] for i in ids],
                                           [other[i]["recall@5"] for i in ids])}
    for cat in CATEGORIES.values():
        sub = [i for i in ids if base[i]["category"] == cat]
        d, lo, hi = stats.paired_delta_ci([base[i]["recall@5"] for i in sub], [other[i]["recall@5"] for i in sub])
        out[cat] = {"n": len(sub), "delta": d, "lo": lo, "hi": hi}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--systems", default="bm25,s1,c2")
    ap.add_argument("--max-convs", type=int, help="first N conversations only (slow models)")
    ap.add_argument("--python-slm")
    ap.add_argument("--python-eg2")
    args = ap.parse_args(argv)
    docs, queries = load(args.home / "locomo" / "data" / "locomo10.json", args.max_convs)
    out_dir = args.home / "text_scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    pythons = {"slm": args.python_slm, "eg2": args.python_eg2}
    for name in args.systems.split(","):
        per_q, timing = run_system(name, docs, queries, pythons)
        mean = {m: float(np.mean([v[m] for v in per_q.values()])) for m in ("recall@5", "mrr@10")}
        (out_dir / f"{name}.json").write_text(json.dumps({"per_query": per_q, "mean": mean, "timings": timing,
                                                          "n_docs": len(docs), "n_queries": len(queries)}))
        print(name, len(queries), mean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
