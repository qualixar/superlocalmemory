"""Calibrate the picture evidence floor (SLM's media_min_score) per candidate.

Reads the image-channel-only runs (c1m, c2m, c3m): the top-1 cosine of each
media query against pictures and pages. On dev media queries only, picks the
floor that best separates answerable queries whose top-1 is correct from
never-stored queries (abstention F1, stats.tune_threshold); reports it on test.
SLM ships media_min_score = 0.30; a floor above a model's real cosines means
no picture ever counts as evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import stats
from build_dataset import bench_home

MEDIA = ("image", "photo", "pdf_page")
SHIPPED_FLOOR = 0.30


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _media_queries(rows: list[dict]) -> list[dict]:
    return [q for q in rows if q["stratum"] in MEDIA or (q["stratum"] == "unanswerable"
                                                          and q["id"].startswith("med:"))]


def _rows(queries, run, scores, qrels):
    """(top-1 cosine, is a never-stored query, top-1 correct) per media query."""
    out = []
    for q in queries:
        top = max(run.get(q["id"], {"": 0}).items(), key=lambda kv: kv[1])[0]
        out.append((scores.get(q["id"], 0.0), not q["answerable"], top in qrels.get(q["id"], {})))
    return out


def _at(rows, floor: float) -> dict:
    kept = [r for r in rows if r[0] >= floor]
    unans = [r for r in rows if r[1]]
    hits = [r for r in rows if not r[1] and r[2]]
    return {"floor": floor,
            "false_answer_rate": float(np.mean([r[0] >= floor for r in unans])) if unans else None,
            "correct_kept": float(np.mean([r[0] >= floor for r in hits])) if hits else None,
            "n_kept": len(kept), "n_unanswerable": len(unans), "n_correct_top1": len(hits)}


def calibrate(home: Path, system: str) -> dict:
    ds, runs = home / "dataset", home / "runs"
    run = json.loads((runs / f"{system}.json").read_text())
    scores = json.loads((runs / f"{system}.abstain.json").read_text())
    split = {s: _media_queries(_jsonl(ds / "queries" / f"{s}.jsonl")) for s in ("dev", "test")}
    qrels = {**json.loads((ds / "qrels" / "dev.json").read_text()),
             **json.loads((ds / "qrels" / "test.json").read_text())}
    dev = _rows(split["dev"], run, scores, qrels)
    test = _rows(split["test"], run, scores, qrels)
    usable = [r for r in dev if r[1] or r[2]]
    tau = stats.tune_threshold([r[0] for r in usable], [r[1] for r in usable])
    correct = [r[0] for r in test if r[2]]
    return {"system": system, "tuned_on_dev": _at(test, tau), "shipped_0_30": _at(test, SHIPPED_FLOOR),
            "test_correct_top1_cosine_p10_p50": [float(np.percentile(correct, p)) for p in (10, 50)]
            if correct else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--systems", default="c1m,c2m,c3m")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    found = [s for s in args.systems.split(",") if (args.home / "runs" / f"{s}.json").exists()]
    result = [calibrate(args.home, s) for s in found]
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
