"""Run one embedding job in its own process (use the right venv's python).

stdin: JSON {"out_dir": str, "threads": int, "stages": [{"embedder": name,
"queries": [...], "docs": [...], "images": [...]}]}
Exactly one stage (one model loadout) per process: load time and peak RSS are
then attributable to that loadout.
stdout: JSON timings. Vectors are written to <out_dir>/<stage>_<kind>.npy.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DOC_BATCH = 16


def _timed(fn, items: list, batch: int) -> tuple[np.ndarray, dict]:
    """Embed items in batches; per-item ms = batch time / batch size."""
    vecs, per_item, t0 = [], [], time.perf_counter()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        t = time.perf_counter()
        vecs.append(fn(chunk))
        per_item += [(time.perf_counter() - t) * 1000 / len(chunk)] * len(chunk)
    return np.concatenate(vecs) if vecs else np.zeros((0, 0), np.float32), {
        "n": len(items), "total_s": time.perf_counter() - t0, "per_item_ms": per_item}


def run_stage(emb, stage: dict, idx: int, out_dir: Path) -> dict:
    jobs = (("queries", emb.embed_queries, 1), ("docs", emb.embed_docs, DOC_BATCH),
            ("images", emb.embed_images, 1))
    info: dict = {"embedder": emb.name}
    for kind, fn, batch in jobs:
        items = stage.get(kind) or []
        if items:
            fn(items[:1])  # warm-up: first call pays one-off setup costs, not timed
            vecs, timing = _timed(fn, items, batch)
            np.save(out_dir / f"{idx}_{kind}.npy", vecs)
            info[kind] = timing
    return info


def main() -> int:
    job = json.load(sys.stdin)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["OMP_NUM_THREADS"] = str(job.get("threads", os.cpu_count() or 1))
    from embedders import REGISTRY

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(job["stages"]) != 1:
        raise SystemExit("one model loadout per worker process: expected exactly one stage")
    stages = []
    for idx, stage in enumerate(job["stages"]):
        emb = REGISTRY[stage["embedder"]]()
        t0 = time.perf_counter()
        emb.load()
        load_s = time.perf_counter() - t0
        info = run_stage(emb, stage, idx, out_dir)
        info["load_s"] = load_s
        stages.append(info)
        del emb
    (out_dir / "timings.json").write_text(json.dumps({"stages": stages}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
