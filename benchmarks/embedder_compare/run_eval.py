"""Embed the shared corpus with each system and write ranx-format runs.

Systems: bm25 (model-free smoke), s1 (nomic as SLM ships it, no task prefixes;
media via OCR text), s1p (nomic with task prefixes, reference only), s2
(EmbeddingGemma 2 one-model: text-only loadout for text, full model for images
and pages), s3 (nomic text plus EmbeddingGemma 2 media channel, weighted
reciprocal-rank fusion, k=60, media weight tuned on dev); model-decision
candidates c1/c2/c3 (merged) and c1f/c2f/c3f (fused), see candidates.py. Every
model loadout runs in its own process.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import candidates
import fusion
import ocr
import ranking
import rss
from build_dataset import bench_home, dataset_hash

HERE = Path(__file__).resolve().parent
ALL_SYSTEMS = ("bm25", "s1", "s2", "s3")  # s1p (prefixed nomic) is opt-in, reference only


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_dataset(ds: Path) -> tuple[list[dict], list[dict]]:
    queries = _jsonl(ds / "queries" / "dev.jsonl") + _jsonl(ds / "queries" / "test.jsonl")
    return _jsonl(ds / "corpus.jsonl"), sorted(queries, key=lambda q: q["id"])


def pctl(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def spawn(job: dict, python: str) -> tuple[dict, float, Path]:
    """Run worker.py in a child process; return (timings, peak RSS MB, out_dir)."""
    out_dir = Path(tempfile.mkdtemp(prefix="slm-embed-"))
    job = {**job, "out_dir": str(out_dir), "threads": os.cpu_count() or 1}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
    proc = subprocess.Popen([python, str(HERE / "worker.py")], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    with rss.RssSampler(proc.pid) as sampler:
        _, err = proc.communicate(json.dumps(job))
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({proc.returncode}): {err.strip()[-600:]}")
    return json.loads((out_dir / "timings.json").read_text()), sampler.peak_mb, out_dir


def _summ(per_item: list[float]) -> dict:
    return {"n": len(per_item), "p50_ms": pctl(per_item, 50), "p95_ms": pctl(per_item, 95)}


def summarise(timings: dict, peak_mb: float, queries_key: str = "queries") -> dict:
    stages = timings["stages"]
    get = lambda kind: [x for s in stages for x in s.get(kind, {}).get("per_item_ms", [])]
    items = sum(s.get(k, {}).get("n", 0) for s in stages for k in ("docs", "images"))
    secs = sum(s.get(k, {}).get("total_s", 0.0) for s in stages for k in ("docs", "images"))
    return {"load_s": sum(s["load_s"] for s in stages), "peak_rss_mb": peak_mb,
            "query": _summ(get(queries_key)), "doc": _summ(get("docs")),
            "image": _summ(get("images")),
            "items_per_min": items / secs * 60 if secs else None}


def run_dicts(qids: list[str], tops: list[list[tuple[str, float]]]) -> tuple[dict, dict]:
    """ranx run (strictly decreasing rank-based scores) and raw top-1 abstention scores."""
    run = {q: {d: float(len(top) - i) for i, (d, _) in enumerate(top)} for q, top in zip(qids, tops)}
    abstain = {q: (top[0][1] if top else 0.0) for q, top in zip(qids, tops)}
    return run, abstain


def cosine_tops(qv: np.ndarray, dv: np.ndarray, doc_ids: list[str]) -> list[list[tuple[str, float]]]:
    if not doc_ids:
        return [[] for _ in qv]
    sims = qv @ dv.T
    return [ranking.rank_topk(row, doc_ids) for row in sims]


def doc_text(item: dict) -> str:
    return item.get("text", "")


def ocr_summary(corpus: list[dict]) -> dict:
    """OCR time per media item (text-layer PDF pages count as 0 ms)."""
    ms = [d.get("ocr_ms", 0.0) for d in corpus if d["kind"] != "text"]
    return {"n": len(ms), "p50_ms": pctl(ms, 50), "p95_ms": pctl(ms, 95), "total_s": sum(ms) / 1000}


def system_bm25(corpus, queries) -> tuple[list, dict]:
    ids = [d["doc_id"] for d in corpus]
    texts = [doc_text(d) for d in corpus]
    t0 = time.perf_counter()
    tops = ranking.bm25_rank(ids, texts, [q["text"] for q in queries])
    each = (time.perf_counter() - t0) * 1000 / max(len(queries), 1)
    return tops, {"load_s": 0.0, "query": {"n": len(queries), "p50_ms": each, "p95_ms": each},
                  "doc": {"n": len(ids)}, "image": {"n": 0}, "items_per_min": None,
                  "ocr": ocr_summary(corpus)}


def system_nomic(corpus, queries, python: str, embedder: str = "nomic"):
    job = {"stages": [{"embedder": embedder, "queries": [q["text"] for q in queries],
                       "docs": [doc_text(d) for d in corpus]}]}
    timings, peak, out = spawn(job, python)
    tops = cosine_tops(np.load(out / "0_queries.npy"), np.load(out / "0_docs.npy"),
                       [d["doc_id"] for d in corpus])
    return tops, {**summarise(timings, peak), "ocr": ocr_summary(corpus)}


def _media(corpus: list[dict], home_ds: Path) -> tuple[list[dict], list[str]]:
    media = [d for d in corpus if d["kind"] != "text"]
    return media, [str(home_ds / d["path"]) for d in media]


_EG2_FULL_CACHE: dict[str, tuple] = {}


def eg2_full_media(queries: list[dict], paths: list[str], python: str):
    """One eg2_full process: query vectors and image vectors. Cached for S2 and S3."""
    key = json.dumps([[q["text"] for q in queries], paths, python])
    if key in _EG2_FULL_CACHE:
        return (*_EG2_FULL_CACHE[key], True)
    job = {"stages": [{"embedder": "eg2_full", "queries": [q["text"] for q in queries],
                       "images": paths}]}
    timings, peak, out = spawn(job, python)
    img = out / "0_images.npy"
    result = (np.load(out / "0_queries.npy"), np.load(img) if img.exists() else None,
              summarise(timings, peak))
    _EG2_FULL_CACHE[key] = result
    return (*result, False)


def system_s2(corpus, queries, python: str, ds: Path):
    """One-model system: text-only loadout in one process, full model in another."""
    text_docs = [d for d in corpus if d["kind"] == "text"]
    media, paths = _media(corpus, ds)
    job = {"stages": [{"embedder": "eg2_text", "queries": [q["text"] for q in queries],
                       "docs": [doc_text(d) for d in text_docs]}]}
    timings, peak, out = spawn(job, python)
    text_sum = summarise(timings, peak)
    dv = np.load(out / "0_docs.npy")
    ids = [d["doc_id"] for d in text_docs]
    procs = {"eg2_text loadout": text_sum}
    if paths:
        _, img, full_sum, _ = eg2_full_media(queries, paths, python)
        dv, ids = np.concatenate([dv, img]), ids + [d["doc_id"] for d in media]
        procs["eg2_full loadout"] = full_sum
    return cosine_tops(np.load(out / "0_queries.npy"), dv, ids), {"processes": procs}


def system_s3(corpus, queries, python_slm: str, python_eg2: str, ds: Path, dev_qrels: dict):
    """Nomic text channel plus eg2_full media channel, weighted RRF tuned on dev."""
    nomic_tops, nomic_sum = system_nomic(corpus, queries, python_slm)
    media, paths = _media(corpus, ds)
    qv, img, eg_sum, reused = eg2_full_media(queries, paths, python_eg2)
    ids = [d["doc_id"] for d in media]
    media_tops = cosine_tops(qv, img, ids) if ids else [[] for _ in queries]
    qids = [q["id"] for q in queries]
    weight, table = fusion.tune_weight(nomic_tops, media_tops, set(ids), qids, dev_qrels)
    fused, abst = fusion.fuse_all(nomic_tops, media_tops, set(ids), weight)
    summary = {"processes": {"nomic loadout": nomic_sum, "eg2_full loadout (media channel)": eg_sum},
               "fusion": {"media_weight": weight, "dev_recall_by_weight": {str(k): v for k, v in table.items()}},
               "eg2_full_reused_from_s2": reused}
    return fused, abst, summary


def write_system(runs: Path, name: str, qids: list[str], tops, timings: dict,
                 abstain: list[float] | None = None) -> None:
    run, abst = run_dicts(qids, tops)
    if abstain is not None:
        abst = dict(zip(qids, abstain))
    (runs / f"{name}.json").write_text(json.dumps(run, sort_keys=True))
    (runs / f"{name}.abstain.json").write_text(json.dumps(abst, sort_keys=True))
    (runs / f"{name}.timings.json").write_text(json.dumps(timings, indent=1))


def _set_status(runs: Path, name: str, ran: bool, reason: str, dataset: str) -> None:
    path = runs / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status[name] = {"ran": ran, "reason": reason, "dataset": dataset}
    path.write_text(json.dumps(status, indent=1, sort_keys=True))


def _need(python: str | None, label: str) -> str:
    if not python:
        raise RuntimeError(f"needs the {label} venv: pass --python-{label}")
    if not Path(python).exists():
        raise RuntimeError(f"python for the {label} venv not found at {python}")
    return python


def run_system(name: str, corpus, queries, args, ds: Path, dev_qrels: dict):
    """Return (tops, abstain|None, timings) for one system; raise RuntimeError if it cannot run."""
    if name == "bm25":
        with rss.RssSampler(os.getpid()) as sampler:
            tops, timings = system_bm25(corpus, queries)
        return tops, None, {**timings, "peak_rss_mb": sampler.peak_mb}
    if name in ("s1", "s1p"):
        embedder = "nomic" if name == "s1" else "nomic_prefixed"
        tops, timings = system_nomic(corpus, queries, _need(args.python_slm, "slm"), embedder)
        return tops, None, timings
    if name == "s2":
        tops, timings = system_s2(corpus, queries, _need(args.python_eg2, "eg2"), ds)
        return tops, None, timings
    if name == "s3":
        return system_s3(corpus, queries, _need(args.python_slm, "slm"),
                         _need(args.python_eg2, "eg2"), ds, dev_qrels)
    if name in candidates.SYSTEMS:
        label = "slm" if name.startswith("c1") else "eg2"
        python = _need(args.python_slm if label == "slm" else args.python_eg2, label)
        return candidates.run(name, corpus, queries, {label: python}, ds, dev_qrels,
                              sys.modules[__name__])
    raise RuntimeError(f"unknown system {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--python-slm", help="python of the venv with SLM's pinned sentence-transformers")
    ap.add_argument("--python-eg2", help="python of the venv for EmbeddingGemma 2 and Qwen3-VL-Embedding")
    args = ap.parse_args(argv)
    ds, runs = args.home / "dataset", args.home / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_dataset(ds)
    dev_qrels = json.loads((ds / "qrels" / "dev.json").read_text())
    digest = dataset_hash(ds)
    ocr_engine = ocr.Ocr(args.home / "cache" / "ocr")
    ocr.attach_text(corpus, ds, ocr_engine)
    qids = [q["id"] for q in queries]
    for name in args.systems.split(","):
        try:
            tops, abst, timings = run_system(name, corpus, queries, args, ds, dev_qrels)
        except RuntimeError as exc:
            print(f"{name}: not run ({exc})", file=sys.stderr)
            _set_status(runs, name, False, str(exc), digest)
            continue
        write_system(runs, name, qids, tops, timings, abst)
        _set_status(runs, name, True, "", digest)
        print(f"{name}: done")
    (runs / "ocr_errors.json").write_text(json.dumps(ocr_engine.errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
