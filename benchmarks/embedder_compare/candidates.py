"""Model-decision candidates: one shared text+image model, two ways to use it.

Merged (c1, c2, c3): one cosine list over text turns and image/page vectors, no OCR
text. This is the pure "one shared space" test: can a text query find a picture
when pictures and text compete on raw cosine?

Fused (c1f, c2f, c3f): what SLM's media recall channel does. The text channel ranks
every item by its text (conversation turns plus OCR or text-layer text of pictures
and pages) with the candidate's text model; the media channel ranks pictures and
pages by their image vector; weighted reciprocal-rank fusion (k=60), media weight
tuned on dev only.

c1 = nomic-embed-text-v1.5 (as SLM ships it) + nomic-embed-vision-v1.5 (slm venv)
c2 = EmbeddingGemma 2: text-only loadout for text, full model for media (eg2 venv)
c3 = Qwen3-VL-Embedding-2B for everything (eg2 venv)
"""
from __future__ import annotations

import json

import numpy as np

import fusion

_CACHE: dict[str, tuple] = {}


def _proc(spawn, summarise, embedder: str, python: str, **kinds) -> tuple[dict, dict, bool]:
    """One worker process per loadout, memoised so merged and fused share it."""
    key = json.dumps([embedder, python, kinds], sort_keys=True)
    if key in _CACHE:
        return (*_CACHE[key], True)
    timings, peak, out = spawn({"stages": [{"embedder": embedder, **kinds}]}, python)
    vecs = {k: np.load(out / f"0_{k}.npy") for k in kinds if (out / f"0_{k}.npy").exists()}
    _CACHE[key] = (vecs, summarise(timings, peak))
    return (*_CACHE[key], False)


def _split(corpus: list[dict], ds) -> tuple[list[dict], list[dict], list[str]]:
    text = [d for d in corpus if d["kind"] == "text"]
    media = [d for d in corpus if d["kind"] != "text"]
    return text, media, [str(ds / d["path"]) for d in media]


def _vectors(name: str, corpus, queries, pythons: dict, ds, rt) -> dict:
    """Embed once per candidate; returns q (text-channel queries), all_docs (every item's
    text, in corpus order), mq (media-channel queries), img, and per-process timings."""
    qt = [q["text"] for q in queries]
    texts = [rt.doc_text(d) for d in corpus]
    _, media, paths = _split(corpus, ds)
    procs, reused = {}, False
    if name == "c1":
        v, s, reused = _proc(rt.spawn, rt.summarise, "nomic_c1", pythons["slm"],
                             queries=qt, docs=texts, images=paths, media_queries=qt)
        procs["nomic text + nomic vision loadout"] = s
        out = {"q": v["queries"], "all_docs": v["docs"], "mq": v["media_queries"], "img": v["images"]}
    elif name == "c2":
        v, s, r1 = _proc(rt.spawn, rt.summarise, "eg2_text", pythons["eg2"], queries=qt, docs=texts)
        mq, img, s_full, r2 = rt.eg2_full_media(queries, paths, pythons["eg2"])
        procs.update({"eg2_text loadout": s, "eg2_full loadout": s_full})
        out, reused = {"q": v["queries"], "all_docs": v["docs"], "mq": mq, "img": img}, r1 and r2
    elif name == "c3":
        v, s, reused = _proc(rt.spawn, rt.summarise, "qwen3vl", pythons["eg2"],
                             queries=qt, docs=texts, images=paths)
        procs["qwen3vl loadout"] = s
        out = {"q": v["queries"], "all_docs": v["docs"], "mq": v["queries"], "img": v["images"]}
    else:
        raise RuntimeError(f"unknown candidate {name!r}")
    return {**out, "processes": procs, "reused": reused, "media": media}


def run(system: str, corpus, queries, pythons: dict, ds, dev_qrels: dict, rt):
    """Return (tops, abstain|None, timings) for c1/c1f/c1m and the c2, c3 forms."""
    base, fused, media_only = system[:2], system.endswith("f"), system.endswith("m")
    v = _vectors(base, corpus, queries, pythons, ds, rt)
    text_idx = [i for i, d in enumerate(corpus) if d["kind"] == "text"]
    media_ids = [d["doc_id"] for d in v["media"]]
    timings = {"processes": v["processes"], "vectors_reused": v["reused"]}
    if media_only:  # diagnostic: the image channel alone, pictures and pages only
        return rt.cosine_tops(v["mq"], v["img"], media_ids), None, timings
    if not fused:
        dv = np.concatenate([v["all_docs"][text_idx], v["img"]])
        ids = [corpus[i]["doc_id"] for i in text_idx] + media_ids
        return rt.cosine_tops(v["q"], dv, ids), None, timings
    text_tops = rt.cosine_tops(v["q"], v["all_docs"], [d["doc_id"] for d in corpus])
    media_tops = rt.cosine_tops(v["mq"], v["img"], media_ids)
    qids = [q["id"] for q in queries]
    weight, table = fusion.tune_weight(text_tops, media_tops, set(media_ids), qids, dev_qrels)
    tops, abst = fusion.fuse_all(text_tops, media_tops, set(media_ids), weight)
    timings["fusion"] = {"media_weight": weight,
                         "dev_recall_by_weight": {str(k): x for k, x in table.items()}}
    return tops, abst, timings


SYSTEMS = ("c1", "c1f", "c1m", "c2", "c2f", "c2m", "c3", "c3f", "c3m")
LABELS = {
    "c1": "C1 nomic text + nomic vision, merged (one cosine list, no OCR)",
    "c1f": "C1 nomic text + nomic vision, fused (text+OCR channel and image channel, RRF)",
    "c2": "C2 EmbeddingGemma 2, merged (one cosine list, no OCR)",
    "c2f": "C2 EmbeddingGemma 2, fused (text+OCR channel and image channel, RRF)",
    "c3": "C3 Qwen3-VL-Embedding-2B, merged (one cosine list, no OCR)",
    "c3f": "C3 Qwen3-VL-Embedding-2B, fused (text+OCR channel and image channel, RRF)",
    "c1m": "C1 image channel alone (diagnostic: pictures and pages only, no OCR, no text)",
    "c2m": "C2 image channel alone (diagnostic)",
    "c3m": "C3 image channel alone (diagnostic)",
}
