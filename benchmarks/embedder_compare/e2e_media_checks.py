"""Pure helpers and file/db checks for e2e_media.py (no daemon needed).

Maps recall results back to dataset doc ids, scores per stratum, builds the
GPS-EXIF fixture, inspects thumbnails, renders E2E-MEDIA.md.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMG_STRATA = ("image", "photo", "pdf_page")
GPS_IFD = 0x8825
PDF_ID = re.compile(r"pdf:(.+)#p(\d+)$")
INSTALL_DONE = ("ready", "failed", "unsupported")


# ------------------------------------------------------------------ dataset

def image_items(corpus: list[dict]) -> list[dict]:
    return sorted((d for d in corpus if d["kind"] == "image"), key=lambda d: d["doc_id"])


def distinct_pdfs(corpus: list[dict]) -> list[str]:
    """Dataset-relative paths of every PDF behind a pdf_page item."""
    return sorted({d["meta"]["pdf"] for d in corpus if d["kind"] == "pdf_page"})


def media_queries(queries: list[dict]) -> list[dict]:
    return [q for q in queries if q["stratum"] in IMG_STRATA
            or (q["stratum"] == "unanswerable" and q["id"].startswith("med:"))]


def spread(items: list, n: int) -> list:
    """n items evenly spaced over the list (all of them when it is short)."""
    if len(items) <= n:
        return list(items)
    return [items[i * len(items) // n] for i in range(n)]


# --------------------------------------------------------------------- maps

@dataclass
class Maps:
    by_media: dict[str, list[str]] = field(default_factory=dict)    # media_id -> img doc ids
    by_memory: dict[str, list[str]] = field(default_factory=dict)   # memory_id -> img doc ids
    doc_pdf: dict[str, str] = field(default_factory=dict)           # document_id -> pdf stem
    titles: dict[str, str] = field(default_factory=dict)            # document_id -> title
    ingested: set[str] = field(default_factory=set)                 # dataset doc ids present in the daemon


def build_maps(img_records: list[dict], doc_records: list[dict]) -> Maps:
    m = Maps()
    for r in img_records:
        if r.get("status") not in ("stored", "duplicate"):
            continue
        for key, table in (("media_id", m.by_media), ("memory_id", m.by_memory)):
            if r.get(key):
                table.setdefault(r[key], []).append(r["doc_id"])
        m.ingested.add(r["doc_id"])
    for r in doc_records:
        if not r.get("document_id") or r.get("job_state") not in ("done", None):
            continue
        stem = Path(r["pdf"]).stem
        m.doc_pdf[r["document_id"]] = stem
        m.titles[r["document_id"]] = str(r.get("title") or "")
        m.ingested |= {f"pdf:{stem}#p{i}" for i in range(1, int(r.get("page_count") or 0) + 1)}
    return m


def hit_doc_ids(item: dict, maps: Maps) -> list[str]:
    """Dataset doc ids one recall result stands for (images by media_id then memory_id, pages by document+page)."""
    block = item.get("media") or {}
    if block.get("kind") == "page":
        stem, page = maps.doc_pdf.get(block.get("document_id")), block.get("page")
        return [f"pdf:{stem}#p{page}"] if stem and isinstance(page, int) else []
    if block.get("media_id") in maps.by_media:
        return list(maps.by_media[block["media_id"]])
    return list(maps.by_memory.get(item.get("memory_id"), []))


def ranked_from_results(results: list[dict], maps: Maps) -> list[str]:
    out: list[str] = []
    for item in results:
        out += [d for d in hit_doc_ids(item, maps) if d not in out]
    return out


def hit_summary(rank: int, item: dict, maps: Maps) -> dict:
    """Compact record of one result: ids, the media block, and whether it names the source file."""
    block = item.get("media") or None
    doc = (block or {}).get("document_id")
    names = [n for n in (maps.doc_pdf.get(doc), maps.titles.get(doc)) if n]
    blob = json.dumps(item, default=str)
    return {"rank": rank, "fact_id": item.get("fact_id"), "memory_id": item.get("memory_id"),
            "media": block, "doc_ids": hit_doc_ids(item, maps),
            "named_file": any(n in blob for n in names), "result_keys": sorted(item)}


def note_keys(acc: dict[str, set], body: dict) -> None:
    """Accumulate which top-level, result and media-block keys recall responses carry."""
    acc.setdefault("response", set()).update(body)
    for item in body.get("results") or []:
        acc.setdefault("result", set()).update(item)
        acc.setdefault("media", set()).update(item.get("media") or {})


# ------------------------------------------------------------------ scoring

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def rank_of(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i for i, d in enumerate(ranked, 1) if d in relevant), None)


def _stratum_row(recs: list[dict]) -> dict:
    n = len(recs)
    hits = sum(1 for r in recs if r["rank"] is not None and r["rank"] <= 5)
    ing = [r for r in recs if r["label_ingested"]]
    hits_ing = sum(1 for r in ing if r["rank"] is not None and r["rank"] <= 5)
    lo, hi = wilson(hits, n)
    rr = [1 / r["rank"] if r["rank"] and r["rank"] <= 10 else 0.0 for r in recs]
    return {"n": n, "hits@5": hits, "recall@5": hits / n if n else None, "lo": lo, "hi": hi,
            "mrr@10": sum(rr) / n if n else None, "n_errors": sum(1 for r in recs if r["error"]),
            "n_label_ingested": len(ing), "recall@5_ingested": hits_ing / len(ing) if ing else None}


def per_query(records: dict, queries: list[dict], qrels: dict, maps: Maps) -> list[dict]:
    out = []
    for q in queries:
        if q["stratum"] not in IMG_STRATA:
            continue
        rec = records.get(q["id"], {"error": "missing", "ranked": []})
        rel = {d for d, g in qrels.get(q["id"], {}).items() if g > 0}
        out.append({"qid": q["id"], "stratum": q["stratum"], "rank": rank_of(rec.get("ranked", []), rel),
                    "error": rec.get("error"), "label_ingested": bool(rel) and rel <= maps.ingested})
    return out


def score_media(rows: list[dict]) -> dict:
    """recall@5 / MRR@10 per stratum and over all media strata, from per_query rows."""
    out = {s: _stratum_row([r for r in rows if r["stratum"] == s]) for s in IMG_STRATA}
    out["media_overall"] = _stratum_row(rows)
    return out


def media_outcomes(records: dict, queries: list[dict]) -> dict:
    """Answer-check outcome split: answerable media queries vs unanswerable med: queries."""
    out = {}
    for label, pick in (("answerable", lambda q: q["stratum"] in IMG_STRATA),
                        ("unanswerable", lambda q: q["stratum"] == "unanswerable")):
        recs = [records[q["id"]] for q in media_queries(queries)
                if pick(q) and "signals" in records.get(q["id"], {})]
        by = {k: sum(1 for r in recs if r["signals"][k]) for k in ("abstained", "no_confident_match", "unsupported")}
        any_n = sum(1 for r in recs if any(r["signals"].values()))
        out[label] = {"n": len(recs), "refused": any_n, "share": any_n / len(recs) if recs else None,
                      "by_signal": by}
    return out


def hits_for_labels(rows: list[dict], records: dict, qrels: dict, strata: tuple) -> list[tuple[dict, dict]]:
    """(query row, hit) for each query of these strata whose labelled item is among the top 5 results."""
    out = []
    for row in rows:
        if row["stratum"] not in strata or row["rank"] is None or row["rank"] > 5:
            continue
        rel = {d for d, g in qrels.get(row["qid"], {}).items() if g > 0}
        for hit in records.get(row["qid"], {}).get("hits", [])[:5]:
            if rel & set(hit["doc_ids"]):
                out.append((row, hit))
                break
    return out


def pick_provenance(rows: list[dict], records: dict, qrels: dict, n: int = 3) -> list[dict]:
    """n page hits from distinct PDFs, with the fields the response gave for file and page."""
    out, seen = [], set()
    for row, hit in hits_for_labels(rows, records, qrels, ("pdf_page",)):
        m = PDF_ID.match(hit["doc_ids"][0])
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        out.append({"qid": row["qid"], "pdf": m.group(1), "doc_id": hit["doc_ids"][0], "media": hit["media"],
                    "named_file": hit["named_file"], "result_keys": hit["result_keys"]})
        if len(out) == n:
            break
    return out


def pick_erasure(rows: list[dict], records: dict, qrels: dict, strata: tuple) -> dict | None:
    """First answered query of these strata, with the hit to erase and the query id to re-ask."""
    found = hits_for_labels(rows, records, qrels, strata)
    if not found:
        return None
    row, hit = found[0]
    block = hit.get("media") or {}
    return {"qid": row["qid"], "doc_id": hit["doc_ids"][0], "fact_id": hit["fact_id"],
            "memory_id": hit["memory_id"], "media_id": block.get("media_id"),
            "document_id": block.get("document_id")}


def still_returned(results: list[dict], pick: dict) -> bool:
    """True when any result still stands for the erased item (by fact, memory, media or document id)."""
    for item in results:
        block = item.get("media") or {}
        if (pick.get("fact_id") and item.get("fact_id") == pick["fact_id"]) \
                or (pick.get("memory_id") and item.get("memory_id") == pick["memory_id"]) \
                or (pick.get("media_id") and block.get("media_id") == pick["media_id"]) \
                or (pick.get("document_id") and block.get("document_id") == pick["document_id"]):
            return True
    return False


# ------------------------------------------------------------ images / EXIF

def make_gps_jpeg(path: Path, seed: int = 7) -> Path:
    """A unique 640x480 JPEG that carries a GPS EXIF block (and a capture date)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), (30 + seed * 11 % 200, 90, 160))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        top = 30 + (i * seed * 17) % 300
        draw.rectangle([20 + i * 45, top, 50 + i * 45, top + 20 + (i * seed * 29) % 120],
                       fill=((i * 40 + seed) % 255, (i * 70) % 255, (i * 20 + seed * 3) % 255))
    draw.text((40, 440), f"gps fixture {seed}", fill=(255, 255, 255))
    exif = Image.Exif()
    exif[0x9003] = "2024:05:01 10:20:30"
    exif[GPS_IFD] = {1: "N", 2: (37.0, 46.0, 29.7), 3: "W", 4: (122.0, 25.0, 9.9)}
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", exif=exif)
    return path


def has_gps(data: bytes) -> bool:
    """True when Pillow finds GPS tags in the image bytes (any format Pillow reads)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        exif = im.getexif()
        return bool(exif.get_ifd(GPS_IFD)) or GPS_IFD in exif


def thumb_info(status: int, ctype: str | None, data: bytes) -> dict:
    """HTTP status, content type, size and (when decodable) dimensions and GPS state."""
    info = {"status": status, "content_type": ctype, "bytes": len(data)}
    if status == 200 and data:
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as im:
                info.update(width=im.width, height=im.height, format=im.format)
            info["has_gps"] = has_gps(data)
        except Exception as exc:  # noqa: BLE001 - a bad thumbnail is a finding, not a crash
            info["decode_error"] = type(exc).__name__
    return info


# -------------------------------------------------------------- data dir checks

def db_rows(db: Path, sql: str, params: tuple = ()) -> list[dict]:
    """Read-only query of media.db; [] when the file or table is missing."""
    if not db.is_file():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def paths_state(root: Path, relpaths: list[str]) -> list[dict]:
    """Existence of each path under the data dir (relative paths only are recorded)."""
    return [{"path": p, "exists": (root / p).exists() or (root / p).is_symlink()} for p in relpaths if p]


def timeline_append(timeline: list[dict], sample: dict, t: float) -> bool:
    """Append when state, step, restart flag, error or progress decile changed."""
    key = (sample.get("env_state"), sample.get("step"), sample.get("restart_required"),
           sample.get("error"), int(float(sample.get("progress") or 0) * 10))
    if timeline and timeline[-1]["_key"] == list(key):
        return False
    timeline.append({"t_s": round(t, 1), "env_state": sample.get("env_state"), "step": sample.get("step"),
                     "progress": sample.get("progress"), "restart_required": sample.get("restart_required"),
                     "enabled": sample.get("enabled"), "error": sample.get("error"), "_key": list(key)})
    return True


def summarise_ingest(records: list[dict]) -> dict:
    lat = sorted(r["latency_ms"] for r in records if r.get("latency_ms") is not None)
    refused = Counter(r.get("reason", "") for r in records if r.get("status") == "refused")
    return {"n": len(records), "status": dict(Counter(r.get("status", "error") for r in records)),
            "refusal_reasons": dict(refused.most_common()),
            "p50_ms": lat[len(lat) // 2] if lat else None, "max_ms": lat[-1] if lat else None}


# ------------------------------------------------------------------ rendering

def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + \
           ["| " + " | ".join(_f(c, 1) if isinstance(c, float) else str(c) for c in r) + " |" for r in rows]


def _md_install(r: dict) -> list[str]:
    i = r.get("install") or {}
    out = ["## Install timeline (POST /api/v3/features/media/enable)", "",
           f"- final state: `{i.get('final_state')}`, total {_f(i.get('total_s'), 1)} s, "
           f"restart required: {i.get('restart_required')} (done: {i.get('restart_done')})", ""]
    out += _table(["t (s)", "state", "step", "progress", "restart_required", "error"],
                  [[t["t_s"], t["env_state"], t["step"], t["progress"], t["restart_required"], t["error"]]
                   for t in i.get("timeline", [])])
    return out


def _md_ingest(r: dict) -> list[str]:
    out = ["", "## Ingest summary", ""]
    rows = []
    for name, key in (("images (POST /api/v3/media/remember)", "images"), ("documents (POST /api/v3/documents)", "documents")):
        s = (r.get("ingest") or {}).get(key, {}).get("summary")
        if s:
            rows.append([name, s["n"], json.dumps(s["status"]), _f(s["p50_ms"], 0), _f(s["max_ms"], 0),
                         json.dumps(s["refusal_reasons"])])
    out += _table(["kind", "n", "status", "p50 ms", "max ms", "refusal reasons"], rows)
    docs = (r.get("ingest") or {}).get("documents", {}).get("records", [])
    out += [""] + _table(["pdf", "status", "job", "pages", "text/ocr/empty", "seconds"],
                         [[d["pdf"], d.get("status"), d.get("job_state"), d.get("page_count"),
                           f"{d.get('pages_text_layer')}/{d.get('pages_ocr')}/{d.get('pages_empty')}",
                           _f(d.get("job_s"), 1)] for d in docs])
    return out


def _md_recall(r: dict) -> list[str]:
    rc = r.get("recall") or {}
    out = ["", "## Recall@5 per stratum (GET /recall?limit=10)", ""]
    for split, scores in (rc.get("scores") or {}).items():
        out += [f"### {split}", ""]
        out += _table(["stratum", "n", "recall@5 [95% CI]", "MRR@10", "recall@5, label ingested", "errors"],
                      [[s, c["n"], f"{_f(c['recall@5'])} [{_f(c['lo'])}, {_f(c['hi'])}]", _f(c["mrr@10"]),
                        f"{_f(c['recall@5_ingested'])} (n={c['n_label_ingested']})", c["n_errors"]]
                       for s, c in scores.items()]) + [""]
    out += ["## Answer check (media queries)", ""]
    out += _table(["group", "n", "refused", "share", "by signal"],
                  [[g, o["n"], o["refused"], _f(o["share"]), json.dumps(o["by_signal"])]
                   for g, o in (rc.get("outcomes") or {}).items()])
    out += ["", f"Latency: p50 {_f(rc.get('p50_ms'), 0)} ms, p95 {_f(rc.get('p95_ms'), 0)} ms. "
            f"Fields observed: `{json.dumps(rc.get('fields'))}`", "",
            f"Response keys: `{json.dumps(rc.get('keys'))}`"]
    return out


def _md_checks(r: dict) -> list[str]:
    th = r.get("thumbnails") or {}
    out = ["", "## Thumbnails (GET /api/v3/media/{id}/thumb)", ""]
    out += _table(["item", "status", "content-type", "bytes", "size", "GPS"],
                  [[t["doc_id"], t["status"], t["content_type"], t["bytes"],
                    f"{t.get('width')}x{t.get('height')}", t.get("has_gps")] for t in th.get("items", [])])
    out += ["", f"GPS fixture: `{json.dumps(th.get('gps'))}`", "", "## Page provenance", ""]
    out += _table(["query", "pdf", "media block", "named_file", "result keys"],
                  [[p["qid"], p["pdf"], json.dumps(p["media"]), p["named_file"], ",".join(p["result_keys"])]
                   for p in r.get("provenance", [])])
    out += ["", "## Erasure", ""]
    for kind, e in (r.get("erasure") or {}).items():
        out += [f"### {kind}", "", "```", json.dumps(e, indent=1, default=str), "```", ""]
    return out


def _md_ops(r: dict) -> list[str]:
    rss, le = r.get("rss") or {}, r.get("log_errors") or {}
    out = ["## Peak memory", ""] + _table(["process", "pid", "peak RSS MB"],
                                         [[p["label"], p["pid"], p["peak_mb"]] for p in rss.get("processes", [])])
    out += ["", f"Peak of summed daemon tree: {_f(rss.get('peak_total_mb'), 0)} MB.", "", "## Daemon log errors", "",
            f"{le.get('error_lines')} ERROR/Traceback lines, {le.get('distinct')} distinct."]
    return out + [f"- `{ln}`" for ln in le.get("first_distinct", [])]


def render_md(r: dict) -> str:
    head = ["# End-to-end media test", "",
            f"Install: `{r.get('slm_bin')}`  |  port {r.get('port')}  |  dataset `{str(r.get('manifest'))[:16]}`"
            f"  |  all labels PROVISIONAL"]
    if r.get("skipped"):
        head += ["", f"**Remaining steps skipped: {r['skipped']}**"]
    head += [""]
    return "\n".join(head + _md_install(r) + _md_ingest(r) + _md_recall(r) + _md_checks(r) + _md_ops(r) + [""])
