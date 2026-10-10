"""End-to-end test of images and documents on a real SLM daemon.

Starts the daemon from the given install in a throwaway data and home directory,
turns images and documents on through POST /api/v3/features/media/enable, waits for
the managed environment, ingests the dataset's images and PDFs through the normal
routes, recalls every media query, then checks thumbnails, EXIF GPS stripping, page
provenance and erasure. Writes e2e_media.json and E2E-MEDIA.md.

The daemon capability token is read from daemon.json only to build request headers;
it is never printed, logged or written to the results.
"""
from __future__ import annotations

import argparse
import http.client
import json
import shutil
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import e2e_daemon as ed
import e2e_media_checks as chk

HTTP_ERR = (OSError, http.client.HTTPException)
FEATURES = "/api/v3/features"
NEW_STATE = lambda: {"n_429": 0, "backoff_s": 0.0, "warming_s": 0.0}  # noqa: E731


@dataclass
class Ctx:
    args: argparse.Namespace
    env: dict
    port: int
    data: Path
    sampler: ed.ProcessSampler
    client: ed.Client
    pid: int | None = None
    throttle: dict = field(default_factory=NEW_STATE)


# ------------------------------------------------------------------ http bits

def raw_call(client: ed.Client, method: str, path: str) -> tuple[int, str | None, bytes]:
    """Binary-safe request on the client's keep-alive connection (status, content-type, body)."""
    for attempt in (1, 2):
        try:
            resp, raw = client._send(method, path, None, dict(client._headers))  # headers stay private
            return resp.status, resp.getheader("Content-Type"), raw
        except HTTP_ERR:
            client._conn = None
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


def call_retry(client: ed.Client, method: str, path: str, body: dict | None, st: dict,
               cap_s: float = 600.0) -> tuple[int, dict, float]:
    """One call, waiting out a 429 (Retry-After) or a 'warming' 202 / 503 until cap_s."""
    t0 = time.perf_counter()
    while True:
        status, data, ms = client.call(method, path, body)
        if status == 429:
            wait = min(60.0 if client.retry_after is None else client.retry_after, ed.MAX_BACKOFF_S)
            st["n_429"] += 1
            st["backoff_s"] += wait
        elif (status == 202 and data.get("status") == "warming") or status == 503:
            wait = 5.0
            st["warming_s"] += wait
        else:
            return status, data, ms
        if time.perf_counter() - t0 > cap_s:
            return status, data, ms
        time.sleep(wait)


def restart(ctx: Ctx) -> dict:
    """Stop the daemon and start it again the normal way (`slm serve stop` / `start`)."""
    stop = ed.stop_daemon(ctx.env, ctx.args.slm_bin, ctx.port, ctx.pid)
    startup = ed.start_daemon(ctx.args.slm_bin, ctx.env, ctx.port, ctx.args.ready_timeout)
    ctx.pid = startup.get("pid")
    ctx.sampler.root_pid = ctx.pid
    ctx.client = ed.Client(ctx.port, ed.read_auth(ctx.data))
    return {"stop": stop, "startup": startup}


# -------------------------------------------------------------------- install

def poll_features(ctx: Ctx, timeline: list, t0: float, timeout_s: float) -> dict:
    """Poll GET /api/v3/features until the environment is ready/failed/unsupported or timeout."""
    last: dict = {}
    while time.perf_counter() - t0 < timeout_s:
        try:
            status, body, _ = ctx.client.call("GET", FEATURES)
        except HTTP_ERR:
            time.sleep(3)
            continue
        last = body.get("media") or last
        if status == 200 and last:
            chk.timeline_append(timeline, last, time.perf_counter() - t0)
            if last.get("env_state") in chk.INSTALL_DONE:
                return last
        time.sleep(3)
    return last


def install_step(ctx: Ctx) -> dict:
    t0 = time.perf_counter()
    status, body, _ = ctx.client.call("POST", FEATURES + "/media/enable", {"yes": True, "source": "api"})
    out: dict = {"enable_http": status, "timeline": [], "restart_required": None, "restart_done": False}
    if status not in (200, 202):
        out.update(final_state="enable_failed", total_s=time.perf_counter() - t0, error=body)
        return out
    chk.timeline_append(out["timeline"], body.get("media") or {}, time.perf_counter() - t0)
    last = poll_features(ctx, out["timeline"], t0, ctx.args.enable_timeout_s)
    out.update(final_state=last.get("env_state", "timeout"), total_s=time.perf_counter() - t0,
               restart_required=bool(last.get("restart_required")), error=last.get("error"))
    if out["final_state"] == "ready" and out["restart_required"]:
        out["restart"] = restart(ctx)
        out["restart_done"] = True
        _, body, _ = ctx.client.call("GET", FEATURES)
        after = body.get("media") or {}
        chk.timeline_append(out["timeline"], after, time.perf_counter() - t0)
        out["restart_still_required"] = bool(after.get("restart_required"))
    return out


# --------------------------------------------------------------------- ingest

def ingest_images(ctx: Ctx, ds: Path, items: list[dict]) -> list[dict]:
    recs = []
    for i, item in enumerate(items, 1):
        rec: dict = {"doc_id": item["doc_id"]}
        try:
            status, d, ms = call_retry(ctx.client, "POST", "/api/v3/media/remember",
                                       {"path": str((ds / item["path"]).resolve())}, ctx.throttle)
            rec.update(http=status, status=d.get("status") or "error", latency_ms=ms,
                       reason=str(d.get("reason") or d.get("detail") or ""), media_id=d.get("media_id"),
                       memory_id=d.get("memory_id"), duplicate_of=d.get("duplicate_of"),
                       near_duplicate_of=d.get("near_duplicate_of"))
        except HTTP_ERR as exc:
            rec.update(status="error", reason=type(exc).__name__)
        recs.append(rec)
        if i % 25 == 0:
            print(f"images {i}/{len(items)}", file=sys.stderr, flush=True)
    return recs


def submit_pdfs(ctx: Ctx, ds: Path, pdfs: list[str]) -> list[dict]:
    recs = []
    for rel in pdfs:
        rec: dict = {"pdf": rel, "_t0": time.perf_counter()}
        try:
            path = (ds / rel).resolve()
            status, d, ms = call_retry(ctx.client, "POST", "/api/v3/documents",
                                       {"path": str(path), "file_name": path.name}, ctx.throttle)
            rec.update(http=status, status=d.get("status") or "error", latency_ms=ms, document_id=d.get("document_id"),
                       job_id=d.get("job_id"), reason=str(d.get("reason") or d.get("detail") or ""))
        except HTTP_ERR as exc:
            rec.update(status="error", reason=type(exc).__name__)
        recs.append(rec)
    return recs


def wait_jobs(ctx: Ctx, recs: list[dict]) -> None:
    """Poll GET /api/v3/jobs/{id} for every queued document until done/failed/cancelled or timeout."""
    todo = [r for r in recs if r.get("job_id")]
    t_end = time.perf_counter() + ctx.args.doc_timeout_s
    while todo and time.perf_counter() < t_end:
        for rec in list(todo):
            try:
                status, d, _ = ctx.client.call("GET", f"/api/v3/jobs/{rec['job_id']}")
            except HTTP_ERR:
                continue
            doc = d.get("document") or {}
            rec.update(job_state=d.get("state"), job_done=d.get("done"), job_total=d.get("total"),
                       job_error=d.get("error"), page_count=doc.get("page_count"), title=doc.get("title"),
                       pages_text_layer=doc.get("pages_text_layer"), pages_ocr=doc.get("pages_ocr"),
                       pages_empty=doc.get("pages_empty"), doc_state=doc.get("state"))
            if status == 404 or d.get("state") in ("done", "failed", "cancelled"):
                rec["job_s"] = time.perf_counter() - rec["_t0"]
                todo.remove(rec)
        time.sleep(2)
    for rec in recs:
        rec.pop("_t0", None)


# --------------------------------------------------------------------- recall

def recall_media(ctx: Ctx, queries: list[dict], maps: chk.Maps, keys: dict) -> dict:
    """GET /recall?q=..&limit=10 for each media query; one record per query id."""
    records = {}
    for q in queries:
        path = "/recall?q=" + urllib.parse.quote(q["text"]) + "&limit=10"
        try:
            status, body, ms = ctx.client.call("GET", path)
        except HTTP_ERR as exc:
            records[q["id"]] = {"error": type(exc).__name__, "ranked": []}
            continue
        if status != 200:
            records[q["id"]] = {"error": f"http {status}", "ranked": [], "latency_ms": ms}
            continue
        results = body.get("results", [])
        chk.note_keys(keys, body)
        records[q["id"]] = {
            "latency_ms": ms, "n_results": len(results), "ranked": chk.ranked_from_results(results, maps),
            "hits": [chk.hit_summary(i, it, maps) for i, it in enumerate(results, 1)],
            "signals": ed.refusal_signals(body), "fields": ed.observed_fields(body)}
    return records


def recall_text(ctx: Ctx, text: str) -> tuple[int, list[dict]]:
    status, body, _ = ctx.client.call("GET", "/recall?q=" + urllib.parse.quote(text) + "&limit=10")
    return status, body.get("results", [])


# ------------------------------------------------------------------ thumbnails

def thumb_of(ctx: Ctx, media_id: str) -> dict:
    status, ctype, raw = raw_call(ctx.client, "GET", f"/api/v3/media/{media_id}/thumb")
    return chk.thumb_info(status, ctype, raw)


def gps_check(ctx: Ctx, work: Path) -> dict:
    src = chk.make_gps_jpeg(work / "gps_fixture.jpg")
    out: dict = {"fixture_has_gps": chk.has_gps(src.read_bytes())}
    status, d, _ = call_retry(ctx.client, "POST", "/api/v3/media/remember", {"path": str(src)}, ctx.throttle)
    out.update(http=status, status=d.get("status"), reason=str(d.get("reason") or d.get("detail") or ""))
    mid = d.get("media_id")
    if not mid:
        return out
    rows = chk.db_rows(ctx.data / "media.db", "SELECT original_relpath, exif_json FROM media_items WHERE media_id=?", (mid,))
    row = rows[0] if rows else {}
    out["exif_json_mentions_gps"] = "gps" in str(row.get("exif_json", "")).lower()
    rel = row.get("original_relpath")
    orig = ctx.data / "media" / rel if rel else None
    out["original_exists"] = bool(orig and orig.is_file())
    out["original_has_gps"] = chk.has_gps(orig.read_bytes()) if out["original_exists"] else None
    out["thumb"] = thumb_of(ctx, mid)
    out["no_gps"] = not (out["exif_json_mentions_gps"] or out["original_has_gps"] or out["thumb"].get("has_gps"))
    return out


def thumbs_step(ctx: Ctx, img_recs: list[dict], work: Path) -> dict:
    stored = [r for r in img_recs if r.get("status") == "stored" and r.get("media_id")]
    items = []
    for r in chk.spread(stored, 5):
        items.append({"doc_id": r["doc_id"], "media_id": r["media_id"], **thumb_of(ctx, r["media_id"])})
    return {"items": items, "gps": gps_check(ctx, work)}


# --------------------------------------------------------------------- erasure

def wait_clean(check, cap_s: float = 20.0) -> dict:
    """Run check() until it reports clean or cap_s passes (erasure may finish just after the reply)."""
    t0 = time.perf_counter()
    while True:
        res = check()
        if res["clean"] or time.perf_counter() - t0 > cap_s:
            return {**res, "waited_s": round(time.perf_counter() - t0, 1)}
        time.sleep(1)


def erase_image(ctx: Ctx, pick: dict, query: str) -> dict:
    media_db = ctx.data / "media.db"
    sql = "SELECT original_relpath, state FROM media_items WHERE media_id=?"
    before = chk.db_rows(media_db, sql, (pick["media_id"],))
    rels = ["media/" + r["original_relpath"] for r in before if r.get("original_relpath")]
    shared = chk.db_rows(media_db, "SELECT COUNT(*) AS n FROM media_items WHERE original_relpath=? AND media_id!=?",
                         (before[0]["original_relpath"] if before else "", pick["media_id"]))
    status, body, _ = ctx.client.call("DELETE", f"/api/memories/{pick['fact_id']}")

    def check() -> dict:
        _, results = recall_text(ctx, query)
        thumb = thumb_of(ctx, pick["media_id"])["status"]
        files = chk.paths_state(ctx.data, rels)
        rows = chk.db_rows(media_db, sql, (pick["media_id"],))
        clean = (not chk.still_returned(results, pick) and thumb == 404 and not any(f["exists"] for f in files))
        return {"clean": clean, "recall_still_returns": chk.still_returned(results, pick), "thumb_status": thumb,
                "files": files, "media_rows_after": rows}
    return {"pick": pick, "rows_before": before, "other_rows_sharing_file": shared[0]["n"] if shared else None,
            "delete_route": "DELETE /api/memories/{fact_id}", "delete_http": status, "delete_body": body,
            "after": wait_clean(check)}


def erase_document(ctx: Ctx, pick: dict, query: str) -> dict:
    media_db, did = ctx.data / "media.db", pick["document_id"]
    docs = chk.db_rows(media_db, "SELECT source_relpath, state FROM documents WHERE document_id=?", (did,))
    pages = chk.db_rows(media_db, "SELECT media_id, original_relpath FROM media_items WHERE document_id=?", (did,))
    rels = [f"media/{r['source_relpath']}" for r in docs if r.get("source_relpath")] + \
           [f"media/{p['original_relpath']}" for p in pages if p.get("original_relpath")]
    status, body, _ = ctx.client.call("DELETE", f"/api/v3/documents/{did}?hard=true")

    def check() -> dict:
        _, results = recall_text(ctx, query)
        thumbs = [thumb_of(ctx, p["media_id"])["status"] for p in pages[:3] if p.get("media_id")]
        files = chk.paths_state(ctx.data, rels)
        left = {"documents": chk.db_rows(media_db, "SELECT state FROM documents WHERE document_id=?", (did,)),
                "media_items": chk.db_rows(media_db, "SELECT COUNT(*) AS n FROM media_items WHERE document_id=?", (did,)),
                "doc_pages": chk.db_rows(media_db, "SELECT COUNT(*) AS n FROM doc_pages WHERE document_id=?", (did,))}
        clean = (not chk.still_returned(results, pick) and all(t == 404 for t in thumbs)
                 and not any(f["exists"] for f in files))
        return {"clean": clean, "recall_still_returns": chk.still_returned(results, pick),
                "page_thumb_statuses": thumbs, "files": files, "rows_after": left}
    return {"pick": pick, "n_page_items": len(pages), "document_rows_before": docs,
            "delete_route": "DELETE /api/v3/documents/{id}?hard=true", "delete_http": status,
            "delete_body": body, "after": wait_clean(check)}


# ------------------------------------------------------------------ orchestration

def score_all(records: dict, queries: dict, qrels: dict, maps: chk.Maps) -> tuple[dict, list[dict]]:
    scores, allrows = {}, []
    for split in ("test", "dev"):
        rows = chk.per_query(records, queries[split], qrels[split], maps)
        scores[split] = chk.score_media(rows)
        allrows += rows
    scores["all"] = chk.score_media(allrows)
    return scores, allrows


def recall_summary(records: dict, queries: dict, qrels: dict, maps: chk.Maps, keys: dict) -> tuple[dict, list[dict]]:
    ok = [r["latency_ms"] for r in records.values() if "signals" in r]
    all_q = queries["dev"] + queries["test"]
    scores, rows = score_all(records, queries, qrels, maps)
    summary = {"n_queries": len(records), "n_ok": len(ok), "p50_ms": ed.percentile(ok, 50),
               "p95_ms": ed.percentile(ok, 95), "scores": scores,
               "outcomes": chk.media_outcomes(records, all_q), "fields": ed.tally_fields(list(records.values())),
               "keys": {k: sorted(v) for k, v in keys.items()}, "per_query": rows}
    return summary, rows


def run_media_steps(ctx: Ctx, res: dict, work: Path) -> None:
    ds = ctx.args.dataset
    corpus, queries, qrels = ed.load_dataset(ds)
    img_recs = ingest_images(ctx, ds, chk.image_items(corpus))
    doc_recs = submit_pdfs(ctx, ds, chk.distinct_pdfs(corpus))
    wait_jobs(ctx, doc_recs)
    res["ingest"] = {"images": {"records": img_recs, "summary": chk.summarise_ingest(img_recs)},
                     "documents": {"records": doc_recs, "summary": chk.summarise_ingest(doc_recs)},
                     "throttle": ctx.throttle, "idle": ed.wait_idle(ctx.port, ctx.args.idle_timeout)}
    maps, keys = chk.build_maps(img_recs, doc_recs), {}
    qs = chk.media_queries(queries["dev"] + queries["test"])
    records = recall_media(ctx, qs, maps, keys)
    res["recall"], rows = recall_summary(records, queries, qrels, maps, keys)
    qrels_all = {**qrels["dev"], **qrels["test"]}
    text_of = {q["id"]: q["text"] for q in qs}
    res["thumbnails"] = thumbs_step(ctx, img_recs, work)
    res["provenance"] = chk.pick_provenance(rows, records, qrels_all)
    res["erasure"] = {}
    for kind, strata, fn in (("image", ("image", "photo"), erase_image), ("document", ("pdf_page",), erase_document)):
        pick = chk.pick_erasure(rows, records, qrels_all, strata)
        res["erasure"][kind] = fn(ctx, pick, text_of[pick["qid"]]) if pick else {"skipped": "no answered query"}


def execute(args: argparse.Namespace, work: Path) -> dict:
    data, home = work / "data", work / "home"
    data.mkdir(), home.mkdir()
    port = args.port or ed.free_port()
    env = ed.daemon_env(data, home, port, offline=False)
    sampler = ed.ProcessSampler(str(data))
    sampler.start()
    res: dict = {"slm_bin": args.slm_bin, "port": port, "install": {}, "ingest": {}, "recall": {},
                 "thumbnails": {}, "provenance": [], "erasure": {}, "skipped": None}
    ctx = Ctx(args, env, port, data, sampler, ed.Client(port))
    try:
        startup = ed.start_daemon(args.slm_bin, env, port, args.ready_timeout)
        ctx.pid = sampler.root_pid = startup.get("pid")
        ctx.client = ed.Client(port, ed.read_auth(data))
        res["startup"] = startup
        res["install"] = install_step(ctx)
        if res["install"]["final_state"] != "ready" or res["install"].get("restart_still_required"):
            res["skipped"] = f"install ended in {res['install']['final_state']}"
        else:
            run_media_steps(ctx, res, work)
    finally:
        res["stop"] = ed.stop_daemon(env, args.slm_bin, port, ctx.pid)
        procs = sampler.stop()
    logs = "\n".join(p.read_text(errors="replace") for p in sorted((data / "logs").glob("*.log"))) \
        if (data / "logs").is_dir() else ""
    res.update(rss={"processes": procs, "peak_total_mb": sampler.peak_total_mb}, log_errors=ed.collect_log_errors(logs))
    lock = args.dataset / "manifest.lock"
    res["manifest"] = json.loads(lock.read_text())["sha256"] if lock.exists() else ""
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slm-bin", required=True, help="path to the `slm` executable of the install under test")
    ap.add_argument("--dataset", type=Path, required=True, help="$SLM_BENCH_HOME/dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=0, help="daemon port (default: first free from 8821)")
    ap.add_argument("--enable-timeout-s", type=float, default=1800.0)
    ap.add_argument("--doc-timeout-s", type=float, default=3600.0)
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    ap.add_argument("--idle-timeout", type=float, default=300.0)
    ap.add_argument("--work-dir", type=Path, default=None, help="parent for the throwaway data dir (models are large)")
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="slm-e2e-media-", dir=args.work_dir))
    try:
        result = execute(args, work)
    finally:
        if not args.keep_data:
            shutil.rmtree(work, ignore_errors=True)
    (args.out / "e2e_media.json").write_text(json.dumps(result, indent=1, default=str))
    (args.out / "E2E-MEDIA.md").write_text(chk.render_md(result))
    print(f"wrote {args.out / 'E2E-MEDIA.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
