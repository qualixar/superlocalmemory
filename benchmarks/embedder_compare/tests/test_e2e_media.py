"""Pure parts of the media end-to-end script, with fake daemon responses."""
from __future__ import annotations

import io
import json

import e2e_media_checks as chk

IMG = [{"doc_id": "img:a", "status": "stored", "media_id": "m1", "memory_id": "mem1"},
       {"doc_id": "img:b", "status": "duplicate", "media_id": "m1", "memory_id": "mem1"},
       {"doc_id": "img:c", "status": "stored", "media_id": "m3", "memory_id": "mem3"},
       {"doc_id": "img:d", "status": "refused", "reason": "too small"}]
DOCS = [{"pdf": "media/pdf/paper.pdf", "document_id": "d1", "job_state": "done", "page_count": 3, "title": "A Paper"}]


def maps() -> chk.Maps:
    return chk.build_maps(IMG, DOCS)


def test_build_maps_ingested_set_and_refused_excluded():
    m = maps()
    assert m.by_media["m1"] == ["img:a", "img:b"] and m.by_memory["mem3"] == ["img:c"]
    assert "img:d" not in m.ingested and "pdf:paper#p3" in m.ingested and "pdf:paper#p4" not in m.ingested


def test_hit_doc_ids_images_pages_and_fallback():
    m = maps()
    assert chk.hit_doc_ids({"media": {"kind": "image", "media_id": "m3"}}, m) == ["img:c"]
    assert chk.hit_doc_ids({"memory_id": "mem1"}, m) == ["img:a", "img:b"]
    page = {"media": {"kind": "page", "document_id": "d1", "page": 2, "media_id": "pg"}}
    assert chk.hit_doc_ids(page, m) == ["pdf:paper#p2"]
    assert chk.hit_doc_ids({"media": {"kind": "page", "document_id": "zz", "page": 2}}, m) == []
    assert chk.hit_doc_ids({"media": {"kind": "page", "document_id": "d1", "page": None}}, m) == []
    assert chk.hit_doc_ids({"memory_id": "text-memory"}, m) == []


def test_ranked_from_results_dedups_in_order():
    results = [{"memory_id": "x"}, {"media": {"kind": "image", "media_id": "m3"}},
               {"media": {"kind": "image", "media_id": "m1"}}, {"memory_id": "mem3"}]
    assert chk.ranked_from_results(results, maps()) == ["img:c", "img:a", "img:b"]


def test_hit_summary_flags_named_file_and_keys():
    item = {"fact_id": "f", "memory_id": "mm", "content": "from A Paper p2",
            "media": {"kind": "page", "document_id": "d1", "page": 2, "citation": "page 2"}}
    hit = chk.hit_summary(1, item, maps())
    assert hit["named_file"] and hit["doc_ids"] == ["pdf:paper#p2"] and "media" in hit["result_keys"]
    item["content"] = "nothing"
    assert not chk.hit_summary(1, item, maps())["named_file"]


def test_note_keys_accumulates():
    acc: dict = {}
    chk.note_keys(acc, {"ok": 1, "results": [{"a": 1, "media": {"page": 1}}, {"b": 2}]})
    assert acc["response"] == {"ok", "results"} and acc["result"] == {"a", "media", "b"} and acc["media"] == {"page"}


def _records() -> tuple[dict, list[dict], dict]:
    queries = [{"id": "q1", "stratum": "image"}, {"id": "q2", "stratum": "image"},
               {"id": "q3", "stratum": "pdf_page"}, {"id": "q4", "stratum": "photo"},
               {"id": "med:u", "stratum": "unanswerable"}, {"id": "loc:1", "stratum": "temporal"}]
    qrels = {"q1": {"img:a": 1}, "q2": {"img:c": 1}, "q3": {"pdf:paper#p2": 1}, "q4": {"img:zz": 1}}
    records = {"q1": {"ranked": ["img:a"]}, "q2": {"ranked": ["x1", "x2", "x3", "x4", "x5", "img:c"]},
               "q3": {"ranked": ["pdf:paper#p2"]}, "q4": {"error": "http 500", "ranked": []}}
    return records, queries, qrels


def test_per_query_and_score_media():
    records, queries, qrels = _records()
    rows = chk.per_query(records, queries, qrels, maps())
    assert [r["qid"] for r in rows] == ["q1", "q2", "q3", "q4"]  # unanswerable and text excluded
    assert [r["rank"] for r in rows] == [1, 6, 1, None]
    scores = chk.score_media(rows)
    assert scores["image"]["n"] == 2 and scores["image"]["recall@5"] == 0.5
    assert scores["image"]["mrr@10"] == (1 + 1 / 6) / 2
    assert scores["photo"]["n_errors"] == 1 and scores["photo"]["recall@5_ingested"] is None  # label not ingested
    assert scores["media_overall"]["n"] == 4 and scores["media_overall"]["hits@5"] == 2
    assert scores["pdf_page"]["recall@5"] == 1.0 and scores["pdf_page"]["lo"] < 1.0


def test_media_outcomes_split():
    yes = {"abstained": True, "no_confident_match": False, "unsupported": False}
    no = {"abstained": False, "no_confident_match": False, "unsupported": False}
    queries = [{"id": "q1", "stratum": "image"}, {"id": "q2", "stratum": "photo"},
               {"id": "med:u", "stratum": "unanswerable"}, {"id": "loc:u", "stratum": "unanswerable"}]
    out = chk.media_outcomes({"q1": {"signals": no}, "q2": {"signals": yes}, "med:u": {"signals": yes},
                              "loc:u": {"signals": yes}}, queries)
    assert out["answerable"]["n"] == 2 and out["answerable"]["refused"] == 1
    assert out["unanswerable"] == {"n": 1, "refused": 1, "share": 1.0,
                                   "by_signal": {"abstained": 1, "no_confident_match": 0, "unsupported": 0}}


def test_media_queries_and_dataset_pickers():
    queries = [{"id": "a", "stratum": "image"}, {"id": "med:x", "stratum": "unanswerable"},
               {"id": "loc:x", "stratum": "unanswerable"}, {"id": "t", "stratum": "entity"}]
    assert [q["id"] for q in chk.media_queries(queries)] == ["a", "med:x"]
    corpus = [{"doc_id": "img:b", "kind": "image"}, {"doc_id": "img:a", "kind": "image"}, {"doc_id": "t", "kind": "text"},
              {"doc_id": "pdf:p#p1", "kind": "pdf_page", "meta": {"pdf": "media/pdf/p.pdf"}},
              {"doc_id": "pdf:p#p2", "kind": "pdf_page", "meta": {"pdf": "media/pdf/p.pdf"}}]
    assert [d["doc_id"] for d in chk.image_items(corpus)] == ["img:a", "img:b"]
    assert chk.distinct_pdfs(corpus) == ["media/pdf/p.pdf"]
    assert chk.spread(list(range(10)), 5) == [0, 2, 4, 6, 8] and chk.spread([1, 2], 5) == [1, 2]


def _page_hit(doc_id: str, did: str = "d1", page: int = 2) -> dict:
    return {"rank": 1, "fact_id": "f" + doc_id, "memory_id": "mem" + doc_id, "doc_ids": [doc_id],
            "media": {"kind": "page", "document_id": did, "page": page}, "named_file": False, "result_keys": ["media"]}


def test_pick_provenance_distinct_pdfs_and_erasure_pick():
    rows = [{"qid": "q1", "stratum": "pdf_page", "rank": 1}, {"qid": "q2", "stratum": "pdf_page", "rank": 1},
            {"qid": "q3", "stratum": "pdf_page", "rank": 7}, {"qid": "q4", "stratum": "pdf_page", "rank": 2}]
    qrels = {"q1": {"pdf:a#p2": 1}, "q2": {"pdf:a#p3": 1}, "q3": {"pdf:b#p1": 1}, "q4": {"pdf:c#p1": 1}}
    records = {"q1": {"hits": [_page_hit("pdf:a#p2")]}, "q2": {"hits": [_page_hit("pdf:a#p3")]},
               "q3": {"hits": [_page_hit("pdf:b#p1")]}, "q4": {"hits": [_page_hit("pdf:c#p1", "d3")]}}
    prov = chk.pick_provenance(rows, records, qrels)
    assert [p["pdf"] for p in prov] == ["a", "c"]  # same pdf skipped, rank 7 is not in the top 5
    pick = chk.pick_erasure(rows, records, qrels, ("pdf_page",))
    assert pick["document_id"] == "d1" and pick["qid"] == "q1"
    assert chk.pick_erasure(rows, records, qrels, ("image",)) is None
    assert chk.still_returned([{"media": {"document_id": "d1"}}], pick)
    assert not chk.still_returned([{"fact_id": "other", "memory_id": "o", "media": {"document_id": "d9"}}], pick)


def test_gps_fixture_has_gps_and_resave_strips(tmp_path):
    from PIL import Image

    path = chk.make_gps_jpeg(tmp_path / "sub" / "g.jpg")
    raw = path.read_bytes()
    assert chk.has_gps(raw)
    buf = io.BytesIO()
    with Image.open(path) as im:
        im.save(buf, "WEBP")  # a re-encode without exif= drops it
    assert not chk.has_gps(buf.getvalue())
    info = chk.thumb_info(200, "image/webp", buf.getvalue())
    assert info["width"] == 640 and info["has_gps"] is False and info["content_type"] == "image/webp"
    assert chk.thumb_info(404, "application/json", b"{}") == {"status": 404, "content_type": "application/json", "bytes": 2}
    assert "decode_error" in chk.thumb_info(200, "image/webp", b"not an image")


def test_timeline_append_only_on_change():
    tl: list[dict] = []
    s = {"env_state": "installing", "step": "download", "progress": 0.11, "restart_required": False}
    assert chk.timeline_append(tl, s, 1.0)
    assert not chk.timeline_append(tl, {**s, "progress": 0.19}, 2.0)  # same decile
    assert chk.timeline_append(tl, {**s, "progress": 0.31}, 3.0)
    assert chk.timeline_append(tl, {**s, "env_state": "ready", "restart_required": True}, 4.0)
    assert [t["t_s"] for t in tl] == [1.0, 3.0, 4.0]


def test_db_rows_and_paths_state(tmp_path):
    import sqlite3

    assert chk.db_rows(tmp_path / "none.db", "select 1") == []
    db = tmp_path / "media.db"
    con = sqlite3.connect(db)
    con.execute("create table t (a text)"), con.execute("insert into t values ('x')"), con.commit(), con.close()
    assert chk.db_rows(db, "select a from t") == [{"a": "x"}] and chk.db_rows(db, "select * from nope") == []
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "f.png").write_bytes(b"1")
    assert chk.paths_state(tmp_path, ["media/f.png", "media/g.png", ""]) == [
        {"path": "media/f.png", "exists": True}, {"path": "media/g.png", "exists": False}]


def test_summarise_ingest_and_render_md():
    recs = [{"status": "stored", "latency_ms": 10.0}, {"status": "refused", "reason": "x", "latency_ms": 30.0},
            {"status": "error"}]
    s = chk.summarise_ingest(recs)
    assert s["status"] == {"stored": 1, "refused": 1, "error": 1} and s["refusal_reasons"] == {"x": 1}
    records, queries, qrels = _records()
    rows = chk.per_query(records, queries, qrels, maps())
    result = {"slm_bin": "/x/slm", "port": 1, "manifest": "abc", "skipped": None,
              "install": {"final_state": "ready", "total_s": 12.0, "restart_required": True, "restart_done": True,
                          "timeline": [{"t_s": 1.0, "env_state": "ready", "step": "", "progress": 1.0,
                                        "restart_required": True, "error": None}]},
              "ingest": {"images": {"summary": s, "records": []}, "documents": {"summary": s, "records": []}},
              "recall": {"scores": {"test": chk.score_media(rows)}, "outcomes": chk.media_outcomes(records, queries),
                         "fields": {}, "keys": {}, "p50_ms": 5.0, "p95_ms": 9.0},
              "thumbnails": {"items": [{"doc_id": "img:a", "status": 200, "content_type": "image/webp", "bytes": 5}],
                             "gps": {"no_gps": True}},
              "provenance": [], "erasure": {"image": {"after": {"clean": True}}},
              "rss": {"processes": [{"label": "d", "pid": 1, "peak_mb": 100.0}], "peak_total_mb": 100.0},
              "log_errors": {"error_lines": 0, "distinct": 0, "first_distinct": []}}
    md = chk.render_md(result)
    for heading in ("Install timeline", "Ingest summary", "Recall@5 per stratum", "Answer check", "Thumbnails",
                    "Page provenance", "Erasure", "Peak memory", "Daemon log errors"):
        assert heading in md
    assert "| image | 2 |" in md
    json.dumps(result)
    assert "capability" not in md.lower()
    skipped = chk.render_md({"slm_bin": "x", "skipped": "install ended in failed"})
    assert "skipped" in skipped
