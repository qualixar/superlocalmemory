"""Pure parts of the daemon end-to-end script, and its client against a fake server."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import e2e_daemon as e2e


def test_fact_map_and_ranked_docs_with_content_fallback():
    fmap = e2e.build_fact_map([("t:a", ["f1"]), ("t:b", ["f2", "f3"]), ("t:c", [])])
    results = [{"fact_id": "f3"}, {"fact_id": "zz", "content": "hello"}, {"fact_id": "f1"}, {"fact_id": "f2"}]
    ranked = e2e.ranked_doc_ids(results, fmap, {"hello": "t:c"})
    assert ranked == ["t:b", "t:c", "t:a"]


def test_refusal_signals_and_outcomes():
    sig = e2e.refusal_signals({"abstained": False, "no_confident_match": False, "answerability": "unsupported"})
    assert sig["unsupported"] and e2e.refused(sig)
    assert not e2e.refused(e2e.refusal_signals({"answerability": "unjudged"}))
    queries = [{"id": "loc:1", "answerable": False, "stratum": "unanswerable"},
               {"id": "loc:2", "answerable": True, "stratum": "temporal"},
               {"id": "med:1", "answerable": False, "stratum": "unanswerable"}]
    yes = {"abstained": True, "no_confident_match": False, "unsupported": False}
    no = {"abstained": False, "no_confident_match": False, "unsupported": False}
    out = e2e.refusal_outcomes({"loc:1": {"signals": yes}, "loc:2": {"signals": yes}}, queries)
    assert out["unanswerable"] == {"n": 1, "refused": 1, "share": 1.0,
                                   "by_signal": {"abstained": 1, "no_confident_match": 0, "unsupported": 0}}
    assert out["answerable"]["wrongly_refused"] == 1
    assert e2e.refusal_outcomes({"loc:1": {"signals": no}}, queries)["unanswerable"]["share"] == 0.0


def test_score_split_matches_hand_computation():
    queries = [{"id": "loc:s:1", "stratum": "temporal", "answerable": True},
               {"id": "loc:s:2", "stratum": "temporal", "answerable": True},
               {"id": "loc:s:3", "stratum": "unanswerable", "answerable": False},
               {"id": "med:x", "stratum": "image", "answerable": True}]
    qrels = {"loc:s:1": {"a": 1}, "loc:s:2": {"b": 1}, "loc:s:3": {}, "med:x": {"img": 1}}
    records = {"loc:s:1": {"ranked": ["a", "z"]}, "loc:s:2": {"ranked": ["z", "y", "b"]}}
    scores = e2e.score_split(records, queries, qrels)
    assert scores["temporal"]["recall@5"]["mean"] == 1.0
    assert abs(scores["temporal"]["mrr@10"]["mean"] - (1 + 1 / 3) / 2) < 1e-9
    assert scores["text_overall"]["recall@5"]["n"] == 2
    assert scores["entity"]["recall@5"]["n"] == 0


def test_log_error_collection_groups_repeats():
    log = ("2026-10-09 17:00:00,001 starting\n"
           "2026-10-09 17:00:01,002 ERROR boom 12\n"
           "2026-10-09 17:00:02,003 ERROR boom 13\n"
           "Traceback (most recent call last):\n"
           "ValueError: bad\n")
    out = e2e.collect_log_errors(log)
    assert out["error_lines"] == 4 and out["distinct"] == 3
    assert out["first_distinct"][0].endswith("boom 12")


def test_scrub_cmdline_drops_long_arguments():
    assert e2e.scrub_cmdline(["/usr/bin/python3", "-m", "x" * 90, "serve"]) == "python3 -m serve"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not getattr(self.server, "throttled", True):
            self.server.throttled = True
            raw = b"{}"
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        n = int(self.headers["Content-Length"])
        content = json.loads(self.rfile.read(n))["content"]
        self.server.stored.append(content)
        self._send({"fact_ids": [f"f{len(self.server.stored)}"]})

    def do_GET(self):
        self._send({"results": [{"fact_id": "f1", "content": self.server.stored[0]}],
                    "abstained": False, "answerability": "unjudged", "channel_status": {"bm25": "ok"}})


def test_ingest_and_recall_against_a_fake_daemon():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.stored = []
    srv.throttled = False  # first POST answers 429 once
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = e2e.Client(srv.server_port, {"X-SLM-Daemon-Capability": "secret"})
        assert "secret" not in repr(client)
        items = [{"doc_id": "t:1", "text": "alpha"}, {"doc_id": "t:2", "text": "beta"}]
        ing = e2e.ingest(client, items)
        assert ing["ok"] == 2 and ing["n_errors"] == 0 and ing["throttle"]["n_429"] == 1 and ing["latency"]["p50_ms"] > 0
        recs = e2e.recall_all(client, [{"id": "loc:q", "text": "what"}], e2e.build_fact_map(ing["pairs"]), {})
        assert recs["loc:q"]["ranked"] == ["t:1"]
        assert e2e.tally_fields(list(recs.values()))["channel_status.bm25"] == {"ok": 1}
    finally:
        srv.shutdown()


def test_process_sampler_sees_own_process():
    import time

    s = e2e.ProcessSampler("/nonexistent", interval=0.02, scan_every=1000)
    s.root_pid = os.getpid()
    s.start()
    time.sleep(0.15)
    procs = s.stop()
    assert procs and procs[0]["peak_mb"] > 0 and s.peak_total_mb > 0
