"""End-to-end test of a real SLM daemon on the comparison dataset's LoCoMo text.

Starts the daemon from the given install in a fresh data and home directory,
ingests every LoCoMo turn through POST /remember, runs every text query through
GET /recall (default answer check), and writes e2e.json and E2E.md.

The daemon capability token is read from daemon.json only to build request
headers; it is never printed, logged or written to the results.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

import report
from build_dataset import bench_home

TEXT_QID_PREFIX = "loc:"
TEXT_GROUPS = ("text_single_hop", "text_multi_hop", "entity", "temporal", "text_overall")
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL|FATAL)\b|Traceback \(most recent call last\)|\b\w+(Error|Exception):")
FIELD_PREFIXES = ("answer_check", "answerability")
STATUS_FIELDS = ("reranker_status", "calibration_status", "abstention_reason", "query_type")


# ---------------------------------------------------------------- pure parts

def percentile(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def text_corpus(corpus: list[dict]) -> list[dict]:
    """LoCoMo text memories only, in a stable order."""
    return sorted((d for d in corpus if d["kind"] == "text"), key=lambda d: d["doc_id"])


def text_queries(queries: list[dict]) -> list[dict]:
    return [q for q in queries if q["id"].startswith(TEXT_QID_PREFIX)]


def build_fact_map(pairs: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
    """fact id -> doc ids, from (doc_id, fact_ids returned by /remember) pairs."""
    out: dict[str, list[str]] = {}
    for doc_id, fact_ids in pairs:
        for fact_id in fact_ids:
            out.setdefault(fact_id, []).append(doc_id)
    return out


def ranked_doc_ids(results: list[dict], fact_map: dict[str, list[str]],
                   content_map: dict[str, str]) -> list[str]:
    """Doc ids of a recall response in rank order, deduplicated.

    A result maps by fact id; when the daemon returned no usable id the exact
    stored content is the fallback.
    """
    out: list[str] = []
    for item in results:
        docs = fact_map.get(item.get("fact_id", ""), [])
        if not docs and item.get("content") in content_map:
            docs = [content_map[item["content"]]]
        out += [d for d in docs if d not in out]
    return out


def refusal_signals(resp: dict) -> dict[str, bool]:
    """What the daemon said about whether it has the answer."""
    return {"abstained": bool(resp.get("abstained")),
            "no_confident_match": bool(resp.get("no_confident_match")),
            "unsupported": resp.get("answerability") == "unsupported"}


def refused(signals: dict[str, bool]) -> bool:
    return any(signals.values())


def observed_fields(resp: dict) -> dict:
    """Answer-check, answerability and status fields present in one recall response."""
    out = {k: v for k, v in resp.items() if k.startswith(FIELD_PREFIXES)}
    out.update({k: resp[k] for k in STATUS_FIELDS if k in resp})
    out["incomplete_channels"] = resp.get("incomplete_channels", [])
    out["channel_status"] = resp.get("channel_status", {})
    return out


def tally_fields(records: list[dict]) -> dict:
    """Counters of every observed answer-check / status value, per field."""
    tallies: dict[str, Counter] = {}
    for rec in records:
        for key, value in rec.get("fields", {}).items():
            if key == "channel_status":
                for channel, state in value.items():
                    tallies.setdefault(f"channel_status.{channel}", Counter())[str(state)] += 1
            elif key == "incomplete_channels":
                tallies.setdefault(key, Counter())[",".join(sorted(value))] += 1
            else:
                tallies.setdefault(key, Counter())[str(value)] += 1
    return {k: dict(v.most_common()) for k, v in sorted(tallies.items())}


def score_split(records: dict[str, dict], queries: list[dict], qrels: dict) -> dict:
    """recall@5 / MRR@10 with 95% CIs per text group on one split (answerable queries)."""
    q_text = text_queries(queries)
    qrels_text = {q["id"]: qrels[q["id"]] for q in q_text if q["answerable"]}
    run = {}
    for qid in qrels_text:
        ranked = records.get(qid, {}).get("ranked", [])
        run[qid] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
    res = report.evaluate(run, qrels_text, q_text)
    return {g: {m: {k: v for k, v in res[g][m].items() if k != "values"}
                for m in ("recall@5", "mrr@10")} for g in TEXT_GROUPS}


def refusal_outcomes(records: dict[str, dict], queries: list[dict]) -> dict:
    """Unanswerable: share the daemon refused. Answerable: share wrongly refused."""
    out = {}
    for label, want_answerable in (("unanswerable", False), ("answerable", True)):
        recs = [records[q["id"]] for q in text_queries(queries)
                if q["answerable"] is want_answerable and q["id"] in records
                and "signals" in records[q["id"]]]
        n = len(recs)
        by = {k: sum(1 for r in recs if r["signals"][k]) for k in ("abstained", "no_confident_match", "unsupported")}
        any_n = sum(1 for r in recs if refused(r["signals"]))
        key = "refused" if not want_answerable else "wrongly_refused"
        out[label] = {"n": n, key: any_n, "share": any_n / n if n else None, "by_signal": by}
    return out


def normalise_line(line: str) -> str:
    """Log line without timestamp and digits, to group repeats."""
    return re.sub(r"\d+", "#", re.sub(r"^\d{4}-\d\d-\d\d[ T][\d:,.]+\s*", "", line.strip()))[:200]


def collect_log_errors(text: str, first: int = 20) -> dict:
    """Count ERROR / Traceback lines in a log and list the first distinct ones."""
    lines = [ln for ln in text.splitlines() if ERROR_RE.search(ln)]
    seen: dict[str, str] = {}
    for ln in lines:
        seen.setdefault(normalise_line(ln), ln.strip()[:300])
    return {"error_lines": len(lines), "distinct": len(seen), "first_distinct": list(seen.values())[:first]}


def scrub_cmdline(cmd: list[str]) -> str:
    """Short process label: basenames, long arguments dropped."""
    parts = [os.path.basename(c) if c.startswith("/") else c for c in cmd[:5] if len(c) <= 60]
    return " ".join(parts)[:100] or "?"


# ------------------------------------------------------------ process sampler

class ProcessSampler:
    """Peak RSS per process: the daemon tree every interval, plus a periodic scan for
    processes that carry the same SLM_DATA_DIR (e.g. re-parented workers)."""

    def __init__(self, data_dir: str, interval: float = 0.1, scan_every: int = 20):
        self.data_dir, self.interval, self.scan_every = data_dir, interval, scan_every
        self.root_pid: int | None = None
        self.peaks: dict[int, dict] = {}
        self.peak_total_mb = 0.0
        self._extra: set[int] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _scan_env(self) -> set[int]:
        found = set()
        for p in psutil.process_iter(["pid"]):
            try:
                if p.environ().get("SLM_DATA_DIR") == self.data_dir:
                    found.add(p.pid)
            except (psutil.Error, OSError):
                continue
        return found

    def _tree(self) -> set[int]:
        if self.root_pid is None:
            return set()
        try:
            root = psutil.Process(self.root_pid)
            return {root.pid, *(c.pid for c in root.children(recursive=True))}
        except psutil.Error:
            return set()

    def _record(self, pid: int) -> float:
        try:
            p = psutil.Process(pid)
            mb = p.memory_info().rss / 1e6
            entry = self.peaks.setdefault(pid, {"label": scrub_cmdline(p.cmdline()), "peak_mb": 0.0})
            entry["peak_mb"] = max(entry["peak_mb"], mb)
            return mb
        except psutil.Error:
            return 0.0

    def _loop(self) -> None:
        tick = 0
        while not self._stop.is_set():
            if tick % self.scan_every == 0:
                self._extra = self._scan_env()
            total = sum(self._record(pid) for pid in self._tree() | self._extra)
            self.peak_total_mb = max(self.peak_total_mb, total)
            tick += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        self._thread.join()
        return sorted(({"pid": pid, **v} for pid, v in self.peaks.items()),
                      key=lambda e: -e["peak_mb"])


# --------------------------------------------------------------- daemon access

class Client:
    """Small keep-alive HTTP client; auth headers are held privately, never printed."""

    def __init__(self, port: int, headers: dict[str, str] | None = None, timeout: float = 120.0):
        self.port, self._headers, self.timeout = port, dict(headers or {}), timeout
        self._conn: http.client.HTTPConnection | None = None
        self.retry_after: float | None = None  # Retry-After of the last response, seconds

    def __repr__(self) -> str:
        return f"Client(port={self.port})"

    def _send(self, method: str, path: str, payload: bytes | None, headers: dict):
        if self._conn is None:
            self._conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        self._conn.request(method, path, payload, headers)
        resp = self._conn.getresponse()
        raw = resp.read()
        self.retry_after = _seconds(resp.getheader("Retry-After"))
        return resp, raw

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict, float]:
        """(status, parsed JSON or {}, latency ms). Raises OSError on connection failure."""
        payload = json.dumps(body).encode() if body is not None else None
        headers = {**self._headers, **({"Content-Type": "application/json"} if payload else {})}
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                resp, raw = self._send(method, path, payload, headers)
                break
            except (OSError, http.client.HTTPException):
                self._conn = None
                if attempt == 2:  # a kept-alive connection the server closed: retry once, fresh
                    raise
        ms = (time.perf_counter() - t0) * 1000
        try:
            data = json.loads(raw) if raw else {}
        except ValueError:
            data = {}
        return resp.status, data if isinstance(data, dict) else {}, ms


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def free_port(start: int = 8821) -> int:
    for port in range(start, start + 100):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port found")


def daemon_env(data: Path, home: Path, port: int, offline: bool) -> dict[str, str]:
    env = {**os.environ, "SLM_DATA_DIR": str(data), "SLM_DAEMON_PORT": str(port), "HOME": str(home)}
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def poll_health(port: int, timeout: float, want_ready: bool, proc: subprocess.Popen | None = None):
    """Poll /health until 200 (or ready:true). Returns (seconds, health dict) or (None, last)."""
    t0, last = time.perf_counter(), {}
    client = Client(port, timeout=5)
    while time.perf_counter() - t0 < timeout:
        try:
            status, body, _ = client.call("GET", "/health")
            if status == 200:
                last = body
                if not want_ready or body.get("ready") is True:
                    return time.perf_counter() - t0, body
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    return None, last


def read_auth(data: Path) -> dict[str, str]:
    d = json.loads((data / "daemon.json").read_text())
    return {"X-SLM-Daemon-Capability": d["capability"], "X-SLM-Target-Instance": d["instance_id"]}


# ------------------------------------------------------------------ run stages

MAX_BACKOFF_S = 65.0


def remember_one(client: Client, item: dict, throttle: dict) -> tuple[int, dict, float]:
    """POST /remember, honouring the daemon's write rate limit (429 + Retry-After)."""
    for _ in range(30):
        status, body, ms = client.call("POST", "/remember", {"content": item["text"]})
        if status != 429:
            return status, body, ms
        wait = min(60.0 if client.retry_after is None else client.retry_after, MAX_BACKOFF_S)
        throttle["n_429"] += 1
        throttle["backoff_s"] += wait
        time.sleep(wait)
    return status, body, ms


def ingest(client: Client, items: list[dict], log_every: int = 250) -> dict:
    """POST every corpus item once; returns latencies, id pairs and error info.

    Latency covers accepted calls only; time spent waiting out a 429 is reported
    separately as throttling.
    """
    pairs, latencies, errors = [], [], []
    throttle = {"n_429": 0, "backoff_s": 0.0}
    for i, item in enumerate(items, 1):
        try:
            status, body, ms = remember_one(client, item, throttle)
        except (OSError, http.client.HTTPException) as exc:
            errors.append({"doc_id": item["doc_id"], "error": type(exc).__name__})
            continue
        if status != 200:
            errors.append({"doc_id": item["doc_id"], "status": status})
        else:
            latencies.append(ms)
            pairs.append((item["doc_id"], list(body.get("fact_ids") or [])))
        if i % log_every == 0:
            print(f"ingested {i}/{len(items)}", file=sys.stderr, flush=True)
    stats_ = {"p50_ms": percentile(latencies, 50), "p95_ms": percentile(latencies, 95),
              "max_ms": max(latencies, default=None)}
    return {"ok": len(pairs), "attempted": len(items), "errors": errors[:20], "n_errors": len(errors),
            "latency": stats_, "pairs": pairs, "throttle": throttle,
            "without_fact_id": sum(1 for _, f in pairs if not f)}


def wait_idle(port: int, cap_s: float) -> dict:
    """Wait for the projection queue to drain (depth 0 on three polls in a row)."""
    t0, calm, last = time.perf_counter(), 0, {}
    client = Client(port, timeout=10)
    while time.perf_counter() - t0 < cap_s:
        try:
            _, body, _ = client.call("GET", "/health")
            last = body.get("projection") or {}
            calm = calm + 1 if last.get("depth", 1) == 0 else 0
        except (OSError, http.client.HTTPException):
            calm = 0
        if calm >= 3:
            return {"idle": True, "wait_s": time.perf_counter() - t0, "projection": last}
        time.sleep(2)
    return {"idle": False, "wait_s": time.perf_counter() - t0, "projection": last}


def recall_all(client: Client, queries: list[dict], fact_map: dict, content_map: dict) -> dict:
    """GET /recall?q=..&limit=10 for each query; one record per query id."""
    records = {}
    for q in queries:
        path = "/recall?q=" + urllib.parse.quote(q["text"]) + "&limit=10"
        try:
            status, body, ms = client.call("GET", path)
        except (OSError, http.client.HTTPException) as exc:
            records[q["id"]] = {"error": type(exc).__name__, "ranked": []}
            continue
        if status != 200:
            records[q["id"]] = {"error": f"http {status}", "ranked": [], "latency_ms": ms}
            continue
        records[q["id"]] = {
            "latency_ms": ms, "n_results": len(body.get("results", [])),
            "ranked": ranked_doc_ids(body.get("results", []), fact_map, content_map),
            "signals": refusal_signals(body), "fields": observed_fields(body)}
    return records


def stop_daemon(env: dict, slm_bin: str, port: int, pid: int | None) -> dict:
    subprocess.run([slm_bin, "serve", "stop"], env=env, capture_output=True, timeout=120)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30:
        down = not (pid and psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
        with socket.socket() as sock:
            closed = sock.connect_ex(("127.0.0.1", port)) != 0
        if down and closed:
            return {"stopped": True, "wait_s": time.perf_counter() - t0}
        time.sleep(0.5)
    return {"stopped": False, "wait_s": time.perf_counter() - t0}


def start_daemon(slm_bin: str, env: dict, port: int, ready_cap: float) -> dict:
    """Start the daemon; time to /health 200 and to ready:true (from the start command)."""
    proc = subprocess.Popen([slm_bin, "serve", "start"], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    t_health, health = poll_health(port, 300, want_ready=False)
    t_ready, last = poll_health(port, ready_cap, want_ready=True)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    return {"health_200_s": t_health,
            "ready_s": None if t_ready is None or t_health is None else t_health + t_ready,
            "ready_within_cap": t_ready is not None, "ready_cap_s": ready_cap,
            "readiness": last.get("readiness", {}), "pid": last.get("pid") or health.get("pid")}


# ------------------------------------------------------------------- reporting

def _fmt(x, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def _group_table(scores: dict) -> list[str]:
    lines = ["| group | recall@5 [95% CI] | MRR@10 [95% CI] | n |", "|---|---|---|---|"]
    for g, cell in scores.items():
        r, m = cell["recall@5"], cell["mrr@10"]
        flag = " *" if r["n"] < report.MIN_STRATUM_N else ""
        lines.append(f"| {g} | {_fmt(r['mean'])} [{_fmt(r['lo'])}, {_fmt(r['hi'])}] | "
                     f"{_fmt(m['mean'])} [{_fmt(m['lo'])}, {_fmt(m['hi'])}] | {r['n']}{flag} |")
    return lines


def _lat(d: dict) -> str:
    return f"p50 {_fmt(d['p50_ms'], 1)} ms, p95 {_fmt(d['p95_ms'], 1)} ms"


def render_md(r: dict) -> str:
    s, ing, rec = r["startup"], r["ingest"], r["recall"]
    out = ["# End-to-end daemon test", "",
           f"Install: `{r['slm_bin']}`  |  offline: {r['offline']}  |  port {r['port']}  |  "
           f"dataset `{r['manifest'][:16]}`  |  all labels PROVISIONAL", "",
           "## Startup", "",
           f"- time to /health 200: {_fmt(s['health_200_s'], 1)} s",
           f"- time to ready:true: " + (f"{_fmt(s['ready_s'], 1)} s" if s["ready_within_cap"]
                                       else f"never within {s['ready_cap_s']:.0f} s (readiness: `{json.dumps(s['readiness'])}`)"),
           "", "## Ingest (POST /remember, concurrency 1)", "",
           f"- stored {ing['ok']} of {ing['attempted']} turns, {ing['n_errors']} errors, "
           f"{ing['without_fact_id']} without a fact id",
           f"- remember latency (accepted calls): {_lat(ing['latency'])}, max {_fmt(ing['latency']['max_ms'], 1)} ms",
           f"- write rate limit: {ing['throttle']['n_429']} HTTP 429 responses, "
           f"{_fmt(ing['throttle']['backoff_s'], 0)} s spent waiting out Retry-After",
           f"- idle after ingest: {r['idle']['idle']} (waited {_fmt(r['idle']['wait_s'], 1)} s, "
           f"projection `{json.dumps(r['idle']['projection'])}`)",
           "", "## Recall (GET /recall?limit=10, default answer check)", "",
           f"- {rec['n_ok']} of {rec['n_queries']} queries answered, latency: {_lat(rec['latency'])}", "",
           "### Test split (answerable text queries; `*` = n < 30)", ""] + _group_table(r["scores"]["test"])
    out += ["", "### Dev split", ""] + _group_table(r["scores"]["dev"])
    out += ["", "## Unanswerable outcomes", ""]
    for split in ("test", "dev"):
        o = r["outcomes"][split]
        u, a = o["unanswerable"], o["answerable"]
        out.append(f"- {split}: daemon said it does not have it on {u['refused']}/{u['n']} unanswerable "
                   f"({_fmt(u['share'])}; by signal {u['by_signal']}); wrongly refused "
                   f"{a['wrongly_refused']}/{a['n']} answerable ({_fmt(a['share'])})")
    out += ["", "## Answer-check and channel fields observed", ""]
    for key, counts in r["fields"].items():
        out.append(f"- `{key}`: {json.dumps(counts)}")
    out += ["", "## Peak memory per process", "", "| process | pid | peak RSS MB |", "|---|---|---|"]
    out += [f"| {p['label']} | {p['pid']} | {p['peak_mb']:.0f} |" for p in r["rss"]["processes"]]
    out.append(f"\nPeak of the summed daemon tree: {r['rss']['peak_total_mb']:.0f} MB.")
    le = r["log_errors"]
    out += ["", "## Daemon log errors", "", f"{le['error_lines']} ERROR/Traceback lines, {le['distinct']} distinct."]
    out += [f"- `{ln}`" for ln in le["first_distinct"]]
    out += ["", "## Stop", "", f"- stopped cleanly: {r['stop']['stopped']} ({_fmt(r['stop']['wait_s'], 1)} s)",
            f"- media: {r['media']}", "", "Cloud caveat: " + r["caveat"], ""]
    return "\n".join(out)


# ------------------------------------------------------------------------ main

def load_dataset(ds: Path) -> tuple[list[dict], dict, dict]:
    jl = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    corpus = jl(ds / "corpus.jsonl")
    queries = {s: jl(ds / "queries" / f"{s}.jsonl") for s in ("dev", "test")}
    return corpus, queries, {s: report.load_qrels(ds, s) for s in ("dev", "test")}


def channel_caveat(fields: dict) -> str:
    chans = {k.split(".", 1)[1]: v for k, v in fields.items() if k.startswith("channel_status.")}
    return ("channel status seen (counts per state): " + json.dumps(chans) +
            ". Without an embedding or reranker model the daemon answers by keyword (bm25) only.")


def execute(args, work: Path) -> dict:
    ds = args.home / "dataset"
    corpus, queries, qrels = load_dataset(ds)
    items = text_corpus(corpus)
    data, home = work / "data", work / "home"
    data.mkdir(), home.mkdir()
    port = args.port or free_port()
    env = daemon_env(data, home, port, args.offline)
    sampler = ProcessSampler(str(data))
    sampler.start()
    startup = {}
    try:
        startup = start_daemon(args.slm_bin, env, port, args.ready_timeout)
        sampler.root_pid = startup.get("pid")
        client = Client(port, read_auth(data))
        ing = ingest(client, items)
        idle = wait_idle(port, args.idle_timeout)
        fact_map = build_fact_map(ing.pop("pairs"))
        content_map = {d["text"]: d["doc_id"] for d in items}
        all_q = text_queries(queries["dev"] + queries["test"])
        records = recall_all(client, all_q, fact_map, content_map)
    finally:
        stop = stop_daemon(env, args.slm_bin, port, startup.get("pid"))
        procs = sampler.stop()
    log = data / "logs" / "daemon.log"
    return assemble(args, port, startup, ing, idle, records, queries, qrels, procs, sampler, stop,
                    collect_log_errors(log.read_text(errors="replace") if log.exists() else ""), ds)


def assemble(args, port, startup, ing, idle, records, queries, qrels, procs, sampler, stop, log_errors, ds) -> dict:
    ok = [r for r in records.values() if "latency_ms" in r]
    fields = tally_fields(list(records.values()))
    return {
        "slm_bin": args.slm_bin, "offline": args.offline, "port": port,
        "manifest": json.loads((ds / "manifest.lock").read_text())["sha256"],
        "startup": startup, "ingest": ing, "idle": idle,
        "recall": {"n_queries": len(records), "n_ok": len(ok),
                   "n_errors": len(records) - len(ok),
                   "latency": {"p50_ms": percentile([r["latency_ms"] for r in ok], 50),
                               "p95_ms": percentile([r["latency_ms"] for r in ok], 95)}},
        "scores": {s: score_split(records, queries[s], qrels[s]) for s in ("test", "dev")},
        "outcomes": {s: refusal_outcomes(records, queries[s]) for s in ("test", "dev")},
        "fields": fields,
        "rss": {"processes": procs, "peak_total_mb": sampler.peak_total_mb},
        "log_errors": log_errors, "stop": stop,
        "media": "not tested, no media branch yet" if not args.media else "requested; media ingestion is not implemented in this version",
        "caveat": channel_caveat(fields),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slm-bin", required=True, help="path to the `slm` executable of the install under test")
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=0, help="daemon port (default: first free from 8821)")
    ap.add_argument("--offline", action="store_true", help="set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    ap.add_argument("--idle-timeout", type=float, default=300.0)
    ap.add_argument("--media", action="store_true", help="reserved: media is not tested yet")
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="slm-e2e-"))
    try:
        result = execute(args, work)
    finally:
        if not args.keep_data:
            shutil.rmtree(work, ignore_errors=True)
    (args.out / "e2e.json").write_text(json.dumps(result, indent=1, default=str))
    (args.out / "E2E.md").write_text(render_md(result))
    print(f"wrote {args.out / 'E2E.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
