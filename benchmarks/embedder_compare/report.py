"""Metrics, paired bootstrap CIs, randomisation tests and the one-model rule.

Writes RESULTS.md and RESULTS.json. Refuses to run when the test queries or
qrels no longer match manifest.lock. All labels are PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import stats
from build_dataset import bench_home, manifest_hash

TEXT = ("text_single_hop", "text_multi_hop", "entity", "temporal")
GROUPS = {**{s: (s,) for s in (*TEXT, "image", "pdf_page")},
          "text_overall": TEXT, "media_overall": ("image", "pdf_page")}
METRICS = ("recall@5", "mrr@10", "ndcg@10")
MIN_STRATUM_N = 30
SYSTEM_LABELS = {"bm25": "bm25 (smoke, never a verdict)", "s1": "S1 nomic text, media via OCR",
                 "s2": "S2 EmbeddingGemma 2 one-model", "s3": "S3 nomic text + EmbeddingGemma 2 media (RRF)"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_queries(ds: Path) -> dict[str, list[dict]]:
    return {s: _jsonl(ds / "queries" / f"{s}.jsonl") for s in ("dev", "test")}


def load_qrels(ds: Path, split: str) -> dict:
    return json.loads((ds / "qrels" / f"{split}.json").read_text())


def check_manifest(ds: Path) -> str | None:
    """Return an error message when test data changed since build, else None."""
    want = json.loads((ds / "manifest.lock").read_text())["sha256"]
    got = manifest_hash(ds / "queries" / "test.jsonl", ds / "qrels" / "test.json")
    if want != got:
        return f"manifest.lock mismatch: test queries or qrels changed (expected {want[:12]}, got {got[:12]})"
    return None


def tune_tau(dev_queries: list[dict], scores: dict[str, float]) -> float:
    """Abstention threshold from dev queries only."""
    return stats.tune_threshold([scores[q["id"]] for q in dev_queries],
                                [not q["answerable"] for q in dev_queries])


def abstention(queries: dict, scores: dict[str, float]) -> dict:
    tau = tune_tau(queries["dev"], scores)
    test = queries["test"]
    rep = stats.abstention_report([scores[q["id"]] for q in test],
                                  [not q["answerable"] for q in test], tau)
    return {**rep, "tau": None if tau == float("-inf") else tau}


def group_values(per_q: dict[str, float], queries: list[dict], strata: tuple) -> list[tuple[str, float]]:
    ids = sorted(q["id"] for q in queries if q["stratum"] in strata and q["answerable"] and q["id"] in per_q)
    return [(i, per_q[i]) for i in ids]


def evaluate(run: dict, qrels: dict, queries: list[dict]) -> dict:
    """{group: {metric: {'n', 'mean', 'lo', 'hi', 'values': {qid: v}}}} on one split."""
    per = {m: stats.per_query(qrels, run, m) for m in METRICS}
    out = {}
    for group, strata in GROUPS.items():
        out[group] = {}
        for m in METRICS:
            vals = group_values(per[m], queries, strata)
            mean, lo, hi = stats.bootstrap_ci([v for _, v in vals])
            out[group][m] = {"n": len(vals), "mean": mean, "lo": lo, "hi": hi,
                             "values": dict(vals)}
    return out


def fisher_p(run_a: dict, run_b: dict, qrels: dict, qids: list[str]) -> float | None:
    """ranx Fisher randomisation test p-value on recall@5 over the given queries."""
    if len(qids) < 2:
        return None
    from ranx import Qrels, Run, compare

    sub = {q: qrels[q] for q in qids}
    rep = compare(Qrels(sub), [Run({q: run_a[q] for q in qids}, name="a"),
                               Run({q: run_b[q] for q in qids}, name="b")],
                  ["recall@5"], stat_test="fisher", n_permutations=1000,
                  random_seed=stats.SEED, max_p=0.05)
    return rep.to_dict()["a"]["comparisons"]["b"]["recall@5"]


def paired(res_a: dict, res_b: dict, runs: tuple, qrels: dict) -> dict:
    out = {}
    for group in ("text_overall", "image", "pdf_page"):
        va, vb = res_a[group]["recall@5"]["values"], res_b[group]["recall@5"]["values"]
        ids = sorted(set(va) & set(vb))
        d, lo, hi = stats.paired_delta_ci([va[i] for i in ids], [vb[i] for i in ids])
        out[group] = {"n": len(ids), "delta": d, "lo": lo, "hi": hi,
                      "fisher_p": fisher_p(runs[0], runs[1], qrels, ids)}
    return out


def one_model(results: dict) -> dict:
    if "s1" not in results or "s2" not in results:
        return {"verdict": "NOT EVALUATED (S1 and S2 are both required)"}
    a = results["s1"]["text_overall"]["recall@5"]["values"]
    b = results["s2"]["text_overall"]["recall@5"]["values"]
    ids = sorted(set(a) & set(b))
    return stats.one_model_rule([a[i] for i in ids], [b[i] for i in ids])


def _fmt(cell: dict) -> str:
    if cell["n"] == 0:
        return "n/a"
    flag = " *" if cell["n"] < MIN_STRATUM_N else ""
    return f"{cell['mean']:.3f} [{cell['lo']:.3f}, {cell['hi']:.3f}] n={cell['n']}{flag}"


def _table(results: dict, metric: str) -> list[str]:
    names = list(results)
    lines = ["| group | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for group in GROUPS:
        lines.append(f"| {group} | " + " | ".join(_fmt(results[s][group][metric]) for s in names) + " |")
    return lines


def _flatten_timings(timings: dict) -> list[tuple[str, dict]]:
    if "nomic_process" in timings:
        return [("nomic process", timings["nomic_process"]), ("eg2 media process", timings["eg2_media_process"])]
    return [("", timings)]


def _ms(x) -> str:
    return "-" if x is None else f"{x:.1f}"


def _latency_lines(all_timings: dict) -> list[str]:
    lines = ["| system | load s | peak RSS MB | query p50/p95 ms | doc p50/p95 ms | image p50/p95 ms | items/min |",
             "|---|---|---|---|---|---|---|"]
    for sys_name, t in all_timings.items():
        for label, s in _flatten_timings(t):
            row = [f"{sys_name} {label}".strip(), f"{s['load_s']:.1f}", f"{s.get('peak_rss_mb', 0):.0f}"]
            for k in ("query", "doc", "image"):
                row.append(f"{_ms(s[k].get('p50_ms'))}/{_ms(s[k].get('p95_ms'))}")
            row.append(_ms(s.get("items_per_min")))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def render(ctx: dict) -> str:
    q = ctx["queries"]
    counts: dict[str, list[int]] = {}
    for split in ("dev", "test"):
        for item in q[split]:
            counts.setdefault(item["stratum"], [0, 0])[0 if split == "dev" else 1] += 1
    out = ["# Embedder comparison results", "",
           "All labels are PROVISIONAL (maintainer review pending). Test split reported once; "
           "tuning (abstention threshold) used the dev split only. `*` marks groups with n < 30: "
           "no stratum-level claim. Intervals are 95% paired bootstrap (10,000 resamples, seed 1729).",
           f"Manifest: `{ctx['manifest'][:16]}`", "", "## Systems", ""]
    for name, st in ctx["status"].items():
        out.append(f"- {SYSTEM_LABELS.get(name, name)}: " + ("ran" if st["ran"] else f"not run ({st['reason']})"))
    out += ["", "Queries per stratum (dev / test): " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(counts.items())), ""]
    out += ["## Recall@5", ""] + _table(ctx["results"], "recall@5")
    out += ["", "## MRR@10", ""] + _table(ctx["results"], "mrr@10")
    out += ["", "## nDCG@10", ""] + _table(ctx["results"], "ndcg@10")
    out += ["", "## Unanswerable", "",
            "| system | tau (dev) | false-answer rate | abstention precision | abstention recall | n unanswerable (test) |",
            "|---|---|---|---|---|---|"]
    for name, a in ctx["abstention"].items():
        out.append(f"| {name} | {_pct(a['tau'])} | {_pct(a['false_answer_rate'])} | "
                   f"{_pct(a['abstention_precision'])} | {_pct(a['abstention_recall'])} | {a['n_unanswerable']} |")
    out += ["", "## Paired comparisons (recall@5, delta = second minus first)", ""]
    for pair, groups in ctx["paired"].items():
        for g, c in groups.items():
            out.append(f"- {pair} {g}: delta {c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] "
                       f"n={c['n']}, Fisher randomisation p={_pct(c['fisher_p'])}")
    if not ctx["paired"]:
        out.append("- no pair of systems ran")
    out += ["", "## Latency and memory", ""] + _latency_lines(ctx["timings"])
    out += ["", "## One-model rule", "", _one_model_text(ctx["one_model"]), ""]
    return "\n".join(out)


def _one_model_text(r: dict) -> str:
    if "delta" not in r:
        return f"Verdict: {r['verdict']}"
    return (f"Verdict: **{r['verdict']}** (text recall@5, S2 minus S1, n={r['n']}): delta {r['delta']:+.3f}, "
            f"95% CI [{r['delta_ci'][0]:+.3f}, {r['delta_ci'][1]:+.3f}], rule: CI lower bound >= -{r['margin']}. "
            f"Literal form (S2 lower CI {r['s2_ci_low']:.3f} >= S1 point {r['s1_point']:.3f} - {r['margin']}): "
            f"{'pass' if r['literal_form_pass'] else 'fail'}.")


def build_context(home: Path, runs_dir: Path) -> dict:
    ds = home / "dataset"
    queries = load_queries(ds)
    qrels = load_qrels(ds, "test")
    all_qrels = {**load_qrels(ds, "dev"), **qrels}
    status = json.loads((runs_dir / "status.json").read_text())
    ctx = {"queries": queries, "status": status, "results": {}, "abstention": {}, "timings": {},
           "runs": {}, "paired": {}, "manifest": json.loads((ds / "manifest.lock").read_text())["sha256"]}
    for name, st in status.items():
        if not st["ran"]:
            continue
        run = json.loads((runs_dir / f"{name}.json").read_text())
        ctx["runs"][name] = run
        ctx["results"][name] = evaluate(run, qrels, queries["test"])
        ctx["abstention"][name] = abstention(queries, json.loads((runs_dir / f"{name}.abstain.json").read_text()))
        ctx["timings"][name] = json.loads((runs_dir / f"{name}.timings.json").read_text())
    for a, b in (("s1", "s2"), ("s1", "s3"), ("s2", "s3")):
        if a in ctx["results"] and b in ctx["results"]:
            ctx["paired"][f"{a} -> {b}"] = paired(ctx["results"][a], ctx["results"][b],
                                                  (ctx["runs"][a], ctx["runs"][b]), all_qrels)
    ctx["one_model"] = one_model(ctx["results"])
    return ctx


def _strip_values(results: dict) -> dict:
    return {s: {g: {m: {k: v for k, v in c.items() if k != "values"} for m, c in ms.items()}
                for g, ms in gs.items()} for s, gs in results.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=bench_home())
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    err = check_manifest(args.home / "dataset")
    if err:
        print(f"refusing to report: {err}", file=sys.stderr)
        return 2
    ctx = build_context(args.home, args.runs or args.home / "runs")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "RESULTS.md").write_text(render(ctx))
    summary = {k: ctx[k] for k in ("status", "abstention", "paired", "one_model", "timings", "manifest")}
    summary["results"] = _strip_values(ctx["results"])
    (args.out / "RESULTS.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
