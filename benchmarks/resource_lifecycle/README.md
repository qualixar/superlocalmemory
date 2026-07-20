# Resource-lifecycle benchmark

This harness measures repeated canonical ingestion and fixed-corpus recall on
the shipped Mode A engine in an isolated temporary SQLite namespace. It uses a
deterministic 768-dimensional local embedding boundary, so the result measures
SLM lifecycle behavior without downloading a model or starting an embedding
worker.

Run from the repository root:

```bash
.venv/bin/python benchmarks/resource_lifecycle/run_benchmark.py
```

The command writes `RESULTS.json` and `RESULTS.md` beside the harness. The raw
result includes the source commit, package and Python versions, machine details,
warm-up and sample counts, unique-ingest/idempotent-retry/recall p50 and p95,
every RSS sample, tail slope, and child-process ownership after close. Unique
ingests intentionally grow the corpus and indexes; fixed-state idempotent retry
and fixed-corpus recall are the leak-sensitive plateau phases.

The conclusion is deliberately narrow. A finite local run can show that RSS
plateaued in its fixed-corpus recall tail; it cannot prove unlimited-corpus
behavior. There is no universal one-second latency gate. Mode B Ollama and Mode
C cloud-provider end-to-end latency require separate measurements on the actual
model host/provider/network and must not be inferred from this harness.
