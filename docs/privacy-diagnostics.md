# Local privacy diagnostics

SuperLocalMemory keeps a bounded 31-day operational summary in the active SLM
data directory. It stores only daily counters, fixed client-family buckets,
latency buckets, coarse error classes, and aggregate cross-client transitions.
Memory content, queries, fact identifiers, filesystem locations, exception
messages, and identity values are not stored in this diagnostics database.

There is **no network reporting endpoint**. The aggregate file is never
exported automatically. To create a local support artifact, the operator must
explicitly run:

```bash
slm diagnostics export ./slm-diagnostics.json
```

The output is deterministic for unchanged counters and is written with owner
read/write permissions. Inspect it before choosing whether to share it. Running
the command does not enable recurring reporting or contact a remote service.
