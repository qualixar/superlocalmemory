---
description: Apply SLM context-optimization (compress large output / cache repeats) and report savings.
argument-hint: [stats]
allowed-tools: slm_compress, slm_cache_set, slm_cache_get, slm_optimize_stats, Bash
---

Apply SuperLocalMemory context-optimization. Arguments: $ARGUMENTS

If arguments are empty or "stats":
- Call `slm_optimize_stats()` and report: cache_kv_hits, compress_runs, tokens_saved_compress.

If there is large output to compress (>2000 chars):
- Call `slm_compress(content=<output>, mode="auto", reversible=True)`.
- Work from the compressed form; retain ccr_id if the result is lossy.

Rules (apply strictly):
- Rule 6 — NEVER compress/cache: code intended for Edit/Write; JSON being parsed; secrets/keys/tokens; ccr_ids; content <500 chars.
- Rule 8 — FAIL-OPEN: if ok:false, continue with original; do not retry; do not surface the error unless the user asks.

MCP unavailable → CLI fallback: `slm optimize status`.

SuperLocalMemory v3.6.15 · Qualixar · AGPL-3.0-or-later
