---
description: Show SuperLocalMemory health and optimization counters.
argument-hint: (no arguments)
allowed-tools: slm_optimize_stats, Bash
---

Show SuperLocalMemory status and optimization counters.

1. Run `slm status` via Bash — this is the canonical health check (memory count, profile, daemon state, integrity).
2. Also call `slm_optimize_stats()` via MCP for Surface-B counters (cache_kv_hits, compress_runs, tokens_saved_compress). If ok:false, omit silently — do not surface the error.
3. Summarize both outputs in a concise report. Flag any integrity warnings from `slm status`.

Note: MCP get_status is intentionally NOT used here — it is outside the core profile and would error. Use `slm status` (CLI) + `slm_optimize_stats` (MCP) only.

SuperLocalMemory v3.6.16 · Qualixar · AGPL-3.0-or-later
