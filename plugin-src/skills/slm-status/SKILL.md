---
name: slm-status
description: Health and optimization stats for SuperLocalMemory — call slm_optimize_stats() for live compression and cache counters (compress_runs, tokens_saved_compress, cache_proxy_hits, cache_proxy_misses, cache_kv_hits, cache_kv_misses); run slm status [--json] for system state (mode, profile, DB size, fact/entity/edge counts) and slm doctor [--json] for preflight including the "Optimize (Surface B)" health line; use together to confirm optimization is actually saving tokens.
when_to_use: "check slm status, health check, is slm working, optimize stats, tokens saved, cache hits, compress runs, slm doctor, preflight, db size, slm info, diagnose slm"
allowed-tools: slm_optimize_stats, Bash
---

# slm-status — Health and Optimize Stats

## Purpose

Use this skill to answer: "Is SLM healthy?", "Is compression/caching actually saving tokens?", and "What does the system look like right now?" It covers three surfaces: the MCP stats tool, the `slm status` CLI, and the `slm doctor` preflight.

## Primary MCP Tool: slm_optimize_stats

```
slm_optimize_stats() -> dict
```

No arguments. Returns counters from the current daemon and MCP process session.

### Return dict (all keys always present)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on success; `False` on internal error |
| `compress_runs` | int | Total compress calls recorded by the daemon (persisted across restarts) |
| `tokens_saved_compress` | int | Cumulative tokens saved by compression (daemon-persisted) |
| `cache_proxy_hits` | int | Proxy-layer cache hits (daemon-persisted) |
| `cache_proxy_misses` | int | Proxy-layer cache misses (daemon-persisted) |
| `cache_kv_hits` | int | MCP KV cache hits — **this MCP process session only**, resets on restart |
| `cache_kv_misses` | int | MCP KV cache misses — **this MCP process session only**, resets on restart |
| `ccr_note` | str \| None | Note about CCR entry count (not tracked per-session; see daemon `/api/v1/metrics`) |
| `note` | str \| None | Scope clarification or error detail |

### Important scope distinction

`compress_runs`, `tokens_saved_compress`, `cache_proxy_hits`, and `cache_proxy_misses` are **daemon-persisted** — they survive MCP restarts and accumulate over the full install lifetime.

`cache_kv_hits` and `cache_kv_misses` are **in-process counters** — they reset to 0 each time the MCP server starts. Use them to gauge cache effectiveness within the current session only.

### Reading whether optimization is saving tokens

```python
stats = await slm_optimize_stats()
if stats["ok"]:
    savings = stats["tokens_saved_compress"]
    kv_hit_rate = (
        stats["cache_kv_hits"] / max(stats["cache_kv_hits"] + stats["cache_kv_misses"], 1)
    )
    # savings > 0 and kv_hit_rate > 0.5 means Surface B is actively reducing costs
```

If `compress_runs` is 0 after several sessions, compression is not being triggered — check daemon config and whether `slm_compress` is being called.

If `cache_kv_hits` is 0 after repeated work, verify key naming consistency (the same key string must be used for set and get).

## Secondary CLI: slm status

```bash
slm status [--json] [--verbose]
```

Reports system-level state — not optimization counters. Canonical fields (WP-02):

- **mode** — active operation mode (e.g. `local`)
- **profile** — current memory profile name
- **DB size** — database file size on disk
- **fact count** — number of stored memory facts
- **entity count** — entity graph node count
- **edge count** — entity graph edge count

`--verbose` / `-v` adds: migration log, daemon port, disabled marker, last version.

`--json` outputs a machine-readable dict with the same fields — preferred for agent consumption.

Example agent-native invocation:

```bash
slm status --json
```

Typical JSON shape (exact field names depend on runtime; use `--json` and read what arrives):

```json
{
  "mode": "local",
  "profile": "code",
  "db_size_mb": 12.4,
  "facts": 384,
  "entities": 201,
  "edges": 519
}
```

Do not rely on the human-readable format for parsing — always use `--json` when the output feeds another tool.

## Secondary CLI: slm doctor

```bash
slm doctor [--json] [--quick]
```

Preflight check covering dependencies, embedding worker, daemon connectivity, and Surface B health. The **"Optimize (Surface B)"** line (WP-03) confirms whether the compression and cache subsystem initialised correctly.

`--quick` skips the daemon and embedding probes — runs only dependency and config checks; faster but incomplete.

`--json` outputs structured results per check — use this in automated health pipelines.

A passing doctor output confirms:
- Python deps present
- Embedding worker reachable
- Daemon responding
- Surface B (Optimize) initialised

A failing "Optimize (Surface B)" line means `slm_compress`, `slm_cache_set`, and `slm_cache_get` may not function correctly — investigate daemon config before relying on those tools.

## Secondary CLI: slm optimize status

```bash
slm optimize status [--json]
```

Shows whether the Optimize module (cache + compress) is currently enabled or disabled at the daemon level. Available subcommands also include `optimize on`, `optimize off`, and `optimize savings`.

The `optimize savings` subcommand accepts:

```bash
slm optimize savings [--since <days>] [--provider anthropic|openai|gemini] [--json]
```

`--since` defaults to 7 days. `--provider` filters by the target AI provider.

Note: the `slm optimize` subcommands have known pre-existing parse-test failures — if a subcommand errors, use `slm_optimize_stats()` via MCP as the authoritative source.

## Recommended Health Workflow

1. Run `slm doctor --json` at session start to confirm all subsystems are up.
2. Call `slm_optimize_stats()` after a batch of work to check token savings.
3. Run `slm status --json` when you need DB size or memory counts.
4. If `ok: false` on any MCP tool — check `note` field, then run `slm doctor` to isolate the failure.

## Fail-Open

`slm_optimize_stats()` never raises. On internal error it returns `ok: false` with all counters at 0. Continue the session — stats unavailability does not affect compression or caching operations.

---

## Profile-aware status (v3.8.0+)

`slm status --json` reports the currently active profile name in the `profile`
field. Use this to confirm which workspace is active before starting work on a
multi-profile setup. To switch the active profile, see `slm-profile`.

---

## Related skills

- `slm-session` — session lifecycle (session_init/close_session)
- `slm-profile` — workspace isolation and profile switching
- `slm-cache` — KV cache performance metrics
- `slm-compress` — reversible context compression

---

SuperLocalMemory v3.8.3 · Qualixar · AGPL-3.0-or-later
