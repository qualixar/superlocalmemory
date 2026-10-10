---
name: slm-status
description: Health and optimization stats for SuperLocalMemory — call slm_optimize_stats() for live compression and cache counters (compress_runs, tokens_saved_compress, cache_proxy_hits, cache_proxy_misses, cache_kv_hits, cache_kv_misses); run slm status [--json] for system state (mode, profile, DB size, fact/entity/edge counts) and slm doctor [--json] for preflight including the "Optimize (Surface B)" health line; use together to confirm optimization is actually saving tokens.
when_to_use: "check slm status, health check, is slm working, optimize stats, tokens saved, cache hits, compress runs, slm doctor, preflight, db size, slm info, diagnose slm, slm media status, are pictures on, slm features, picture search not working, slm media repair, embedder status"
allowed-tools: slm_optimize_stats, get_status, get_brain_evidence_status, Bash
---

# slm-status — Health and Optimize Stats

## Purpose

Use this skill to answer: "Is SLM healthy?", "Is compression/caching actually saving tokens?", and "What does the system look like right now?" It covers the MCP stats and status tools, the `slm status` CLI, the `slm doctor` preflight, and the store-health commands.

## Primary MCP Tool: slm_optimize_stats

```
slm_optimize_stats() -> dict
```

No arguments. Returns the counters the daemon has persisted.

### Return dict (all keys always present)

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on success; `False` on internal error |
| `compress_runs` | int | Total compress calls recorded by the daemon (persisted across restarts) |
| `tokens_saved_compress` | int | Cumulative tokens saved by compression (daemon-persisted) |
| `cache_proxy_hits` | int | Proxy-layer cache hits (daemon-persisted) |
| `cache_proxy_misses` | int | Proxy-layer cache misses (daemon-persisted) |
| `cache_kv_hits` | int | Hits on the `slm_cache_get` key-value cache (daemon-persisted) |
| `cache_kv_misses` | int | Misses on the `slm_cache_get` key-value cache (daemon-persisted) |
| `ccr_note` | str \| None | Note about CCR entry count (not tracked per-session; see daemon `/api/v1/metrics`) |
| `note` | str \| None | Scope clarification or error detail |

### Important scope distinction

All six counters are **daemon-persisted**: they survive MCP restarts and accumulate over the install's lifetime, so they are totals, not per-session figures. Only if the persisted KV counters cannot be read does the tool fall back to this process's own tally (the `note` field says so). To judge one stretch of work, call `slm_optimize_stats` before and after and subtract.

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

If `cache_kv_hits` does not grow after repeated work, verify key naming consistency (the same key string must be used for set and get, by the same agent).

## Secondary CLI: slm status

```bash
slm status [--json] [--verbose]
```

Reports system-level state — not optimization counters. Canonical fields:

- **mode** — active operation mode (`a`, `b` or `c`)
- **provider** — the LLM provider for modes B and C, or `none`
- **profile** — current memory profile name
- **db_size_mb**, **db_path**, **base_dir** — where the store is and how big
- **fact_count**, **entity_count**, **edge_count** — counts for the active profile
- **version**, **profile_generation**, **projection_queue_depth**, and (daemon running) **saves_waiting** and **unreadable_saves**

`--verbose` / `-v` adds: the disabled marker, last booted version, and daemon port.

`--json` prints the standard envelope `{"success", "command", "version", "data": {...}}`; read the fields above from `data`. Prefer it for agent consumption:

```bash
slm status --json
```

```json
{"success":true,"command":"status","version":"...","data":{"mode":"a","provider":"none","profile":"default","db_size_mb":12.4,"fact_count":384,"entity_count":201,"edge_count":519,"profile_generation":0,"projection_queue_depth":0,"saves_waiting":0,"unreadable_saves":0}}
```

The MCP equivalent is `get_status(profile_id="")`, which returns the same fields at the top level (it is not part of the smallest `core` tool set). `saves_waiting` above zero means saved memories are durable but still being indexed (searchable within seconds). `unreadable_saves` above zero means that many saves could not be read back with this computer's key and were kept unchanged in the admission journal (the daemon log has their ids); a negative value means the journal did not answer. Status does not report a user role.

Do not rely on the human-readable format for parsing — always use `--json` when the output feeds another tool.

## Secondary CLI: slm doctor

```bash
slm doctor [--json] [--quick] [--deep] [--fix]
```

Preflight check covering dependencies, embedding worker, daemon connectivity, and Surface B health. The **"Optimize (Surface B)"** line confirms whether the compression and cache subsystem initialised correctly.

`--quick` skips the daemon and embedding probes — runs only dependency and config checks; faster but incomplete.

`--deep` reads every database page (`PRAGMA integrity_check`) instead of the structural check; slow on a large store. `--fix` repairs what it can (re-downloads missing models, installs sqlite-vec) before checking, then reports.

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

`slm_optimize_stats()` via MCP is the same data as `slm optimize savings`; use whichever surface you have.

## Store health and recovery

```bash
slm db integrity [--pages] [--json]   # read-only, safe while SLM runs
slm db repair [--json]                # preview of what a repair would do (read-only)
slm db repair --apply --root <data folder> [--batch-size N] [--pause-ms MS] [--max-seconds S]
slm db repair --undo <run_id> --root <data folder>
slm ops list | status | resolve <operation_id> --action retry|force_reconcile|cancel
slm brain status [--json]             # observation-only Living Brain evidence totals
slm embedder status                   # progress of an embedding-model switch
slm models                            # recommended and installed local models
```

`slm db integrity` answers five separate questions so one cannot hide another:
page integrity (only with `--pages`), relational integrity (orphan rows, erased
words still stored), source fidelity (facts withheld from answers or no longer
saying what their memory said), projection readiness (keyword, vector and date
search work still owed), and any repair running or last run. It prints counts
only, never memory text.

`slm db repair` fixes leftover rows, erased-word leftovers, unfinished deletes
and memories that lost their searchable fact, with receipts and an undo. It
previews by default; `--apply` and `--undo` insist on `--root` naming the data
folder you mean, and refuse any other. A repair never brings back anything that
was erased, deleted or withheld. `slm ops` lists failed, stuck or degraded
operations and, for an owner or admin, resolves them. The MCP counterpart for
Living Brain totals is `get_brain_evidence_status(profile_id="")`; it only
observes and does not change recall, ranking or review.

## Pictures and documents: status and repair

Pictures and documents are an optional feature, off by default. Check them with
the CLI (the MCP `get_status` does not report them):

```bash
slm media status [--json]     # off | on, setting up (state, percent) | on | failed, with the step
slm features [--json]         # what is on and what can be turned on: pictures and documents, folder sources, bot messages
slm media repair --dry-run    # count pictures that cannot be found by what they show yet (owner or admin)
slm media repair              # re-embed them; up to four minutes per run, run again if it says some are left
slm media gc                  # report picture records without a memory, files without a record (removes nothing)
slm embedder status           # progress of an embedding-model switch or the memory-engine upgrade
```

If `slm media status` says "set-up failed", run `slm doctor` for the step that
failed. The five picture tools (`remember_media`, `get_media`,
`remember_document`, `media_status`, `media_upload_link`) are listed only while
the feature is on and only in the `full`, `power` and `whole` tool sets; a
restart (`slm restart`) starts the picture worker. A document's own progress
is the `media_status` MCP tool, not this command. Details, limits and the
16 GB requirement: `slm-media`. `slm status` also prints an "Embedding switch" line while a model switch or
the memory-engine upgrade is pending.

## Recommended Health Workflow

1. Run `slm doctor --json` at session start to confirm all subsystems are up.
2. Call `slm_optimize_stats()` after a batch of work to check token savings.
3. Run `slm status --json` when you need DB size or memory counts.
4. If `ok: false` on any MCP tool — check `note` field, then run `slm doctor` to isolate the failure.
5. If recall seems to miss memories that were saved, run `slm db integrity` before concluding anything.
6. If a picture will not come back for a question about what it shows, run `slm media status`, then `slm media repair --dry-run`.

## Fail-Open

`slm_optimize_stats()` never raises. On internal error it returns `ok: false` with all counters at 0. Continue the session — stats unavailability does not affect compression or caching operations.

---

## Profile-aware status

`slm status --json` reports the active profile in `data.profile`. Use it to confirm which workspace is active before starting work on a multi-profile setup. `get_status(profile_id="<name>")` counts another profile without moving the active one. To change the active profile, see `slm-profile`.

---

## Related skills

- `slm-session` — session lifecycle (session_init/close_session)
- `slm-profile` — workspace isolation and profile switching
- `slm-cache` — KV cache performance metrics
- `slm-compress` — reversible context compression
- `slm-media` — pictures, PDFs and folders
- `slm-web-access` — the Connected apps page and web-app connection states

---

SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later
