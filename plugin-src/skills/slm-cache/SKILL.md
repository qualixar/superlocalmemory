---
name: slm-cache
description: KV cache for repeated reads — call slm_cache_get(key) first; on a miss do the expensive operation then slm_cache_set(key, value, ttl_seconds) to store it; on a hit use the returned value directly; always fail-open (hit:false on any error, never raises); saves tokens when the same file, query result, or tool output is read more than once in a session.
when_to_use: "cache file, avoid re-reading, repeated read, cache result, cache tool output, save re-read, reuse across session, cache check, cache hit, cache miss"
allowed-tools: slm_cache_set, slm_cache_get, Bash
---

# slm-cache — KV Cache for Repeated Reads (Surface B)

## Purpose

When the same file, query result, or expensive tool output is needed more than once in a session, fetching it again wastes tokens and time. `slm_cache_set` stores a result under a stable key; `slm_cache_get` retrieves it on subsequent calls. The cache is agent-scoped (automatically namespaced by tenant/agent ID), TTL-bounded, and fail-open.

This is an agent-routed cache — it caches results the agent explicitly routes through SLM. It cannot cache Claude conversation turns.

## Tool: slm_cache_set

```
slm_cache_set(
    key:          str,          # required — cache key (max 512 chars)
    value:        str,          # required — value to store (max 1 MB)
    ttl_seconds:  int = 86400,  # time-to-live in seconds (default 24 h)
) -> dict
```

### Return dict

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on success; `False` on validation error or internal error |
| `stored` | bool | `True` when the value was written to the cache |
| `note` | str \| None | Error detail or `None` on success |

Keys are SHA-256-hashed internally per agent so they do not collide across agents. The raw key string you supply is the only handle you need.

## Tool: slm_cache_get

```
slm_cache_get(
    key: str,   # required — same key used in slm_cache_set
) -> dict
```

### Return dict

| Key | Type | Meaning |
|-----|------|---------|
| `ok` | bool | `True` on clean execution (including miss); `False` on internal error |
| `hit` | bool | `True` when the key exists and has not expired |
| `value` | str \| None | The stored value on a hit; `None` on miss |
| `note` | str \| None | Error detail or `None` |

A miss returns `{"ok": true, "hit": false, "value": null, "note": null}`. `ok: false` means something went wrong internally but the miss behaviour is the same — treat both as a cache miss and proceed with the real fetch.

## Standard Pattern: Cache-Aside

Always check the cache first, then fill on miss:

```python
# 1. Check cache
cached = await slm_cache_get(key="file:/absolute/path/to/config.json")

if cached["hit"]:
    content = cached["value"]
else:
    # 2. Expensive operation (file read, search, API call)
    content = read_file("/absolute/path/to/config.json")

    # 3. Store for the rest of the session
    await slm_cache_set(
        key="file:/absolute/path/to/config.json",
        value=content,
        ttl_seconds=3600,   # 1 h — adjust to data volatility
    )

# 4. Use content
```

## Key Naming Convention

Use a stable, human-readable prefix so keys are recognisable in stats and won't collide accidentally:

| Content type | Suggested prefix | Example |
|---|---|---|
| File read | `file:` | `file:/repo/src/config.py` |
| Search result | `search:` | `search:recall:session_init_context` |
| Tool output | `tool:` | `tool:build_code_graph:/repo` |
| External fetch | `url:` | `url:https://api.example.com/v1/data` |

Key length cap: 512 characters. Keys longer than that are rejected (`ok: false`).

## When Caching Pays Off

**Cache when:**
- You will read the same file more than once in a session.
- A search or recall result is reused across multiple reasoning steps.
- An expensive MCP tool call (graph build, semantic search) produces output that is stable for the session duration.

**Do NOT cache:**
- Volatile data (live API responses that change minute-to-minute, current timestamps, streaming output).
- Secrets, credentials, tokens, or `ccr_id` values (CCR already handles its own storage).
- Data that must be fresh for correctness — a stale cache is worse than a cache miss.
- Intermediate scratchpad text you will discard.

## Fail-Open Guarantee

Neither tool raises an exception. On any internal error:

- `slm_cache_get` returns `{"ok": false, "hit": false, "value": null, ...}` — treat as a miss and proceed with the real fetch.
- `slm_cache_set` returns `{"ok": false, "stored": false, ...}` — log the note if useful, but continue; the value is still available in memory this step.

Never block a task on a cache failure.

## TTL Guidance

| Data type | Suggested TTL |
|---|---|
| Static config / generated file | 86400 s (24 h — the default) |
| Session-specific tool output | 3600 s (1 h) |
| Rapidly changing API data | Do not cache, or 60–300 s |

Set `ttl_seconds` to match how long the data remains valid. After expiry `slm_cache_get` returns a miss automatically.

## Secondary CLI (fallback when MCP is unavailable)

The `slm cache` subcommand exists but has known pre-existing parse-test failures. Prefer the MCP tools above. If you must use CLI:

```bash
slm cache status [--json] [--tenant default]
slm cache clear [--json] [--tenant default]
slm cache invalidate --tag <tag> [--json] [--tenant default]
slm cache ttl --set <seconds> [--semantic <seconds>] [--json] [--tenant default]
slm cache semantic on|off [--json] [--tenant default]
```

These subcommands control daemon-level cache settings. They do not read or write individual cache entries — use the MCP tools for that.

---

## Related skills

- `slm-compress` — for large content reduction; cache and compress work together
- `slm-status` — view `cache_kv_hits`/`cache_kv_misses` counters from `slm_optimize_stats`
- `slm-profile` — cache entries are namespaced per profile; switching profiles gives a fresh cache namespace

---

SuperLocalMemory v3.8.2 · Qualixar · AGPL-3.0-or-later
