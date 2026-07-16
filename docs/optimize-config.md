# Optimize Configuration — v3.6
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Reference for all SLM v3.6 Optimize configuration options.

---

## Config File

Optimize settings are stored in a single JSON file:

```bash
~/.superlocalmemory/optimize.json
```

**Runtime behavior:** When proxy routes were mounted at daemon startup, cache and
compression changes saved through the daemon API reload immediately; direct file
edits and standalone CLI writes are detected by the watchdog within 2 seconds.
Changing `proxy_enabled` is different: it is evaluated during daemon startup, so
config file changes do not mount proxy routes in an already-running daemon.

**Write via:**
- **UI** — Dashboard → Optimize tab (runtime toggles, TTL sliders, per-provider settings)
- **CLI** — `slm optimize|cache|compress` commands
- **API** — `PUT /api/optimize/config`
- **Direct** — Edit `optimize.json` manually

---

## Master Switches

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | SDK adapter enablement; it does not gate proxy route mounting |
| `proxy_enabled` | bool | `false` | Mount proxy routes at daemon startup; changing it requires a daemon restart |
| `config_version` | int | `1` | Revision written by `ConfigStore.save()`; it does not trigger hot-reload |

---

## Cache Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_enabled` | bool | `true` | Enable cache lookups |
| `semantic_enabled` | bool | `false` | Enable vCache semantic cache (opt-in) |
| `semantic_return_threshold` | float | `0.98` | Cosine score threshold for direct return |
| `semantic_verify_lo` | float | `0.90` | Lower bound for verify-and-rewrite path |
| `semantic_error_target` | float | `0.02` | vCache epsilon — max acceptable error rate |
| `semantic_explore_rate` | float | `0.10` | vCache exploration rate |
| `semantic_constant_time` | bool | `false` | Enable side-channel padding (blocks CacheAttack timing-based inference) |
| `semantic_centroid_defense` | bool | `true` | Enable SAFE-CACHE centroid defense |
| `semantic_multiturn_guard` | bool | `true` | Multi-turn guard — skips semantic after `max_turns_for_semantic` |
| `semantic_ann_top_k` | int | `5` | ANN search — top-K candidates |
| `semantic_boundary_init` | float | `0.95` | Initial semantic-cache decision boundary |
| `semantic_boundary_floor` | float | `0.85` | Lowest adaptive semantic-cache boundary |
| `semantic_boundary_ceiling` | float | `0.995` | Highest adaptive semantic-cache boundary |
| `semantic_boundary_step` | float | `0.01` | vCache boundary adjustment step |
| `semantic_max_turns_for_semantic` | int | `6` | Max conversation turns before semantic cache is bypassed |
| `semantic_context_window_turns` | int | `3` | Context window size for multi-turn matching |
| `semantic_centroid_distance_floor` | float | `0.15` | Minimum centroid distance for safe return |
| `semantic_verifier_model` | str | `""` | Optional model for verify path |
| `semantic_pad_latency_ms` | float | `0` | Optional latency padding for CacheAttack mitigation |
| `semantic_centroid_min_similarity` | float | `0.85` | Minimum centroid similarity for a semantic-cache candidate |
| `semantic_max_index_entries` | int | `10000` | Maximum semantic-cache index entries |
| `semantic_max_tenants` | int | `10000` | Maximum semantic-cache tenants |

> **Tuning the semantic tier (read before enabling).** The semantic cache is
> **off by default** because, unlike the exact cache, a too-loose threshold can
> return a *near-duplicate's* answer for a different question (a wrong-answer
> hit). Recommended rollout:
> 1. Start at the default `semantic_return_threshold = 0.98` (conservative;
>    targets a sub-1% false-hit rate).
> 2. Measure your own false-hit rate before trusting it — enable
>    `SLM_OPTIMIZE_CAPTURE=1`, collect real traffic, and replay near-duplicate
>    vs. genuinely-different prompt pairs through `benchmarks/optimize` to see
>    how often a "hit" would have returned the wrong answer.
> 3. Only lower the threshold if your measured false-hit rate stays inside
>    `semantic_error_target`. Raise it (toward 0.99) if you see any wrong hits.
>
> The exact cache (always on when `cache_enabled`) has **zero** false-hit risk —
> it keys on the full system+messages+params hash — so semantic is purely an
> opt-in recall booster, never a correctness dependency.

### Cache TTL

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ttl.exact_seconds` | int | `86400` | Exact cache TTL (24 hours) |
| `ttl.semantic_seconds` | int | `3600` | Semantic cache TTL (1 hour) |
| `ttl.ccr_seconds` | int | `604800` | CCR originals TTL (7 days) |
| `ttl.sweep_interval_seconds` | int | `3600` | TTL sweep interval (1 hour) |

---

## Compression Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `compress_enabled` | bool | `false` | Enable compression |
| `compress_mode` | str | `"safe"` | `"safe"` (lossless whitespace normalization) or `"aggressive"` (lossy prose allowed) |
| `compress_prose` | bool | `false` | Prose compression (LLMLingua-2, opt-in) |
| `compress_protect_recent` | int | `4` | Number of most recent messages to skip compression |

Layer 1 lossless whitespace normalization runs whenever compression is enabled.
Layer 2 prose compression requires `compress_mode="aggressive"`,
`compress_prose=true`, and the optional LLMLingua dependency. The legacy
`code`, `ccr`, and `align` compression subcommands and their configuration
fields were removed in v3.6.10; they are not active configuration options.

---

## Proxy Configuration

Configured at startup via CLI flags, not in `optimize.json`:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8765` | Proxy server port |
| `--provider` | auto-detect | Target provider: `anthropic`, `openai`, `gemini` |
| `--no-compress` | `false` | Disable compression (cache only) |
| `--semantic` | `false` | Enable semantic cache on proxy |
| `--json` | `false` | JSON output |

---

## Pricing Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pricing_overrides` | dict | `{}` | Per-provider pricing overrides (e.g. `{"anthropic": {"input_per_1m_usd": 3.0, "output_per_1m_usd": 15.0}}`) |
| `usd_to_inr_rate` | float | `83.5` | USD to INR conversion rate |
| `prometheus_port` | int | `9091` | Prometheus metrics port |

Default pricing (used when no overrides set):

| Provider | Input ($/1M tokens) | Output ($/1M tokens) |
|----------|:-------------------:|:--------------------:|
| Anthropic | $3.00 | $15.00 |
| OpenAI | $2.50 | $10.00 |
| Gemini | $1.25 | $10.00 |

**Pricing rules:**
- Cache SKIP saves BOTH input + output tokens (whole call avoided)
- Compression saves INPUT tokens only (output not compressed)
- Pricing data older than 90 days is flagged as `is_stale: true`

---

## Storage Configuration

The optimize module uses a separate SQLite database:

| Property | Value |
|----------|-------|
| **Database path** | `~/.superlocalmemory/llmcache.db` |
| **Encryption** | AES-256-GCM with PBKDF2-HMAC-SHA256 key derivation (100K iterations) |
| **File permissions** | `chmod 600` |
| **Corruption recovery** | Invalid files moved to `.corrupt` sidecar — fresh DB created |
| **Database manager** | Reuses SLM's `DatabaseManager` (WAL, busy-timeout, retry) |

**Isolation:** `assert_no_memory_db_tables()` prevents accidental cross-contamination with `memory.db`.

---

## Config File Example

```json
{
  "enabled": false,
  "proxy_enabled": false,
  "config_version": 1,
  "cache_enabled": true,
  "semantic_enabled": false,
  "semantic_return_threshold": 0.98,
  "semantic_verify_lo": 0.9,
  "semantic_error_target": 0.02,
  "semantic_explore_rate": 0.1,
  "semantic_centroid_defense": true,
  "semantic_multiturn_guard": true,
  "semantic_ann_top_k": 5,
  "semantic_boundary_init": 0.95,
  "semantic_boundary_floor": 0.85,
  "semantic_boundary_ceiling": 0.995,
  "semantic_centroid_distance_floor": 0.15,
  "semantic_pad_latency_ms": 0,
  "semantic_centroid_min_similarity": 0.85,
  "semantic_max_index_entries": 10000,
  "semantic_max_tenants": 10000,
  "compress_enabled": false,
  "compress_mode": "safe",
  "compress_prose": false,
  "compress_protect_recent": 4,
  "ttl": {
    "exact_seconds": 86400,
    "semantic_seconds": 3600,
    "ccr_seconds": 604800,
    "sweep_interval_seconds": 3600
  },
  "pricing_overrides": {},
  "usd_to_inr_rate": 83.5,
  "prometheus_port": 9091
}
```

---

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/optimize/config` | Full runtime config as JSON |
| `PUT` | `/api/optimize/config` | Partial config update (hot-reloaded) |
| `GET` | `/api/optimize/savings` | Savings report (USD/INR, hit rate, compress ratio, cache stats) |
| `GET` | `/api/optimize/stats` | Raw MetricsSnapshot (16 fields) |
| `DELETE` | `/api/optimize/cache/clear` | Clear cache for a tenant |

---

## Related Pages

- [optimize-overview.md](optimize-overview.md) — Product overview and quick start
- [optimize-cli.md](optimize-cli.md) — CLI command reference
- [cli-reference.md](cli-reference.md) — Master CLI reference (all commands)
- [configuration.md](configuration.md) — Master configuration guide

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com) | AI Reliability Engineering*
