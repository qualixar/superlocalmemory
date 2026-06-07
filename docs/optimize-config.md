# Optimize Configuration — v3.6
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Reference for all SLM v3.6 Optimize configuration options.

---

## Config File

All settings are stored in a single JSON file that the daemon hot-reloads on change:

```bash
~/.superlocalmemory/optimize.json
```

**Hot-reload:** config changes are picked up within 2 seconds via a background watchdog thread. No daemon restart required.

**Write via:**
- **UI** — Dashboard → Optimize tab (runtime toggles, TTL sliders, per-provider settings)
- **CLI** — `slm optimize|cache|compress` commands
- **API** — `PUT /api/optimize/config`
- **Direct** — Edit `optimize.json` manually

---

## Master Switches

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Master Optimize ON/OFF (changed from `false` in v3.6.0 final) |
| `proxy_enabled` | bool | `false` | Proxy server auto-start |
| `config_version` | int | `0` | Incremented on each save — triggers hot-reload |

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
| `semantic_boundary_ceiling` | float | `0.99` | vCache boundary ceiling |
| `semantic_boundary_step` | float | `0.01` | vCache boundary adjustment step |
| `semantic_max_turns_for_semantic` | int | `6` | Max conversation turns before semantic cache is bypassed |
| `semantic_context_window_turns` | int | `3` | Context window size for multi-turn matching |
| `semantic_centroid_distance_floor` | float | `0.70` | Minimum centroid distance for safe return |
| `semantic_verifier_model` | str | `""` | Optional model for verify path |
| `semantic_pad_latency_ms` | int | `50` | Randomized padding for CacheAttack mitigation |

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
| `compress_enabled` | bool | `true` | Enable compression |
| `compress_mode` | str | `"safe"` | `"safe"` (lossless extractive) or `"aggressive"` (lossy prose allowed) |
| `compress_code` | bool | `true` | Code/JSON extractive compression |
| `compress_prose` | bool | `false` | Prose compression (LLMLingua-2, opt-in) |
| `compress_ccr` | bool | `false` | Compressed Context Retrieval (reversible) |
| `compress_align` | bool | `true` | CacheAligner prefix stabilization |
| `compress_json` | bool | `true` | JSON-specific extractive compression |
| `compress_protect_recent` | int | `2` | Number of most recent messages to skip compression |

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
  "enabled": true,
  "proxy_enabled": false,
  "config_version": 42,
  "cache_enabled": true,
  "semantic_enabled": false,
  "semantic_return_threshold": 0.98,
  "semantic_verify_lo": 0.9,
  "semantic_error_target": 0.02,
  "semantic_explore_rate": 0.1,
  "semantic_centroid_defense": true,
  "semantic_multiturn_guard": true,
  "semantic_ann_top_k": 5,
  "compress_enabled": true,
  "compress_mode": "safe",
  "compress_code": true,
  "compress_json": true,
  "compress_prose": false,
  "compress_ccr": false,
  "compress_align": true,
  "compress_protect_recent": 2,
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
