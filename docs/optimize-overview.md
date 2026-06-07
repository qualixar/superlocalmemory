# Optimize Overview — v3.6
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SLM v3.6 **Optimize** is a local-first LLM cost-reduction layer that SKIPS, SHRINKS, and DISCOUNTS every LLM call — and remembers — in one install.

It sits between your application and your LLM provider (Anthropic, OpenAI, Gemini), intercepting every API call and applying three levers before the call reaches the provider:

| Lever | Mechanism | Saving | Off by default? |
|-------|-----------|--------|:---------------:|
| **Cache** | Skip the call entirely on repeats — exact-match → vCache-gated semantic | **100% on a hit** | Cache ON, Semantic OFF |
| **Compress** | Shrink the prompt before sending — extractive structured + LLMLingua-2 prose + CCR reversible | **60–95% on a miss** | Safe mode ON, Aggressive OFF |
| **Align** | Stabilize the prompt prefix to maximize native provider prefix-cache discounts | Lossless extra | ON when compression is ON |

**Memory** (SLM v3.5's existing engine) is a fourth, parallel lever — it shapes *what is in* the prompt (relevant facts); Optimize decides *whether and how* it is sent. They share plumbing (SQLite engine, embedder), **never share data** (separate `llmcache.db`).

---

## What's New in v3.6

| Feature | What It Does | User Benefit |
|---------|-------------|--------------|
| **Exact Cache** | Byte-identical repeat calls served from local SQLite — zero provider tokens consumed | Save 100% on repeated prompts |
| **Semantic Cache** (opt-in) | vCache-powered learned thresholds with SAFE-CACHE centroid defense — near-duplicate queries return cached results within error bound | Save 100% on near-repeats with verified accuracy |
| **Extractive Compression** | Structure-preserving compression for JSON, code, and tool outputs — lossless, reversible | 60–95% fewer input tokens, zero accuracy regression |
| **LLMLingua-2 Prose** (opt-in, warned) | Extractive prose summarization for open-ended chat and brainstorming | Additional savings on prose workloads |
| **CCR (Compressed Context Retrieval)** | Pre-compression originals stored and retrievable byte-exact via UUID | Reverse any compression with full fidelity |
| **CacheAligner** | Detects volatile tokens (UUIDs, timestamps) in system prompts — logs stability score | Maximizes native provider prefix-cache discounts (Anthropic 90%, OpenAI 50%) |
| **Interception Proxy** | HTTP proxy on port 8765 serving Anthropic, OpenAI, and Gemini surfaces | Zero-code integration — just set `base_url` |
| **SDK Wrappers** | `withSLM(OpenAI())` — in-process interception, no network hop | Drop-in for existing SDK code |
| **Agent Wrapping** | `slm wrap claude` — one command configures base_url and launches the agent | Fastest path to savings |
| **Savings Dashboard** | Live USD/INR/tokens saved, hit rate, compression ratio, cache size | Real-time cost visibility |
| **Hot-Reload Config** | UI/CLI writes `optimize.json` — daemon reloads within 2 seconds, no restart | No downtime for config changes |

---

## Architecture

```
                  ┌──────────────┐   ┌──────────────┐
                  │  Proxy       │   │  SDK Wrapper  │
   Interception   │  port 8765   │   │  (py -> ts)   │
   surfaces       │  /v1/*       │   │  withSLM()    │
                  └──────┬───────┘   └──────┬────────┘
                         └─────────┬─────────┘
                  ┌──────────────────▼──────────────────┐
   Core           │       optimize/                     │
                  │  Cache · Compress · Align           │
                  │  Exact -> Semantic(gated) -> Passthru│
                  │  Fail-open · Safe defaults          │
                  └───┬──────────────┬────────────┬──────┘
                      │ storage      │ embeddings │
             ┌────────▼───┐  ┌───────▼──────┐  ┌─▼────────────────┐
   Backends   │ llmcache.db│  │ nomic-embed  │  │ fastembed        │
   (separate  │ (SQLite)   │  │ (SLM mode)   │  │ (standalone)     │
    files)    └────────────┘  └──────────────┘  └──────────────────┘
```

### How It Works

1. Your app sends an LLM request (via proxy, SDK wrapper, or agent wrapping)
2. **Cache check** — if an exact match exists in `llmcache.db`, the cached response returns immediately — **zero provider tokens consumed**
3. **Compression** — on cache miss, the prompt is compressed (extractive JSON/code → LLMLingua-2 prose → alignment), then forwarded to the provider
4. **Cache store** — the provider response is cached for future hits
5. **Savings tracked** — every hit, miss, compression, and alignment event is counted and displayed in the dashboard

All errors are **fail-open** — any cache/compress/proxy error passes through to the provider. Your calls never break.

---

## Quick Start

### Step 1: Install

```bash
pip install -U superlocalmemory==3.6.0
# or: npm i -g superlocalmemory@3.6
```

### Step 2: Verify

```bash
slm optimize status
```

You should see Optimize: ON with cache and compress enabled.

### Step 3: Activate (three ways)

**A) Agent wrapping (recommended for Claude Code):**
```bash
slm wrap claude
```
This starts the proxy, sets `ANTHROPIC_BASE_URL`, and launches Claude Code.

**B) Proxy for any OpenAI-compatible client:**
```bash
slm proxy
# Set base_url to http://127.0.0.1:8765 in your client
```

**C) SDK adapter (Python):**
```python
from superlocalmemory.optimize.adapters.openai_adapter import withSLM
from openai import OpenAI
client = withSLM(OpenAI())  # same interface, zero API change
```

### Step 4: See savings

```bash
slm optimize savings --since 1
```

Or open the dashboard: `slm serve` → http://localhost:8700 → **Optimize** tab.

---

## Key Properties

| Property | Guarantee |
|----------|-----------|
| **Data isolation** | Separate `llmcache.db` — never touches `memory.db` / `atomic_facts` |
| **Fail-open** | Any cache/compress/proxy error → passthrough to provider. Calls never break |
| **Safe defaults** | Optimize ON, Cache ON, Semantic OFF, Compress safe mode, Aggressive OFF, CCR OFF. No behavior change without explicit enable |
| **Encryption at rest** | AES-256-GCM on all cache values — random per-install salt, `chmod 600` on DB files |
| **No pickle** | JSON-only serialization (CWE-502 safe) |
| **No LiteLLM** | Zero dependency on the shipped-malware package |
| **Hot-reload** | Config changes take effect within 2 seconds — no daemon restart |
| **Tenant isolation** | All cache entries scoped by tenant ID — no cross-tenant collisions |

---

## Pricing Model (as of 2026-06-07)

| Provider | Input ($/1M tokens) | Output ($/1M tokens) |
|----------|:-------------------:|:--------------------:|
| Anthropic | $3.00 | $15.00 |
| OpenAI | $2.50 | $10.00 |
| Gemini | $1.25 | $10.00 |

- **Cache SKIP** saves both input AND output tokens (whole call avoided)
- **Compression** saves INPUT tokens only (output not compressed)
- INR conversion at configurable rate (default: 83.5)

---

## CLI Commands

```bash
slm optimize status|on|off|savings [--since N] [--provider P] [--json]
slm cache    status|clear|invalidate --tag <t>|ttl --set <s> --semantic <s>|semantic on|off
slm compress status|mode safe|aggressive|code on|off|prose on|off|ccr on|off|align on|off
slm proxy    [--port P] [--provider P] [--no-compress] [--semantic]
slm wrap     <agent> [--list] [--persistent] [--dry-run]
slm help-optimize [cache|compress|proxy|agents|safety]
```

See [optimize-cli.md](optimize-cli.md) for full CLI reference.

---

## API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/optimize/config` | Full runtime config as JSON |
| `PUT` | `/api/optimize/config` | Partial config update (hot-reloaded) |
| `GET` | `/api/optimize/savings` | Savings report (USD/INR, hit rate, compress ratio) |
| `GET` | `/api/optimize/stats` | Raw MetricsSnapshot (16 fields) |
| `DELETE` | `/api/optimize/cache/clear` | Clear cache for a tenant |

---

## Backward Compatibility

v3.6 is a **strict superset** of v3.5:

- **All existing CLI commands work identically** — 17 new commands are additive
- **All existing MCP tools retain their signatures** — 2 optional read-only tools added
- **All existing API endpoints are unchanged** — 5 new optimize endpoints
- **Existing retrieval behavior preserved** — Optimize is OFF by default until enabled
- **SLM MCP server unchanged** — no changes to memory tools, hooks, or file-injection
- **`memory.db` untouched** — all optimize data lives in separate `llmcache.db`

**Migration is a single command:**
```bash
pip install -U superlocalmemory && slm restart
```

No data loss. No downtime. Zero configuration changes needed.

---

## Safety

| Concern | Mitigation |
|---------|------------|
| **Adversarial cache poisoning (CacheAttack)** | Tag-based invalidation, tenant isolation, AES-256-GCM integrity protection. Semantic tier blocks 86% hijack class via SAFE-CACHE centroid defense |
| **Compression fidelity loss** | Safe mode = extractive only (structure-preserving, lossless). Aggressive mode shows ⚠ warning before enabling. CCR stores originals for byte-exact reversal |
| **API key leak** | Header redaction in proxy logs, SSRF allowlist on upstream URLs |
| **Data contamination** | Separate `llmcache.db` with `assert_no_memory_db_tables()` guard |
| **Corruption recovery** | Invalid SQLite files moved to `.corrupt` sidecar — fresh DB created automatically |

---

## License & Attribution

- SuperLocalMemory: **AGPL-3.0-or-later**
- Extract patterns adapted from **Headroom** (Apache-2.0) → NOTICE preserved per project
- Cache patterns adapted from **OmniCache** (MIT) → license notice preserved
- Prose compression via **LLMLingua-2** (MIT, Microsoft) → bundled when enabled

See `ATTRIBUTION.md` and `NOTICE` for full attribution.

---

## Related Pages

- [optimize-cli.md](optimize-cli.md) — Full CLI command reference
- [optimize-config.md](optimize-config.md) — Configuration reference
- [cli-reference.md](cli-reference.md) — Master CLI reference (all commands)
- [configuration.md](configuration.md) — Master configuration guide
- [getting-started.md](getting-started.md) — First-time setup

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com) | AI Reliability Engineering*
