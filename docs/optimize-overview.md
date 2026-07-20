# Optimize Overview — v3.6.11
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SLM exposes optimization controls through a proxy plus content explicitly
routed through MCP tools or a skill. Savings depend on workload, provider,
cacheability, and compression mode.

## Three Surfaces

| Surface | Entry point | Proxy required? | Context window | Cache scope |
|---------|------------|:---------------:|:--------------:|-------------|
| **A — Proxy** | `slm wrap claude` / `ANTHROPIC_BASE_URL` | Yes | Shrinks (intercepts full context) | Full Claude turn |
| **B — MCP tools** | 5 tools in applicable MCP profiles | **No** | Does not intercept the primary window | Results you route through SLM |
| **C — Skill** | `~/.claude/skills/slm-optimize/` | **No** | Depends on agent/tool behavior | Applied only when the agent follows the skill rules |

**Hard constraint:** The primary Claude conversation turn cannot be cached without a proxy. Surfaces B and C cache results you explicitly route through SLM — not the Claude turn itself.

### Surface B: MCP Optimize Tools (v3.6.11)

Five new MCP tools, included in `slm mcp` from v3.6.11+:

| Tool | What it does |
|------|-------------|
| `slm_compress` | Compress text/tool-output. `mode=normalize` (lossless), `auto`, or `aggressive`. Returns `ccr_id` when lossy + reversible. |
| `slm_retrieve` | Recover exact original from `ccr_id` (CCR round-trip). |
| `slm_cache_set` | Cache any string result the agent routes through SLM (file reads, bash output, search results). |
| `slm_cache_get` | Retrieve cached result by key. Returns `hit:True/False`. |
| `slm_optimize_stats` | Session compression + cache statistics. |

The tools are designed to return `ok:False` and preserve the original on
handled optimization failure. Verify this contract against the frozen artifact
before relying on it as a fault-containment boundary.

### Surface C: slm-optimize Skill (v3.6.11)

An agent-behavior instruction file the agent reads and follows. Install:

```bash
mkdir -p ~/.claude/skills/slm-optimize
cp skills/slm-optimize/SKILL.md ~/.claude/skills/slm-optimize/SKILL.md
```

The skill instructs the agent to compress selected context, cache repeated
results, and recover originals via `slm_retrieve`. Actual behavior depends on
the agent following the skill and the selected compression mode.

---

SLM v3.6 **Optimize** is a local-first cost-reduction layer. Surface A sits between your application and your LLM provider, intercepting every API call. Surfaces B/C work at the agent layer with no network interception.

| Lever | Mechanism | Saving | Off by default? |
|-------|-----------|--------|:---------------:|
| **Cache** | Skip an intercepted provider call on an exact match; semantic controls remain experimental | Provider-dependent | Cache ON, Semantic OFF |
| **Compress** | Shrink the prompt before sending — **safe = lossless** normalization; **aggressive = LLMLingua-2 prose only** (opt-in) | Safe: small + lossless · Aggressive: large on prose | Safe mode ON, Aggressive OFF |
| **Align** | Stabilize the prompt prefix to maximize native provider prefix-cache discounts | Lossless extra | ON when compression is ON |

> **v3.6.10:** Cache and Compress are **independent runtime switches** — enable one, both, or neither from the dashboard, applied live with no restart. Compression was rebuilt: the old extractive JSON-string/array/code-body truncation is **removed** (it was lossy); safe mode is now genuinely lossless, and aggressive mode applies LLMLingua-2 to **prose only** — never code, numbers, structured data, instructions, or the current turn.

**Memory** (SLM v3.5's existing engine) is a fourth, parallel lever — it shapes *what is in* the prompt (relevant facts); Optimize decides *whether and how* it is sent. They share plumbing (SQLite engine, embedder), **never share data** (separate `llmcache.db`).

---

## What's New in v3.6

| Feature | What It Does | User Benefit |
|---------|-------------|--------------|
| **Exact Cache** | Byte-identical repeat calls served from local SQLite — zero provider tokens consumed | Save 100% on repeated prompts |
| **Semantic Cache** (experimental) | Controls exist, but stable enablement requires release-linked precision, invalidation, and tenant-isolation evidence | No stable savings or accuracy claim |
| **Lossless safe compression** | Whitespace + compact-JSON normalization only — never removes content; code untouched | Small, zero-risk token reduction; provider prefix-cache stays stable |
| **LLMLingua-2 Prose** (opt-in, aggressive) | Token-classification prose compression (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`) — prose ONLY, never code/numbers/structured/current-turn | Large savings on prose-heavy workloads |
| **CCR (Compressed Context Retrieval)** | Pre-compression originals stored, retrievable byte-exact via UUID (best-effort) | Safety net for aggressive prose compression |
| **Independent runtime toggles** (v3.6.10) | Cache and Compress flip on/off live from the dashboard — no restart | Run cache-only, compress-only, both, or neither, per your workload |
| **Per-agent MCP attribution** (v3.6.10) | `http://127.0.0.1:8765/mcp/{agent_id}` labels audit attribution; authenticated credentials derive mutation authority | Many clients can use one daemon without treating a path label as identity |
| **CacheAligner** | Detects volatile tokens in system prompts and reports a stability signal | Provider-specific effect requires measurement |
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
3. **Compression** — on cache miss, the prompt is compressed (safe = lossless normalization + align; aggressive adds LLMLingua-2 on prose only), then forwarded to the provider
4. **Cache store** — the provider response is cached for future hits
5. **Savings tracked** — every hit, miss, compression, and alignment event is counted and displayed in the dashboard

Handled optimization errors are designed to pass through to the provider. The
frozen release must prove failure behavior for each supported proxy/provider
surface; this is not a guarantee against transport or provider failure.

---

## Quick Start

### Step 1: Install

```bash
npm i -g superlocalmemory
```

For the Python primary path, create and activate a virtual environment, then
run `python -m pip install superlocalmemory`.

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

| Property | Intended contract and required proof |
|----------|-----------|
| **Data isolation** | Separate `llmcache.db` — never touches `memory.db` / `atomic_facts` |
| **Fail-open** | Handled optimization failures should pass through; provider/transport failures remain possible |
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
python -m pip install --upgrade superlocalmemory && slm restart
```

Run that command only while the SLM virtual environment is active. npm users
upgrade with `npm update -g superlocalmemory`.

No data loss. No downtime. Zero configuration changes needed.

---

## Safety

| Concern | Mitigation |
|---------|------------|
| **Adversarial cache poisoning (CacheAttack)** | Tag-based invalidation, tenant isolation, AES-256-GCM integrity protection. Semantic tier blocks 86% hijack class via SAFE-CACHE centroid defense |
| **Compression fidelity loss** | Safe mode = lossless normalization only (no content removed; code untouched). Aggressive mode applies LLMLingua-2 to prose only, shows ⚠ warning before enabling, and CCR stores originals for byte-exact reversal |
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

- [proxy-setup.md](proxy-setup.md) — Per-client proxy activation for documented CLI, SDK, and HTTP paths
- [optimize-cli.md](optimize-cli.md) — Full CLI command reference
- [optimize-config.md](optimize-config.md) — Configuration reference
- [cli-reference.md](cli-reference.md) — Master CLI reference (all commands)
- [configuration.md](configuration.md) — Master configuration guide
- [getting-started.md](getting-started.md) — First-time setup

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com) | AI Reliability Engineering*
