# Optimize CLI Reference — v3.6
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Complete reference for all SLM v3.6 Optimize CLI commands.

---

## Overview

SLM v3.6 adds 6 new top-level commands under `slm`:

```bash
slm optimize   ...    # Master Optimize module control
slm cache      ...    # Cache sub-control
slm compress   ...    # Compression sub-control
slm proxy      ...    # Start/proxy control
slm wrap       ...    # Agent activation
slm help-optimize     # Full developer reference
```

All commands accept `--json` for machine-readable output.

---

## `slm optimize` — Master Module Control

### `slm optimize status`

Show all Optimize settings — cache state, compression mode, proxy status, config version.

```bash
slm optimize status
slm optimize status --json
```

Example output:
```
Optimize: ON
  Cache:     enabled  (exact: 86400s TTL, semantic: OFF)
  Compress:  enabled  (mode: safe, code: ON, prose: OFF, CCR: OFF)
  Proxy:     not running
  Config:    ~/.superlocalmemory/optimize.json  (version 61)
```

### `slm optimize on`

Enable cache + compress (does NOT start the proxy).

```bash
slm optimize on
```

### `slm optimize off`

Disable all Optimize features. Proxy (if running) passes through calls unchanged.

```bash
slm optimize off
```

### `slm optimize savings`

Token/cost savings report from metrics counters.

```bash
slm optimize savings                    # Last 7 days
slm optimize savings --since 30         # Last 30 days
slm optimize savings --provider anthropic  # Filter by provider
slm optimize savings --json
```

Example output:
```
Savings (last 7 days):
  Exact cache hits:        43   (127,580 input tokens saved)
  Semantic cache hits:      0
  Tokens saved (total): 153,096
  Estimated savings:  ~$2.2964 (at $3.00/M tokens)
  Pricing date: 2026-06-07
```

---

## `slm cache` — Cache Control

### `slm cache status`

Entry count, DB size, TTLs, hit rate.

```bash
slm cache status                         # Default tenant
slm cache status --tenant myproject      # Specific tenant
slm cache status --json
```

Example output:
```
Cache status:
  Entries (exact):   127  (not expired)
  Semantic index:    OFF
  DB size:           2.4 MB  (~/.superlocalmemory/llmcache.db)
  TTL (exact):       86400s
  TTL (semantic):    3600s
  Hits:              43
  Misses:            89
  Hit rate:          32.6%
```

### `slm cache clear`

Delete all entries for a tenant.

```bash
slm cache clear                          # Clear default tenant
slm cache clear --tenant myproject       # Clear specific tenant
slm cache clear --json
```

### `slm cache invalidate --tag <t>`

Delete entries whose tag array contains the specified tag.

```bash
slm cache invalidate --tag "project:slm"
slm cache invalidate --tag "user:varun" --json
```

### `slm cache ttl`

Set TTL values for cache entries.

```bash
slm cache ttl --set 3600           # 1 hour exact cache
slm cache ttl --semantic 7200      # 2 hour semantic cache
slm cache ttl --set 86400 --semantic 3600  # Both at once
```

Defaults: exact=86400s (24h), semantic=3600s (1h).

### `slm cache semantic on|off`

Enable or disable semantic cache lookup (vCache).

```bash
slm cache semantic on
slm cache semantic off
```

Note: requires embedding model (~500MB). Run `slm warmup` if not already done.

---

## `slm compress` — Compression Control

### `slm compress status`

Show compression mode, per-channel state.

```bash
slm compress status
slm compress status --json
```

Example output:
```
Compression status:
  Enabled:  yes
  Mode:     safe
  Code:     ON  (extractive JSON/code — lossless structure)
  Prose:    OFF
  CCR:      OFF (reversible context retrieval)
```

### `slm compress mode safe|aggressive`

Set compression aggressiveness.

```bash
slm compress mode safe          # Default — lossless, structure-preserving
slm compress mode aggressive    # ⚠ May reduce output fidelity
```

**Safe mode** (default): extractive JSON/code compression only — structure-preserving, lossless. Production-safe for all use cases.

**Aggressive mode**: also enables LLMLingua-2-style prose compression — may omit nuance, hedges, or low-salience context. Shows a warning before activating. DO NOT use for code generation, legal/compliance text, math, or exact-output tasks.

### `slm compress code on|off`

Enable/disable code/JSON compression (extractive, lossless).

```bash
slm compress code on
slm compress code off
```

### `slm compress prose on|off`

Enable/disable prose compression (LLMLingua-2-style extractive summarization).

```bash
slm compress prose on
slm compress prose off
```

### `slm compress ccr on|off`

Enable/disable Compressed Context Retrieval — stores pre-compression originals in `llmcache.db` for byte-exact reversal.

```bash
slm compress ccr on
slm compress ccr off
```

### `slm compress align on|off`

Enable/disable CacheAligner — detects volatile tokens (UUIDs, timestamps) in system prompts for prefix-stabilization analysis.

```bash
slm compress align on
slm compress align off
```

---

## `slm proxy` — Interception Proxy

Start or report the optimization proxy. Proxy intercepts LLM API calls, applies cache lookup and compression before forwarding.

```bash
slm proxy                              # Default port 8765
slm proxy --port 8080                  # Custom port
slm proxy --provider anthropic         # Provider-specific surface
slm proxy --no-compress                # Cache only, no compression
slm proxy --semantic                   # Enable semantic cache on proxy
slm proxy --json
```

**Endpoints served on proxy:**
- `POST /v1/messages` — Anthropic Messages API
- `POST /v1/chat/completions` — OpenAI Chat API
- `POST /v1beta/models/*` — Gemini native API
- Compatible with all OpenAI-compatible clients

---

## `slm wrap` — Agent Activation

Activate an agent through the SLM proxy. One command: starts proxy + sets environment + launches agent.

```bash
slm wrap claude                       # Claude Code (recommended)
slm wrap claude -- my prompt          # Pass args to agent
slm wrap cursor                       # Cursor
slm wrap aider -- --model gpt-4       # Aider with model
slm wrap antigravity                  # Antigravity
slm wrap --list                       # List registered agents
slm wrap --persistent                 # Write to agent's config file
slm wrap --dry-run                    # Preview without executing
```

**Supported agents:** `claude`, `claude-settings`, `antigravity`, `cline`, `opencode`, `cursor`, `aider`, `codex`, `copilot`, `generic`

**SDK adapters (Python) — manual wrap:**

```python
from superlocalmemory.optimize.adapters.openai_adapter import withSLM
from openai import OpenAI
client = withSLM(OpenAI())        # Same interface — zero API change

from superlocalmemory.optimize.adapters.anthropic_adapter import withSLM
from anthropic import Anthropic
client = withSLM(Anthropic())
```

---

## `slm help-optimize` — Developer Reference

Full reference with per-agent setup recipes and safety notes.

```bash
slm help-optimize                     # Full reference
slm help-optimize cache               # Cache subcommand reference
slm help-optimize compress            # Compress + safety warning
slm help-optimize agents              # Per-agent setup recipes
slm help-optimize safety              # Compression safety warning only
```

---

## Related Pages

- [optimize-overview.md](optimize-overview.md) — Product overview and quick start
- [optimize-config.md](optimize-config.md) — Configuration reference
- [cli-reference.md](cli-reference.md) — Master CLI reference (all commands)

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com) | AI Reliability Engineering*
