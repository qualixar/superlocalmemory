<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory V3.6</h1>
<p align="center"><strong>Save up to 90% on every LLM API call. Cache. Compress. Remember.</strong><br/><em>The only local-first memory system that SKIPS repeat calls (100% saved), SHRINKS prompts 60-95%, and REMEMBERS everything — locally, for free. For Claude Code, Cursor, Windsurf, and any AI client.</em></p>
<p align="center"><code>v3.6.0 "Optimize"</code> — <strong>Cache & Compress & Align. Save up to 90% on every LLM API call — locally.</strong> One command: <code>slm wrap claude</code><br>Also includes v3.5 Scale-Ready: 6-channel recall &lt;1s, CozoDB + LanceDB, Core Memory Block. Your database auto-migrates.</p>
<p align="center"><strong>Backed by 3 published research papers</strong> (arXiv preprints + Zenodo-archived) · <a href="https://arxiv.org/abs/2603.02240">arXiv:2603.02240</a> · <a href="https://arxiv.org/abs/2603.14588">arXiv:2603.14588</a> · <a href="https://arxiv.org/abs/2604.04514">arXiv:2604.04514</a></p>

<p align="center">
  <code>Saves up to 90% on LLM API costs</code> &nbsp;·&nbsp; <code>+10.6pp vs Mem0 zero-LLM</code> &nbsp;·&nbsp; <code>85% Open-Domain (best zero-LLM score)</code> &nbsp;·&nbsp; <code>EU AI Act Ready</code>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14588"><img src="https://img.shields.io/badge/arXiv-2603.14588-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="https://img.shields.io/badge/Saves_90%25_on_LLM_Costs-22c55e?style=for-the-badge"><img src="https://img.shields.io/badge/Saves_90%25_on_LLM_Costs-22c55e?style=for-the-badge" alt="Saves 90% on LLM Costs"/></a>
  <a href="https://pypi.org/project/superlocalmemory/"><img src="https://img.shields.io/pypi/v/superlocalmemory?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/superlocalmemory"><img src="https://img.shields.io/npm/v/superlocalmemory?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=for-the-badge" alt="AGPL v3"/></a>
  <a href="#eu-ai-act-compliance"><img src="https://img.shields.io/badge/EU_AI_Act-Design_Compliant-brightgreen?style=for-the-badge" alt="EU AI Act Design Compliant"/></a>
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Web-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/MCP-Native-blue?style=for-the-badge" alt="MCP Native"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/CLI-Agent--Native-green?style=for-the-badge" alt="CLI Agent-Native"/></a>
  <a href="#multilingual-embedding-support"><img src="https://img.shields.io/badge/Multilingual-30%2B_Languages-ff69b4?style=for-the-badge" alt="Multilingual 30+ Languages"/></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/c3b54a1d-f62a-4ea7-bba7-900435e7b3ab" width="800" autoplay loop muted playsinline></video>
</p>

---

<details>
<summary><strong>What's New in V3.6 — Optimize: SKIP, SHRINK, DISCOUNT, REMEMBER</strong> (click to expand)</summary>

> V3.6 is the only local-first layer that SKIPS repeat LLM calls (cache: 100% saved), SHRINKS prompts 60-95% (compress: extractive + LLMLingua-2), and DISCOUNTS prefix costs (align: native KV-cache) — and remembers everything — in one install. **Your first cache hit pays for the install time. Hours of coding on repeat, minimal API cost.**

### The Three Levers

| Lever | Mechanism | Saving | Off by default? |
|-------|-----------|:------:|:---------------:|
| **Cache** | Skip repeat calls — exact-match SQLite lookup, vCache-gated semantic (opt-in) | **100% on a hit** (input + output) | Cache ON, Semantic OFF |
| **Compress** | Shrink prompts — extractive JSON/code (lossless) + LLMLingua-2 prose (opt-in) | **60–95% on a miss** (input only) | Safe mode ON, Aggressive OFF |
| **Align** | Stabilize prefix — maximize provider prefix-cache discounts | **Lossless extra** | ON when compression is ON |

**Memory** (v3.5's existing engine) runs in parallel — it shapes *what is in* the prompt (relevant facts); Optimize decides *whether and how* it is sent.

### Quick Start

```bash
# One command to start saving
slm wrap claude
# Your first repeat prompt → CACHE HIT → $0.00
# Your first long prompt → COMPRESSED 70% → $0.00 per token saved
```

### New CLI Commands (6 total)

| Command | What It Does |
|:--------|:-------------|
| `slm optimize status\|on\|off\|savings` | Master Optimize control + savings report (USD/INR/tokens) |
| `slm cache status\|clear\|invalidate\|ttl\|semantic` | Cache sub-control — exact + semantic tiers |
| `slm compress status\|mode\|code\|prose\|ccr\|align` | Compression control — per-channel toggles |
| `slm proxy [--port] [--provider]` | Start the interception proxy (port 8765) |
| `slm wrap <agent>` | Proxy-activate an agent — one command to start saving |
| `slm help-optimize [topic]` | Full developer reference + per-agent setup recipes |

### Savings Dashboard

All metrics tracked and displayed live — from the dashboard (Optimize tab) or CLI:

```bash
slm optimize savings --since 7
# Savings (last 7 days):
#   Exact cache hits:        43   (127,580 input tokens saved)
#   Tokens saved (total): 153,096
#   Estimated savings:  ~$2.30 (at $3.00/M tokens — Anthropic rates)
```

### Enable / Disable

```bash
slm optimize on                      # Enable cache + compress
slm optimize off                     # Disable (proxy passes through)
slm cache semantic on                # Enable semantic cache (needs embedding model)
slm compress mode aggressive         # Enable prose compression (with safety warning)
```

**Safety defaults:** Optimize ON. Safe mode ON (extractive only — lossless, production-safe). Semantic OFF. Aggressive OFF. No behavior change until you explicitly enable features.

### How It Works

```
Your App → Proxy/SDK/Wrap → Cache Check → HIT → Return Cached (0 tokens)
                                   |
                                 MISS
                                   |
                            Compress → Provider → Store in Cache
                            60-95%     + Align
```

- **Fail-open** — any error passes through. Your calls never break.
- **Separate database** — `llmcache.db` never touches `memory.db`. AES-256-GCM at rest.
- **Hot-reload config** — UI/CLI writes `~/.superlocalmemory/optimize.json`, daemon reloads in 2s.

### Links

Full docs:
- [Optimize Product Overview](docs/optimize-overview.md)
- [Optimize CLI Reference](docs/optimize-cli.md)
- [Optimize Config Reference](docs/optimize-config.md)
- [Wiki: V3.6 Overview](https://github.com/qualixar/superlocalmemory/wiki/V3.6-Overview)
- [Website: v3.6 Optimize](https://superlocalmemory.com/optimize)

</details>

---

<details>
<summary><strong>What's New in V3.3 — The Living Brain Evolves</strong> (click to expand)</summary>

> V3.3 gives your memory a lifecycle. Memories strengthen when used, fade when neglected, compress when idle, and consolidate into reusable patterns — all automatically, all locally. Your agent gets smarter the longer it runs.

### Features at a Glance

- **Adaptive Memory Lifecycle** — memories naturally strengthen with use and fade when neglected. No manual cleanup, no hardcoded TTLs.
- **Smart Compression** — embedding precision adapts to memory importance. Low-priority memories compress up to 32x. High-value memories stay full-resolution.
- **Cognitive Consolidation** — the system automatically extracts patterns from clusters of related memories. One decision referenced 50 times becomes one reusable insight.
- **Pattern Learning** — auto-learned soft prompts injected into your agent's context at session start. The system teaches itself what matters to you.
- **Hopfield Retrieval (6th Channel)** — vague or partial queries now complete themselves. Ask half a question, get the whole answer.
- **Process Health** — orphaned SLM processes detected and cleaned automatically. No more zombie workers eating RAM.

### New CLI Commands

```bash
# Run a memory lifecycle review — strengthens active memories, archives neglected ones
slm decay

# Run smart compression — adapts embedding precision to memory importance
slm quantize

# Extract reusable patterns from memory clusters
slm consolidate --cognitive

# View auto-learned patterns that get injected into agent context
slm soft-prompts

# Clean up orphaned SLM processes
slm reap
```

### New MCP Tools

| Tool | Description |
|:-----|:------------|
| `forget` | Programmatic memory archival via lifecycle rules |
| `quantize` | Trigger smart compression on demand |
| `consolidate_cognitive` | Extract and store patterns from memory clusters |
| `get_soft_prompts` | Retrieve auto-learned patterns for context injection |
| `reap_processes` | Clean orphaned SLM processes |
| `get_retention_stats` | Memory lifecycle analytics |

### Mode A/B Memory Improvements

| Metric | V3.2 | V3.3 | Change |
|:-------|:----:|:----:|:------:|
| RAM usage (Mode A/B) | ~4GB | ~40MB | **100x reduction** |
| Retrieval channels | 5 | 6 | +Hopfield completion |
| MCP tools (default) | 29 | 33 | +4 new (mesh set) |
| CLI commands | 21 | 26 | +5 new |
| Dashboard tabs | 17 | 17 | (H-22: Reward / Shadow / EvolutionCost tiles deferred to next cycle — data exposed via API today, see [DASHBOARD-COVERAGE.md](docs/DASHBOARD-COVERAGE.md)) |
| API endpoints | 9 | 16 | +7 new |

Embedding migration happens automatically when you switch modes — no manual steps needed.

### Dashboard

Three new tabs: **Memory Lifecycle** (retention curves, decay stats), **Compression** (storage savings, precision distribution), and **Patterns** (auto-learned soft prompts, consolidation history). Seven new API endpoints power the new views.

### Enable V3.3 Features

All new features default OFF. Zero breaking changes. Opt in when ready:

```bash
# Turn on adaptive memory lifecycle
slm config set lifecycle.enabled true

# Turn on smart compression
slm config set quantization.enabled true

# Turn on cognitive consolidation
slm config set consolidation.cognitive.enabled true

# Turn on pattern learning (soft prompts)
slm config set soft_prompts.enabled true

# Turn on Hopfield retrieval (6th channel)
slm config set retrieval.hopfield.enabled true

# Or enable everything at once
slm config set v33_features.all true
```

**Fully backward compatible.** All existing MCP tools, CLI commands, and configs work unchanged. New tables are created automatically on first run. No migration needed.

</details>

---

## Why SuperLocalMemory?

Every **hosted** AI memory platform — Mem0 Cloud, Zep Cloud, Letta Cloud, EverMemOS Cloud — sends your data to cloud LLMs by default. Their self-hosted variants exist (Mem0 OpenMemory, Letta self-hosted, Graphiti) but require Docker + a separate graph DB or Ollama config, and most still default to OpenAI until you flip env vars. After **August 2, 2026**, any of those cloud paths becomes a compliance problem under the EU AI Act.

SuperLocalMemory V3 takes a different approach: **mathematics instead of cloud compute.** Three techniques from differential geometry, algebraic topology, and stochastic analysis replace the work that other systems need LLMs to do — similarity scoring, contradiction detection, and lifecycle management. The result is an agent memory that ships local-first out of the box — no Docker, no graph DB, no API keys — on CPU.

**The numbers** (evaluated on [LoCoMo](https://arxiv.org/abs/2402.09714), the standard long-conversation memory benchmark). Published numbers as of April 2026:

| System | Score | Config | Cloud LLM required? | Open Source | Source |
|:-------|:-----:|:-------|:-------------------:|:-----------:|:-------|
| EverMemOS | 93.05% | Cloud (proprietary) | Yes | Core only | [evermind.ai](https://evermind.ai/) (Feb 2026) |
| Hindsight (LoComo10) | 92.0% | Cloud | Yes | No | [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) (Apr 2026) |
| Mem0 (token-efficient) | 91.6% | Hybrid (Cohere/OpenAI) | Yes | Partial | [mem0.ai blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) (Apr 16 2026) |
| **SLM V3 Mode C** | **87.7%** | Local + optional LLM | Optional (Ollama OK) | **Yes (AGPL-3.0)** | In-house, repro script in `docs/benchmarks/` |
| Zep v3 Cloud | 85.2% | Cloud | Yes | Community deprecated | [getzep.com](https://www.getzep.com/) |
| **SLM V3 Mode A** | **74.8%** | **Local, CPU-only, zero-LLM** | **No** | **Yes (AGPL-3.0)** | In-house, repro script in `docs/benchmarks/` |
| Mem0 (zero-retrieval-LLM) | 64.2% | Local baseline | No | Partial | Mem0 paper, zero-LLM row |

> **How to read this table.** Scores from different papers use different LoCoMo splits, judge models, and prompt variants. We do NOT claim these numbers are apples-to-apples across rows. The rows we re-ran in-house are marked "In-house"; cited rows link to the vendor's public source and date. Mode A is the only zero-LLM configuration in the list, so the comparison that is apples-to-apples is **Mode A 74.8% vs Mem0 zero-retrieval-LLM 64.2%** (+10.6pp). Mem0's 91.6% and EverMemOS's 93.05% use cloud LLMs; Mode C uses a local LLM (Ollama). BEAM-10M, the emerging successor benchmark, will be added in a future release.

**What Mode A is**: CPU-only, SQLite-only, zero-LLM retrieval pipeline on published LoCoMo questions. To the best of our knowledge it is the only publicly-released local-first memory that clears Mem0's zero-LLM baseline on this benchmark. If another fully-local system hits similar numbers, please open an issue so we can update the table.

Mathematical layers contribute **+12.7 percentage points** on average across 6 conversations (n=832 questions), with up to **+19.9pp on the most challenging dialogues**. This isn't more compute — it's better math.

> **Upgrading from V2 (2.8.6)?** V3 is a complete architectural reinvention — new mathematical engine, new retrieval pipeline, new storage schema. Your existing data is preserved but requires migration. After installing V3, run `slm migrate` to upgrade your data. Read the [Migration Guide](https://github.com/qualixar/superlocalmemory/wiki/Migration-from-V2) before upgrading. Backup is created automatically.

---

## Quick Start

### Install via npm (recommended)

```bash
npm install -g superlocalmemory
slm setup     # Choose mode (A/B/C)
slm doctor    # Verify everything is working
slm warmup    # Pre-download embedding model (~500MB, optional)
```

### Install via pip

```bash
pip install superlocalmemory
```

### Start Saving on LLM Costs (v3.6 Optimize)

```bash
# Wrap your agent — starts proxy + sets environment + launches agent
slm wrap claude
# Your first repeat prompt → CACHE HIT → $0.00 saved
# See savings: slm optimize savings --since 1
```

### Upgrading to v3.6 "Optimize" + v3.5.0 "Scale-Ready"

**Migration is automatic.** Upgrade the package, restart the daemon — all migrations run in the background.

```bash
pip install -U superlocalmemory
slm restart
slm doctor
```

No manual commands. No data loss. Zero downtime.

**What you get after upgrading to v3.6.0:**
- **Cache** — skip repeat LLM calls entirely. Exact-match + vCache-gated semantic. **100% cost saved on hit.**
- **Compress** — shrink prompts 60-95% before sending. Extractive JSON/code (lossless) + LLMLingua-2 prose (opt-in). CCR reversible.
- **Align** — stabilize prompt prefix for native provider KV-cache discounts (Anthropic 90%, OpenAI 50%).
- **Savings dashboard** — live USD/INR/tokens saved displayed in the Optimize tab.

**What you get after upgrading to v3.5.0:**
- **CozoDB on the recall path** — entity_graph channel routes through the CozoDB backend (auto-detected, no config needed). Millions of graph edges indexed and traversed in milliseconds.
- **LanceDB vector backend** — embedding search falls through to LanceDB when available (auto-detected, no config needed). Handles millions of vectors.
- **6-channel recall <1s** — BM25→FTS5 (20ms vs 11s), Hopfield ANN prefilter (0.4s vs 6s), temporal fast-parse (0.25s vs 2.6s). All surfaces (MCP/CLI/Dashboard) use the same daemon path.
- **Core Memory Block** — always-injected pinned facts (auto-derived + explicit pin/unpin via the `core_memory` MCP tool). Pinned facts surface even when the query doesn't match.
- **Context Injection v2** — unified formatter, token-budgeted injection (mode-aware 2K/4K/8K), edge-placement ordering, full-fidelity content (no more 200-char stubs).
- **Score normalization** — all fusion scores mapped to [0, 1] via soft-sigmoid. Monotonic — rank order preserved.
- **Vector store auto-backfill** — facts with embeddings missing from the vector store are indexed on daemon start. No more "only 1/3 of corpus searchable."

### Release History

| Version | Codename | Key Features |
|---|---|---|
| **v3.6.0** | Optimize | **Cache** (skip repeat calls, 100% on hit) · **Compress** (shrink prompts 60-95%) · **Align** (KV-cache stabilization) · `slm optimize\|cache\|compress\|proxy\|wrap` CLI · Live savings dashboard (USD/INR/tokens) · Hot-reload config · Safe defaults · Links: [docs/optimize-overview.md](docs/optimize-overview.md) · [V3.6 Wiki](https://github.com/qualixar/superlocalmemory/wiki/V3.6-Overview) |
| **v3.5.0** | Scale-Ready + Context Injection v2 | CozoDB/LanceDB migration, 6-channel recall <1s, Core Memory Block, BM25→FTS5, context injection v2, score normalization |
| **v3.4.5** | Scale-Ready (foundation) | Tiered storage (active/warm/cold), graph pruning, BackendOrchestrator scaffolding, CozoDB + LanceDB init + migration code (read path wired in v3.5.0) |
| **v3.4.51** | Recency Intelligence | Ebbinghaus decay + FSRS stability, age gate, session context time-awareness |
| **v3.4.22** | Scale-Ready (scaling) | v2 ranking pipeline (LightGBM, bandit ensemble), Hopfield channel, 6-channel parallel execution |
| **v3.3.x** | Foundation | BM25Plus, Fisher-Rao manifold, sqlite-vec, RRF fusion, cross-encoder rerank. 3 published papers (arXiv 2603.02240 / 2603.14588 / 2604.04514) |

### First Use

```bash
slm remember "Alice works at Google as a Staff Engineer"
slm recall "What does Alice do?"
slm status
```

### MCP Integration (Claude, Cursor, Windsurf, VS Code, etc.)

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

33 MCP tools by default (+42 optional behind `SLM_MCP_ALL_TOOLS=1`) + 7 resources. Works with any MCP-compatible client — we ship templated configs for Claude Code, Cursor, Windsurf, VS Code Copilot, Continue, Cody, ChatGPT Desktop, Gemini CLI, JetBrains, Zed, and Antigravity (15 IDE configs in `ide/configs/`). **V3.3: Adaptive lifecycle, smart compression, and pattern learning.**

### Dual Interface: MCP + CLI

SLM works everywhere — from IDEs to CI pipelines to Docker containers. Both the MCP server and the agent-native CLI are first-class, so the same backend serves IDE-side integrations and scripted automations.

| Need | Use | Example |
|------|-----|---------|
| IDE integration | MCP | Auto-configured for 17+ IDEs via `slm connect` |
| Shell scripts | CLI + `--json` | `slm recall "auth" --json \| jq '.data.results[0]'` |
| CI/CD pipelines | CLI + `--json` | `slm remember "deployed v2.1" --json` in GitHub Actions |
| Agent frameworks | CLI + `--json` | OpenClaw, Codex, Goose, nanobot |
| Human use | CLI | `slm recall "auth"` (readable text output) |

**Agent-native JSON output** on every command:

```bash
# Human-readable (default)
slm recall "database schema"
#   1. [0.87] Database uses PostgreSQL 16 on port 5432...

# Agent-native JSON
slm recall "database schema" --json
# {"success": true, "command": "recall", "version": "3.0.22", "data": {"results": [...]}}
```

All `--json` responses follow a consistent envelope with `success`, `command`, `version`, `data`, and `next_actions` for agent guidance.

---

## Smart-hook architecture (v3.4.43)

SLM ships a small set of Claude Code hooks that fire memory operations only
when there's a real signal — not on a timer, not on every keystroke. The
hooks are perf-budgeted (<10ms p99 for the hot path) and fail-open (any
crash → silent exit, never blocks your prompt). Install them with one
command:

```bash
slm hooks install      # wires hooks into ~/.claude/settings.json
slm hooks status       # shows what's installed
slm hooks remove       # cleans up, preserves non-SLM hooks
```

| Hook | Event | When it fires | Why |
|---|---|---|---|
| `slm hook start` | SessionStart | Once at session boot | Injects core memory + recent context + learned patterns. ~80ms. |
| `slm hook user_prompt_rehash` | UserPromptSubmit | Every prompt | Detects re-queries within 60s (negative signal that prior recall didn't satisfy). <10ms hot path. |
| **`slm hook topic_shift`** *(new in 3.4.43)* | UserPromptSubmit | When current prompt shares zero content words with every prompt in a 5-turn sliding window | Surfaces a one-line "consider recall" hint on real topic pivots. Replaces the time-based 15-min nag — event-based, not timer-based. <10ms. |
| **`slm hook before_web`** *(new in 3.4.43)* | PreToolUse on `WebSearch\|WebFetch` | Every web search/fetch | Runs `slm recall <query> --limit 5` and injects local memories as a system-reminder BEFORE the web call. Cost: ~500-800ms per fire, fires 5-20× per session. |
| `slm hook checkpoint` | PostToolUse on `Write\|Edit` | Every file write/edit | Auto-observes file changes into SLM. No periodic nag (removed in v3.4.43). |
| `slm hook post_tool_outcome` | PostToolUse (all tools) | Every tool call | Tracks which recalled facts got used (learning signal). |
| `slm hook stop` | Stop | Session end | Saves rich session summary with git context. |

**What "smart" means here:** the hooks don't interrupt you on a schedule.
They watch for specific events that indicate memory work would add value —
a topic pivot, a web call about to fire, a re-asked question, a file edit.
Otherwise they stay out of your way.

**Observability for the new hooks:**
`topic_shift` writes one TSV line per decision to
`~/.superlocalmemory/logs/topic-shift.log`
(`timestamp | session_hash | current_words_count | window_depth | max_overlap |
fired | prompt_preview`). Disable with `SLM_TOPIC_SHIFT_LOG=0`.

**Upgrading from v3.4.42 or older:** Run `slm hooks install` once after
upgrade to pull in the new wiring. `slm hooks status` will flag the
version mismatch. Merge is idempotent — safe to run twice.

---

## Three Operating Modes

| Mode | What | Cloud? | EU AI Act | Best For |
|:----:|:-----|:------:|:---------:|:---------|
| **A** | Local Guardian | **None** | **Compliant** | Privacy-first, air-gapped, enterprise |
| **B** | Smart Local | Local only (Ollama) | Compliant | Better answers, data stays local |
| **C** | Full Power | Cloud LLM | Partial | Maximum accuracy, research |

```bash
slm mode a   # Zero-cloud (default)
slm mode b   # Local Ollama
slm mode c   # Cloud LLM
```

**Mode A** is, to the best of our knowledge, the only publicly-released agent memory that runs with zero cloud calls while clearing Mem0's published LoCoMo score. All data stays on your device. No API keys. No GPU. Runs on 2 vCPUs + 4GB RAM. If another fully-local system hits similar numbers, please open an issue — we'll update this line.

---

## Architecture

```
Query  ──►  Strategy Classifier  ──►  5 Parallel Channels:
                                       ├── Semantic (Fisher-Rao geodesic distance)
                                       ├── BM25 (keyword matching)
                                       ├── Entity Graph (spreading activation, 3 hops)
                                       ├── Temporal (date-aware retrieval)
                                       └── Hopfield (partial-query completion / associative recall)
                                                    │
                                       RRF Fusion (k=60)
                                                    │
                                       Scene Expansion + Bridge Discovery
                                                    │
                                       Cross-Encoder Reranking
                                                    │
                                       ◄── Top-K Results with channel scores
```

### Mathematical Foundations

Three novel contributions replace cloud LLM dependency with mathematical guarantees:

1. **Fisher-Rao Retrieval Metric** — Similarity scoring derived from the Fisher information structure of diagonal Gaussian families. Graduated ramp from cosine to geodesic distance over the first 10 accesses. To the best of our knowledge, the first public application of information geometry specifically to agent memory retrieval — if prior work exists please open an issue so we can credit it.

2. **Sheaf Cohomology for Consistency** — Algebraic topology detects contradictions by computing coboundary norms on the knowledge graph. We are not aware of a prior production agent-memory system that computes sheaf-cohomology coboundary norms this way; corrections welcome.

3. **Riemannian Langevin Lifecycle** — Memory positions evolve on the Poincare ball via discretized Langevin SDE. Frequently accessed memories stay active; neglected memories self-archive. No hardcoded thresholds.

These three layers collectively yield **+12.7pp average improvement** over the engineering-only baseline, with the Fisher metric alone contributing **+10.8pp** on the hardest conversations.

---

## Benchmarks

Evaluated on [LoCoMo](https://arxiv.org/abs/2402.09714) — 10 multi-session conversations, 1,986 total questions, 4 scored categories.

### Mode A (Zero-Cloud, 10 Conversations, 1,276 Questions)

| Category | Score | vs. Mem0 (64.2%) |
|:---------|:-----:|:-----------------:|
| Single-Hop | 72.0% | +3.0pp |
| Multi-Hop | 70.3% | +8.6pp |
| Temporal | 80.0% | +21.7pp |
| **Open-Domain** | **85.0%** | **+35.0pp** |
| **Aggregate** | **74.8%** | **+10.6pp** |

Mode A achieves **85.0% on open-domain questions — the highest of any system in the evaluation**, including cloud-powered ones.

### Math Layer Impact (6 Conversations, n=832)

| Conversation | With Math | Without | Delta |
|:-------------|:---------:|:-------:|:-----:|
| Easiest | 78.5% | 71.2% | +7.3pp |
| Hardest | 64.2% | 44.3% | **+19.9pp** |
| **Average** | **71.7%** | **58.9%** | **+12.7pp** |

Mathematical layers help most where heuristic methods struggle — the harder the conversation, the bigger the improvement.

### Ablation (What Each Component Contributes)

| Removed | Impact |
|:--------|:------:|
| Cross-encoder reranking | **-30.7pp** |
| Fisher-Rao metric | **-10.8pp** |
| All math layers | **-7.6pp** |
| BM25 channel | **-6.5pp** |
| Sheaf consistency | -1.7pp |
| Entity graph | -1.0pp |

Full ablation details in the [Wiki](https://github.com/qualixar/superlocalmemory/wiki/Benchmarks).

---

## EU AI Act Compliance

The EU AI Act (Regulation 2024/1689) takes full effect **August 2, 2026**. Every AI memory system that sends personal data to cloud LLMs for core operations has a compliance question to answer.

| Requirement | Mode A | Mode B | Mode C |
|:------------|:------:|:------:|:------:|
| Data sovereignty (Art. 10) | **Pass** | **Pass** | Requires DPA |
| Right to erasure (GDPR Art. 17) | **Pass** | **Pass** | **Pass** |
| Transparency (Art. 13) | **Pass** | **Pass** | **Pass** |
| No network calls during memory ops | **Yes** | **Yes** | No |

To the best of our knowledge, **no existing agent memory system addresses EU AI Act compliance**. Modes A and B pass all checks by architectural design — no personal data leaves the device during any memory operation.

Built-in compliance tools: GDPR Article 15/17 export + complete erasure, tamper-proof SHA-256 audit chain, data provenance tracking, ABAC policy enforcement.

---

## Multilingual Embedding Support

**v3.4.24+:** Plug in any OpenAI-compatible embedding endpoint — Ollama, vLLM, LiteLLM, or self-hosted models like `bge-m3`, `multilingual-e5`, `Qwen3-Embedding`. Configure from the dashboard (Settings > Step 3) or `config.json`. SLM's math layer (Fisher-Rao, Sheaf, Langevin) is language-agnostic — swap the embedding model and all 30+ languages work at full retrieval quality. No cloud dependency. No code changes. Your data, your language, your model.

---

## Web Dashboard

```bash
slm dashboard    # Opens at http://localhost:8765
```

**v3.4.4 "Neural Glass":** 17-tab sidebar dashboard with light + dark theme. Knowledge Graph (Sigma.js WebGL, community detection), Health Monitor, Entity Explorer (1,300+ entities), Mesh Peers (P2P agent communication), Ingestion Status (Gmail/Calendar/Transcript management), Privacy blur mode. Always-on daemon with auto-start. 8 mesh MCP tools built-in. Cross-platform: macOS + Windows + Linux. All data stays local.

<!-- UX-M1: link dashboard-coverage so users can find deferred Living Brain Evolution tiles -->
> **Living Brain Evolution visibility:** v3.4.21 ships the reward model, shadow test + online retrain, and evolution cost log via the REST API and `slm status --json`; the dedicated dashboard tiles are deferred to the next cycle. See [docs/DASHBOARD-COVERAGE.md](docs/DASHBOARD-COVERAGE.md) for endpoints and workarounds.

---

<details>
<summary><strong>Active Memory (V3.1) — Memory That Learns</strong> (click to expand)</summary>

Every recall generates learning signals. Over time, the system adapts to your patterns — from baseline (0-19 signals) → rule-based (20+) → ML model (200+, LightGBM trained on YOUR usage). Zero LLM tokens spent. Four mathematical signals computed locally: co-retrieval, confidence lifecycle, channel performance, and entropy gap.

Auto-capture hooks: `slm hooks install` + `slm observe` + `slm session-context`. MCP tools: `session_init`, `observe`, `report_feedback`.

**`session_init` MCP parameters:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `project_path` | string | `""` | Working directory — used to derive search query |
| `query` | string | `""` | Override search query |
| `max_results` | int | `10` | Max memories to return |
| `max_age_days` | int | `30` | Suppress memories older than N days (0 = disabled). Memories with score ≥ 0.70 always surface regardless of age. |

**`slm session-context` CLI flags** (consistent with MCP):
```bash
slm session-context                          # fast path, 30-day window (default)
slm session-context --max-age-days 7         # only last 7 days
slm session-context --max-age-days 0         # no age filter
slm session-context "my query" --full        # full engine path (slow, requires Ollama)
slm session-context --json                   # agent-native JSON output
```

**No competitor learns at zero token cost.**

</details>

---

## Multi-Machine Mesh Coordination (New in v3.4.48)

Run SLM on multiple machines (M4 + M5) and have your agents coordinate as one team without any disruption.

### Setup

**M4 (broker):**
```bash
export SLM_MESH_HOST=192.168.1.100
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init  # Starts SLM at http://192.168.1.100:8765
```

**M5 (client):**
```bash
export SLM_MESH_PEER_URL=http://192.168.1.100:8765
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init  # Syncs M4's agents every 30s, proxies messages to M4
```

### How It Works

- **HTTP-based sync** — M5 queries M4's `/mesh/peers` endpoint every 30 seconds
- **Message proxying** — When M5's agent sends a message to an M4 agent, it's routed automatically
- **mDNS discovery (optional)** — M5 can auto-discover M4 on the LAN via `_slm-mesh._tcp` (enable with `SLM_MESH_DISCOVERY=on`, default)
- **Graceful fallback** — Network errors logged but don't crash; offline agents queue messages for delivery
- **Shared secret** — `SLM_MESH_SHARED_SECRET` gates remote peer discovery (required for remote mode)

### Environment Variables

| Variable | Default | Purpose |
|:---------|:--------|:--------|
| `SLM_MESH_HOST` | `127.0.0.1` | Host this SLM listens on (set to IP for remote) |
| `SLM_MESH_PEER_URL` | unset | Full URL of remote SLM (e.g., `http://192.168.1.100:8765`) |
| `SLM_MESH_SHARED_SECRET` | unset | Auth secret (required when remote) |
| `SLM_MESH_DISCOVERY` | `on` | mDNS discovery (`on`/`off`) |
| `SLM_MESH_WS_PORT` | `7900` | WebSocket port for mesh (internal use) |

### Dependencies

- `zeroconf>=0.140` (new in v3.4.48, optional, pure Python, auto-installed)
- `httpx==0.28.1` (already in core deps)
- No Docker. No external broker. Works on WiFi + LAN.

### MCP Tools

All 8 mesh tools work seamlessly across machines:

| Tool | Description |
|:-----|:------------|
| `mesh_peers` | List local + remote peers merged |
| `mesh_send` | Send message to any peer (local or remote) |
| `mesh_broadcast` | Send to all agents (across machines) |
| `mesh_project` | Send to all agents in a project (across machines) |
| `mesh_inbox` | Get messages for this agent |
| `mesh_pending` | Get offline messages (broadcast/project) |
| `mesh_state` | Get/set shared state (replicated) |
| `mesh_lock` | Acquire/release distributed file locks |

---

## Features

### LLM Cost Optimization (v3.6 Optimize)
- **Exact Cache** — byte-identical repeat calls served from local SQLite. SHA-256 key derivation, stampede shield, tag-based invalidation. **100% cost saved on hit** (input + output tokens).
- **Semantic Cache** (opt-in) — vCache-powered learned thresholds with SAFE-CACHE centroid defense. Near-duplicate queries served within error bound. CacheAttack 86% hijack class blocked.
- **Extractive Compression** — structure-preserving compression for JSON, code (AST-aware: Python/JS/Go/Rust/Java/C++), and tool outputs. **60-95% fewer input tokens**, zero accuracy regression.
- **LLMLingua-2 Prose** (opt-in) — extractive prose summarization for open-ended chat. Safety-warned before enable.
- **CCR (Compressed Context Retrieval)** — pre-compression originals stored for byte-exact reversal under UUID. Every compressed block recoverable.
- **CacheAligner** — detects volatile tokens (UUIDs, timestamps, JWTs) in system prompts. Maximizes native provider prefix-cache discounts (Anthropic 90%, OpenAI 50%).
- **Interception Proxy** — HTTP proxy on port 8765 serving Anthropic, OpenAI, and Gemini surfaces. Zero-code integration — just set `base_url`.
- **Agent Wrapping** — `slm wrap claude` — one command starts proxy + sets environment + launches agent. 10 supported agents.
- **Savings Dashboard** — live USD/INR/tokens saved, hit rate, compression ratio, cache size. CLI + UI.
- **Hot-Reload Config** — UI/CLI writes `optimize.json`; daemon reloads in 2 seconds. No restart.
- **Fail-open** — any cache/compress/proxy error passes through. Your calls never break.
- **Data isolation** — separate `llmcache.db` with AES-256-GCM encryption. Never touches `memory.db`.

### Retrieval
- 5-channel hybrid: Semantic (Fisher-Rao) + BM25 + Entity Graph + Temporal + Hopfield (associative / partial-query completion)
- RRF fusion + cross-encoder reranking
- Agentic sufficiency verification (auto-retry on weak results)
- Adaptive ranking with LightGBM (learns from usage)
- Hopfield completion for vague/partial queries

### Intelligence
- 11-step ingestion pipeline (entity resolution, fact extraction, emotional tagging, scene building)
- Automatic contradiction detection via sheaf cohomology
- Adaptive memory lifecycle — memories strengthen with use, fade when neglected
- Smart compression — embedding precision adapts to memory importance (up to 32x savings)
- Cognitive consolidation — automatic pattern extraction from related memories
- Auto-learned soft prompts injected into agent context
- Behavioral pattern detection and outcome tracking

### Skill Evolution
- **Per-skill performance tracking** — tracks which skills succeed and fail across sessions (zero-LLM, always on)
- **Evolution engine** — 3-trigger system with blind verification. Off by default — enable via `slm config set evolution.enabled true`
- **MCP tools** — `evolve_skill`, `skill_health`, `skill_lineage` for programmatic access
- **Lineage DAG** — visual evolution history in the dashboard
- **CLI config** — `slm config get/set` for all evolution settings
- **Post-session triggers** — automatic analysis on session end via Stop hook
- **[ECC](https://github.com/affaan-m/everything-claude-code) integration** — optional enhanced observations via `slm ingest --source ecc`

### Tiered Storage & Scaling
- **4-tier lifecycle** — active, warm, cold, archived with automatic promotion/demotion
- **Deep recall** — archived facts searchable at reduced weight
- **Graph pruning** — automatic cleanup of orphan edges, self-loops, duplicates
- **Fact consolidation** — clusters related facts into consolidated summaries

### Trust & Security
- Bayesian Beta-distribution trust scoring (per-agent, per-fact)
- Trust gates (block low-trust agents from writing/deleting)
- ABAC (Attribute-Based Access Control) with DB-persisted policies
- Tamper-proof hash-chain audit trail (SHA-256 linked entries)

### Infrastructure
- 17-tab web dashboard with real-time visualization
- 17+ IDE integrations (Claude, Cursor, Windsurf, VS Code, JetBrains, Zed, etc.)
- 33 default MCP tools (+42 optional via `SLM_MCP_ALL_TOOLS=1`) + 7 MCP resources
- Profile isolation (independent memory spaces)
- 2,900+ tests, AGPL v3, cross-platform (Mac/Linux/Windows)
- CPU-only — no GPU required
- Automatic orphaned process cleanup

---

## CLI Reference

| Command | What It Does |
|:--------|:-------------|
| `slm optimize status` | Show all Optimize settings (cache, compress, proxy, config version) |
| `slm optimize on\|off` | Enable/disable all Optimize features (hot-reload, no restart) |
| `slm optimize savings [--since N] [--provider P] [--json]` | Token/cost savings report — live USD/INR |
| `slm cache status\|clear\|invalidate\|ttl\|semantic` | Cache sub-control — exact + semantic tiers, TTL management |
| `slm compress status\|mode\|code\|prose\|ccr\|align` | Compression control — per-channel toggle, safe/aggressive mode |
| `slm proxy [--port] [--provider] [--no-compress] [--semantic]` | Start interception proxy (port 8765) |
| `slm wrap <agent> [options]` | Proxy-activate an agent — one command to start saving |
| `slm help-optimize [topic]` | Full developer reference + per-agent setup recipes |
| `slm remember "..."` | Store a memory |
| `slm recall "..."` | Search memories |
| `slm forget "..."` | Delete matching memories |
| `slm trace "..."` | Recall with per-channel score breakdown |
| `slm status` | System status |
| `slm health` | Math layer health (Fisher, Sheaf, Langevin) |
| `slm doctor` | Pre-flight check (deps, worker, Ollama, database) |
| `slm mode a/b/c` | Switch operating mode |
| `slm setup` | Interactive first-time wizard |
| `slm warmup` | Pre-download embedding model |
| `slm migrate` | V2 to V3 migration |
| `slm dashboard` | Launch 17-tab web dashboard |
| `slm mcp` | Start MCP server (for IDE integration) |
| `slm connect` | Configure IDE integrations |
| `slm hooks install` | Wire auto-memory into Claude Code hooks |
| `slm profile list/create/switch` | Profile management |
| `slm decay` | Run memory lifecycle review |
| `slm session-context [query]` | Print session context (for hooks). Flags: `--max-age-days N` (default 30), `--full`, `--json` |
| `slm quantize` | Run smart compression cycle |
| `slm consolidate --cognitive` | Extract patterns from memory clusters |
| `slm soft-prompts` | View auto-learned patterns |
| `slm reap` | Clean orphaned SLM processes |

---

## Research Papers

SuperLocalMemory is backed by three published research papers (arXiv preprints + Zenodo DOIs) covering trust, information geometry, and cognitive memory architecture. These are preprints — not conference-accepted or journal-published yet.

### Paper 3: The Living Brain (V3.3)
> **SuperLocalMemory V3.3: The Living Brain — Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2604.04514](https://arxiv.org/abs/2604.04514) · [Zenodo DOI: 10.5281/zenodo.19435120](https://zenodo.org/records/19435120)

### Paper 2: Information-Geometric Foundations (V3)
> **SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) · [Zenodo DOI: 10.5281/zenodo.19038659](https://zenodo.org/records/19038659)

### Paper 1: Trust & Behavioral Foundations (V2)
> **SuperLocalMemory: A Structured Local Memory Architecture for Persistent AI Agent Context**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) · [Zenodo DOI: 10.5281/zenodo.18709670](https://zenodo.org/records/18709670)

### Cite This Work

```bibtex
@article{bhardwaj2026slmv33,
  title={SuperLocalMemory V3.3: The Living Brain — Biologically-Inspired
         Forgetting, Cognitive Quantization, and Multi-Channel Retrieval
         for Zero-LLM Agent Memory Systems},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2604.04514},
  year={2026},
  url={https://arxiv.org/abs/2604.04514}
}

@article{bhardwaj2026slmv3,
  title={Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.14588},
  year={2026}
}

@article{bhardwaj2026slm,
  title={A Structured Local Memory Architecture for Persistent AI Agent Context},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.02240},
  year={2026}
}
```

---

## Prerequisites

| Requirement | Version | Why |
|:-----------|:--------|:----|
| **Node.js** | 14+ | npm package manager |
| **Python** | 3.11+ | V3 engine runtime |

All Python dependencies install automatically during `npm install` — core math, dashboard server, learning engine, and performance optimizations. If anything fails, the installer shows exact fix commands. Run `slm doctor` after install to verify everything works. BM25 keyword search works even without embeddings — you're never fully blocked.

| Component | Size | When |
|:----------|:-----|:-----|
| Core libraries (numpy, scipy, networkx) | ~50MB | During install |
| Dashboard & MCP server (fastapi, uvicorn) | ~20MB | During install |
| Learning engine (lightgbm) | ~10MB | During install |
| Search engine (sentence-transformers, torch) | ~200MB | During install |
| Embedding model (nomic-embed-text-v1.5, 768d) | ~500MB | First use or `slm warmup` |
| **Mode B** requires [Ollama](https://ollama.com) + a model (`ollama pull llama3.2`) | ~2GB | Manual |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. [Wiki](https://github.com/qualixar/superlocalmemory/wiki) for detailed documentation.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

For commercial licensing (closed-source, proprietary, or hosted use), see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or contact varun.pratap.bhardwaj@gmail.com.

Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar.

## Attribution

Part of [Qualixar](https://qualixar.com) · Author: [Varun Pratap Bhardwaj](https://varunpratap.com)

### Acknowledgments

- **[Everything Claude Code (ECC)](https://github.com/affaan-m/everything-claude-code)** — SLM's skill observation patterns were inspired by ECC's continuous learning architecture. SLM supports direct ingestion of ECC observations via `slm ingest --source ecc`, giving ECC users richer skill performance tracking. We recommend ECC for Claude Code users who want the deepest learning experience alongside SLM.
- **[HKUDS/OpenSpace](https://github.com/HKUDS/OpenSpace)** — The skill evolution research in SLM draws from the EvoSkills co-evolutionary verification concepts (arXiv:2604.01687). We adopted their 3-trigger evolution system and anti-loop guard patterns.

---

<p align="center">
  <sub>Built with mathematical rigor. Not in the race — here to help everyone build better AI memory systems.</sub>
</p>

---

## ⭐ Support This Project

If this project solves a real problem for you, **please star the repo** — it helps other developers discover Qualixar and signals that the AI agent reliability community is growing. Every star matters.

[![Star History Chart](https://api.star-history.com/svg?repos=qualixar/superlocalmemory&type=Date)](https://star-history.com/#qualixar/superlocalmemory&Date)

---

## Part of the Qualixar AI Agent Reliability Platform

Qualixar is building the open-source infrastructure for AI agent reliability engineering. Seven products, seven research papers (published as arXiv preprints + Zenodo archives), one coherent platform. Each tool solves one reliability pillar:

| Product | Purpose | Install | Paper |
|---------|---------|---------|-------|
| **[SuperLocalMemory](https://github.com/qualixar/superlocalmemory)** | Persistent memory + learning for AI agents | `npx superlocalmemory` | [arXiv:2604.04514](https://arxiv.org/abs/2604.04514) |
| **[Qualixar OS](https://github.com/qualixar/qualixar-os)** | Universal agent runtime (13 execution topologies) | `npx qualixar-os` | [arXiv:2604.06392](https://arxiv.org/abs/2604.06392) |
| **[SLM Mesh](https://github.com/qualixar/slm-mesh)** | P2P coordination across AI agent sessions | `npm i slm-mesh` | — |
| **[SLM MCP Hub](https://github.com/qualixar/slm-mcp-hub)** | Federate 430+ MCP tools through one gateway | `pip install slm-mcp-hub` | — |
| **[AgentAssay](https://github.com/qualixar/agentassay)** | Token-efficient AI agent testing | `pip install agentassay` | [arXiv:2603.02601](https://arxiv.org/abs/2603.02601) |
| **[AgentAssert](https://github.com/qualixar/agentassert-abc)** | Behavioral contracts + drift detection |  `pip install agentassert-abc` | [arXiv:2602.22302](https://arxiv.org/abs/2602.22302) |
| **[SkillFortify](https://github.com/qualixar/skillfortify)** | Formal verification for AI agent skills | `pip install skillfortify` | [arXiv:2603.00195](https://arxiv.org/abs/2603.00195) |

**Zero cloud dependency. Local-first. EU AI Act compliant.**

Start here → **[qualixar.com](https://qualixar.com)** · [All papers on Qualixar HuggingFace](https://huggingface.co/Qualixar)

---
