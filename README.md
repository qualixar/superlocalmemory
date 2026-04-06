<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory V3.3</h1>
<p align="center"><strong>Every other AI forgets. Yours won't.</strong><br/><em>Infinite memory for Claude Code, Cursor, Windsurf & 17+ AI tools.</em></p>
<p align="center"><code>v3.3.15</code> — Install once. Every session remembers the last. Automatically.</p>

<p align="center">
  <code>+16pp vs Mem0 (zero cloud)</code> &nbsp;·&nbsp; <code>85% Open-Domain (best of any system)</code> &nbsp;·&nbsp; <code>EU AI Act Ready</code>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14588"><img src="https://img.shields.io/badge/arXiv-2603.14588-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="https://pypi.org/project/superlocalmemory/"><img src="https://img.shields.io/pypi/v/superlocalmemory?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/superlocalmemory"><img src="https://img.shields.io/npm/v/superlocalmemory?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Elastic%202.0-blue?style=for-the-badge" alt="Elastic License 2.0"/></a>
  <a href="#eu-ai-act-compliance"><img src="https://img.shields.io/badge/EU_AI_Act-Compliant-brightgreen?style=for-the-badge" alt="EU AI Act"/></a>
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Web-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/MCP-Native-blue?style=for-the-badge" alt="MCP Native"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/CLI-Agent--Native-green?style=for-the-badge" alt="CLI Agent-Native"/></a>
</p>

---

## Why SuperLocalMemory?

Every major AI memory system — Mem0, Zep, Letta, EverMemOS — sends your data to cloud LLMs for core operations. That means latency on every query, cost on every interaction, and after **August 2, 2026**, a compliance problem under the EU AI Act.

SuperLocalMemory V3 takes a different approach: **mathematics instead of cloud compute.** Three techniques from differential geometry, algebraic topology, and stochastic analysis replace the work that other systems need LLMs to do — similarity scoring, contradiction detection, and lifecycle management. The result is an agent memory that runs entirely on your machine, on CPU, with no API keys, and still outperforms funded alternatives.

**The numbers** (evaluated on [LoCoMo](https://arxiv.org/abs/2402.09714), the standard long-conversation memory benchmark):

| System | Score | Cloud Required | Open Source | Funding |
|:-------|:-----:|:--------------:|:-----------:|:-------:|
| EverMemOS | 92.3% | Yes | No | — |
| Hindsight | 89.6% | Yes | No | — |
| **SLM V3 Mode C** | **87.7%** | Optional | **Yes (EL2)** | $0 |
| Zep v3 | 85.2% | Yes | Deprecated | $35M |
| **SLM V3 Mode A** | **74.8%** | **No** | **Yes (EL2)** | $0 |
| Mem0 | 64.2% | Yes | Partial | $24M |

Mode A scores **74.8% with zero cloud dependency** — outperforming Mem0 by 16 percentage points without a single API call. On open-domain questions, Mode A scores **85.0% — the highest of any system in the evaluation**, including cloud-powered ones. Mode C reaches **87.7%**, matching enterprise cloud systems.

Mathematical layers contribute **+12.7 percentage points** on average across 6 conversations (n=832 questions), with up to **+19.9pp on the most challenging dialogues**. This isn't more compute — it's better math.

> **Upgrading from V2 (2.8.6)?** V3 is a complete architectural reinvention — new mathematical engine, new retrieval pipeline, new storage schema. Your existing data is preserved but requires migration. After installing V3, run `slm migrate` to upgrade your data. Read the [Migration Guide](https://github.com/qualixar/superlocalmemory/wiki/Migration-from-V2) before upgrading. Backup is created automatically.

---

## What's New in V3.3 — The Living Brain Evolves

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
| MCP tools | 29 | 35 | +6 new |
| CLI commands | 21 | 26 | +5 new |
| Dashboard tabs | 20 | 23 | +3 new |
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

---

## What's New in V3.3.7–3.3.15 — Stability & Performance

> Nine maintenance releases fixing memory safety, subprocess leaks, vector quantization, and CI reliability — plus TurboQuant for better compression.

### Fixes

| Version | Fix |
|:--------|:----|
| **3.3.15** | Windows CI `KeyboardInterrupt` handling — prevents false interrupts during process cleanup |
| **3.3.14** | Memory safety for entity lookups + targeted fact loading |
| **3.3.13** | ONNX reranker reliability, L-12-v2 embedding model upgrade, 6 additional fixes |
| **3.3.12** | Retrieval quality improvements, MCP tool fixes, memory stability |
| **3.3.11** | Store verbatim content alongside extracted facts |
| **3.3.9** | Codebook range now covers full unit-sphere coordinates |
| **3.3.8** | TurboQuant near-optimal vector quantization |
| **3.3.7** | Prevent orphaned subprocess workers from leaking memory |
| **3.3.6** | Zero-friction hooks with auto-install |

### Performance

| Fix | Impact |
|:----|:-------|
| In-memory adjacency cache for entity graph channel | **31s → 500ms** |
| Fisher-Rao perf test threshold relaxed for CI | 3x → 5x margin |

---

<details>
<summary><strong>What's New in V3.2 — The Living Brain</strong> (click to expand)</summary>

100x faster recall (<10ms at 10K facts), automatic memory surfacing, associative retrieval (5th channel), temporal intelligence with bi-temporal validity, sleep-time consolidation, and core memory blocks. All features default OFF, zero breaking changes.

| Metric | V3.0 | V3.2 | Change |
|:-------|:----:|:----:|:------:|
| Recall latency (10K facts) | ~500ms | <10ms | **100x faster** |
| Retrieval channels | 4 | 5 | +spreading activation |
| MCP tools | 24 | 29 | +5 new |
| DB tables | 9 | 18 | +9 new |

Enable with `slm config set v32_features.all true`. See the [V3.2 Overview](https://github.com/qualixar/superlocalmemory/wiki/V3.2-Overview) wiki page for details.

</details>

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

35 MCP tools + 7 resources available. Works with Claude Code, Cursor, Windsurf, VS Code Copilot, Continue, Cody, ChatGPT Desktop, Gemini CLI, JetBrains, Zed, and 17+ AI tools. **V3.3: Adaptive lifecycle, smart compression, and pattern learning.**

### Dual Interface: MCP + CLI

SLM works everywhere -- from IDEs to CI pipelines to Docker containers. The only AI memory system with both MCP and agent-native CLI.

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

**Mode A** is the only agent memory that operates with **zero cloud dependency** while achieving competitive retrieval accuracy on a standard benchmark. All data stays on your device. No API keys. No GPU. Runs on 2 vCPUs + 4GB RAM.

---

## Architecture

```
Query  ──►  Strategy Classifier  ──►  6 Parallel Channels:
                                       ├── Semantic (Fisher-Rao geodesic distance)
                                       ├── BM25 (keyword matching)
                                       ├── Entity Graph (spreading activation, 3 hops)
                                       ├── Temporal (date-aware retrieval)
                                       ├── Associative (multi-hop spreading activation)
                                       └── Hopfield (partial query completion)
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

1. **Fisher-Rao Retrieval Metric** — Similarity scoring derived from the Fisher information structure of diagonal Gaussian families. Graduated ramp from cosine to geodesic distance over the first 10 accesses. The first application of information geometry to agent memory retrieval.

2. **Sheaf Cohomology for Consistency** — Algebraic topology detects contradictions by computing coboundary norms on the knowledge graph. The first algebraic guarantee for contradiction detection in agent memory.

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

## Web Dashboard

```bash
slm dashboard    # Opens at http://localhost:8765
```

<details open>
<summary><strong>Dashboard Screenshots</strong> (click to collapse)</summary>
<p align="center"><img src="docs/screenshots/01-dashboard-main.png" alt="Dashboard" width="600"/></p>
<p align="center">
<img src="docs/screenshots/02-knowledge-graph.png" alt="Graph" width="190"/>
<img src="docs/screenshots/03-math-health.png" alt="Math" width="190"/>
<img src="docs/screenshots/05-trust-dashboard.png" alt="Trust" width="190"/>
</p>
<p align="center">
<img src="docs/screenshots/04-recall-lab.png" alt="Recall" width="190"/>
<img src="docs/screenshots/06-settings.png" alt="Settings" width="190"/>
<img src="docs/screenshots/07-memories-blurred.png" alt="Memories" width="190"/>
</p>
</details>

23 tabs: Dashboard, Recall Lab, Knowledge Graph, Memories, Trust Scores, Math Health, Compliance, Learning, IDE Connections, Settings, Memory Lifecycle, Compression, Patterns, and more. Runs locally — no data leaves your machine.

---

<details>
<summary><strong>Active Memory (V3.1) — Memory That Learns</strong> (click to expand)</summary>

Every recall generates learning signals. Over time, the system adapts to your patterns — from baseline (0-19 signals) → rule-based (20+) → ML model (200+, LightGBM trained on YOUR usage). Zero LLM tokens spent. Four mathematical signals computed locally: co-retrieval, confidence lifecycle, channel performance, and entropy gap.

Auto-capture hooks: `slm hooks install` + `slm observe` + `slm session-context`. MCP tools: `session_init`, `observe`, `report_feedback`.

**No competitor learns at zero token cost.**

</details>

---

## Features

### Retrieval
- 6-channel hybrid: Semantic (Fisher-Rao) + BM25 + Entity Graph + Temporal + Associative + Hopfield
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

### Trust & Security
- Bayesian Beta-distribution trust scoring (per-agent, per-fact)
- Trust gates (block low-trust agents from writing/deleting)
- ABAC (Attribute-Based Access Control) with DB-persisted policies
- Tamper-proof hash-chain audit trail (SHA-256 linked entries)

### Infrastructure
- 23-tab web dashboard with real-time visualization
- 17+ IDE integrations (Claude, Cursor, Windsurf, VS Code, JetBrains, Zed, etc.)
- 35 MCP tools + 7 MCP resources
- Profile isolation (independent memory spaces)
- 1400+ tests, Elastic License 2.0, cross-platform (Mac/Linux/Windows)
- CPU-only — no GPU required
- Automatic orphaned process cleanup

---

## CLI Reference

| Command | What It Does |
|:--------|:-------------|
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
| `slm quantize` | Run smart compression cycle |
| `slm consolidate --cognitive` | Extract patterns from memory clusters |
| `slm soft-prompts` | View auto-learned patterns |
| `slm reap` | Clean orphaned SLM processes |

---

## Research Papers

### V3: Information-Geometric Foundations
> **SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) · [Zenodo DOI: 10.5281/zenodo.19038659](https://zenodo.org/records/19038659)

### V2: Architecture & Engineering
> **SuperLocalMemory: A Structured Local Memory Architecture for Persistent AI Agent Context**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) · [Zenodo DOI: 10.5281/zenodo.18709670](https://zenodo.org/records/18709670)

### Cite This Work

```bibtex
@article{bhardwaj2026slmv3,
  title={Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.14588},
  year={2026},
  url={https://arxiv.org/abs/2603.14588}
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

Elastic License 2.0. See [LICENSE](LICENSE). Previously MIT — changed April 2026 to protect research IP while keeping the project open for use and contributions.

## Attribution

Part of [Qualixar](https://qualixar.com) · Author: [Varun Pratap Bhardwaj](https://varunpratap.com)

---

<p align="center">
  <sub>Built with mathematical rigor. Not in the race — here to help everyone build better AI memory systems.</sub>
</p>
