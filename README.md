<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory V3.4</h1>
<p align="center"><strong>Every other AI forgets. Yours won't.</strong><br/><em>Infinite memory for Claude Code, Cursor, Windsurf, and any MCP-compatible AI client.</em></p>
<p align="center"><code>v3.4.21</code> — Install once. Every session remembers the last. Automatically.</p>
<p align="center"><strong>Backed by 3 published research papers</strong> (arXiv preprints + Zenodo-archived) · <a href="https://arxiv.org/abs/2603.02240">arXiv:2603.02240</a> · <a href="https://arxiv.org/abs/2603.14588">arXiv:2603.14588</a> · <a href="https://arxiv.org/abs/2604.04514">arXiv:2604.04514</a></p>

<p align="center">
  <code>+10.6pp vs Mem0 zero-LLM</code> &nbsp;·&nbsp; <code>85% Open-Domain (best zero-LLM score)</code> &nbsp;·&nbsp; <code>EU AI Act Ready</code>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14588"><img src="https://img.shields.io/badge/arXiv-2603.14588-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="https://pypi.org/project/superlocalmemory/"><img src="https://img.shields.io/pypi/v/superlocalmemory?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/superlocalmemory"><img src="https://img.shields.io/npm/v/superlocalmemory?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=for-the-badge" alt="AGPL v3"/></a>
  <a href="#eu-ai-act-compliance"><img src="https://img.shields.io/badge/EU_AI_Act-Design_Compliant-brightgreen?style=for-the-badge" alt="EU AI Act Design Compliant"/></a>
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Web-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/MCP-Native-blue?style=for-the-badge" alt="MCP Native"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/CLI-Agent--Native-green?style=for-the-badge" alt="CLI Agent-Native"/></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/c3b54a1d-f62a-4ea7-bba7-900435e7b3ab" width="800" autoplay loop muted playsinline></video>
</p>

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

**No competitor learns at zero token cost.**

</details>

---

## Features

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
