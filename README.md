<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory V3</h1>
<p align="center"><strong>Information-Geometric Agent Memory with Mathematical Guarantees</strong></p>

<p align="center">
  The first agent memory system with mathematically grounded retrieval, lifecycle management, and consistency verification. Four-channel hybrid retrieval. Three operating modes. EU AI Act compliant.
</p>

<p align="center">
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Website-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="https://arxiv.org/abs/2603.02240"><img src="https://img.shields.io/badge/arXiv-2603.02240-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="https://zenodo.org/records/19038659"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19038659-blue?style=for-the-badge&logo=doi&logoColor=white" alt="V3 DOI"/></a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
  <a href="#three-operating-modes"><img src="https://img.shields.io/badge/EU_AI_Act-Compliant-brightgreen?style=flat-square" alt="EU AI Act"/></a>
  <a href="#"><img src="https://img.shields.io/badge/tests-1400+-brightgreen?style=flat-square" alt="1400+ Tests"/></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-Mac_|_Linux_|_Windows-blue?style=flat-square" alt="Cross Platform"/></a>
  <a href="https://github.com/qualixar/superlocalmemory/wiki"><img src="https://img.shields.io/badge/Wiki-Documentation-blue?style=flat-square" alt="Wiki"/></a>
</p>

---

## What is SuperLocalMemory?

SuperLocalMemory gives AI assistants persistent, structured memory that survives across sessions. Unlike simple vector stores, V3 uses **information geometry** to provide mathematically grounded retrieval, automatic contradiction detection, and self-organizing memory lifecycle management.

**Works with:** Claude, Cursor, Windsurf, VS Code Copilot, Continue, Cody, ChatGPT Desktop, Gemini CLI, JetBrains, Zed, and 17+ AI tools via MCP.

> **Upgrading from V2 (2.8.6)?** V3 is a complete architectural reinvention — new mathematical engine, new retrieval pipeline, new storage schema. Your existing data is preserved but requires migration. After installing V3, run `slm migrate` to upgrade your data. Read the [Migration Guide](docs/migration-from-v2.md) before upgrading. Backup is created automatically.

### Key Results

| Metric | Score | Context |
|:-------|:-----:|:--------|
| LoCoMo (Mode C, full power) | **87.7%** | On conv-30, 81 scored questions |
| LoCoMo (Mode A, zero-LLM) | **62.3%** | Highest zero-LLM score. No cloud dependency. |
| Math layer improvement | **+12.7pp** | Average gain from mathematical foundations |
| Multi-hop improvement | **+12pp** | 50% vs 38% (math on vs off) |

---

## Three Operating Modes

| Mode | What | LLM Required? | EU AI Act | Best For |
|:----:|:-----|:-------------:|:---------:|:---------|
| **A** | Local Guardian | No | Compliant | Privacy-first, air-gapped, enterprise |
| **B** | Smart Local | Local only (Ollama) | Compliant | Enhanced quality, data stays local |
| **C** | Full Power | Cloud (optional) | Partial | Maximum accuracy, research |

**Mode A** is the only agent memory that operates with **zero cloud dependency** while achieving competitive retrieval accuracy. All data stays on your device. No API keys required.

---

## Architecture

```
Query  ──►  Strategy Classifier  ──►  4 Parallel Channels:
                                       ├── Semantic (Fisher-Rao graduated similarity)
                                       ├── BM25 (keyword matching, k1=1.2, b=0.75)
                                       ├── Entity Graph (spreading activation, 3 hops)
                                       └── Temporal (date-aware retrieval)
                                                    │
                                       RRF Fusion (k=60)
                                                    │
                                       Scene Expansion + Bridge Discovery
                                                    │
                                       Cross-Encoder Reranking
                                                    │
                                       ◄── Top-K Results with channel scores
```

### Mathematical Foundations (Novel Contributions)

1. **Fisher-Rao Retrieval Metric** — Similarity scoring derived from the Fisher information structure of diagonal Gaussian families. Graduated ramp from cosine to Fisher-information-weighted scoring over the first 10 accesses per memory.

2. **Sheaf Cohomology for Consistency** — Algebraic topology detects contradictions between facts by computing coboundary norms on the knowledge graph. Non-trivial restriction maps amplify disagreements along discriminative subspaces.

3. **Riemannian Langevin Lifecycle** — Memory positions evolve on the Poincare ball via a discretized Langevin SDE. Frequently accessed memories stay near the origin (ACTIVE); neglected memories diffuse toward the boundary (ARCHIVED). The potential is modulated by access frequency, age, and importance.

---

## Prerequisites

| Requirement | Version | Why |
|:-----------|:--------|:----|
| **Node.js** | 14+ | npm package manager |
| **Python** | 3.11+ | V3 engine runtime |
| **pip** | Latest | Python dependency installer |

> All Python dependencies are installed automatically during `npm install`. You don't need to run pip manually. If any dependency fails, the installer shows clear instructions.

### What Gets Installed Automatically

| Component | Size | When |
|:----------|:-----|:-----|
| Core math libraries (numpy, scipy, networkx) | ~50MB | During `npm install` |
| Search engine (sentence-transformers, einops, torch) | ~200MB | During `npm install` |
| Embedding model (nomic-ai/nomic-embed-text-v1.5) | ~500MB | On first use OR `slm warmup` |

**If any dependency fails during install**, the installer prints the exact `pip install` command to fix it. BM25 keyword search works even without embeddings — you're never fully blocked.

---

## Quick Start

### Install via npm (recommended — one command, everything included)

```bash
npm install -g superlocalmemory
```

This single command:
- Installs the V3 engine and CLI
- Auto-installs all Python dependencies (numpy, scipy, networkx, sentence-transformers, einops, torch, etc.)
- Creates the data directory at `~/.superlocalmemory/`
- Detects and guides V2 migration if applicable

Then configure and pre-download the embedding model:
```bash
slm setup     # Choose mode, configure provider
slm warmup    # Pre-download embedding model (~500MB, optional)
```

> **First time?** If you skip `slm warmup`, the model downloads automatically on first `slm remember` or `slm recall`. Either way works.

### Install via pip

```bash
pip install superlocalmemory
# or with all features:
pip install "superlocalmemory[full]"
```

### First Use

```bash
# Store a memory
slm remember "Alice works at Google as a Staff Engineer"

# Recall
slm recall "What does Alice do?"

# Check status
slm status

# Switch modes
slm mode a   # Zero-LLM (default)
slm mode b   # Local Ollama
slm mode c   # Full power
```

### MCP Integration (Claude, Cursor, etc.)

Add to your IDE's MCP config:

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

24 MCP tools available: `remember`, `recall`, `search`, `fetch`, `list_recent`, `get_status`, `build_graph`, `switch_profile`, `health`, `consistency_check`, `recall_trace`, and more.

### Web Dashboard (17 tabs)

```bash
slm dashboard    # Opens at http://localhost:8765
```

<p align="center">
  <img src="docs/screenshots/01-dashboard-main.png" alt="V3 Dashboard — Mode A, 528 memories, Math Layers Active" width="800"/>
</p>

<details>
<summary><strong>More screenshots</strong> (click to expand)</summary>

| Knowledge Graph | Math Health | Recall Lab |
|:---:|:---:|:---:|
| <img src="docs/screenshots/02-knowledge-graph.png" alt="Knowledge Graph" width="280"/> | <img src="docs/screenshots/03-math-health.png" alt="Math Health" width="280"/> | <img src="docs/screenshots/04-recall-lab.png" alt="Recall Lab" width="280"/> |

| Trust Dashboard | Settings | Memory Browser |
|:---:|:---:|:---:|
| <img src="docs/screenshots/05-trust-dashboard.png" alt="Trust Dashboard" width="280"/> | <img src="docs/screenshots/06-settings.png" alt="Settings" width="280"/> | <img src="docs/screenshots/07-memories-blurred.png" alt="Memories" width="280"/> |

</details>

The V3 dashboard provides real-time visibility into your memory system:

- **Dashboard** — Mode switcher, health score, quick store/recall
- **Recall Lab** — Search with per-channel score breakdown (Semantic, BM25, Entity, Temporal)
- **Knowledge Graph** — Interactive entity relationship visualization
- **Memories** — Browse, search, and manage stored memories
- **Trust Dashboard** — Bayesian trust scores per agent with Beta distribution visualization
- **Math Health** — Fisher-Rao confidence, Sheaf consistency, Langevin lifecycle state
- **Compliance** — GDPR export/erasure, EU AI Act status, audit trail
- **Learning** — Adaptive ranking progress, behavioral patterns, outcome tracking
- **IDE Connections** — Connected AI tools status and configuration
- **Settings** — Mode, provider, auto-capture/recall configuration

> The dashboard runs locally at `http://localhost:8765`. No data leaves your machine.

---

## V3 Engine Features

### Retrieval (4-Channel Hybrid)
- Semantic similarity with Fisher-Rao information geometry
- BM25 keyword matching (persisted tokens, survives restart)
- Entity graph with spreading activation (3-hop, decay=0.7)
- Temporal date-aware retrieval with interval support
- RRF fusion (k=60) + cross-encoder reranking

### Intelligence
- 11-step ingestion pipeline (entity resolution, fact extraction, emotional tagging, scene building, sheaf consistency)
- Adaptive learning with LightGBM-based ranking (3-phase bootstrap)
- Behavioral pattern detection (query habits, entity preferences, active hours)
- Outcome tracking for retrieval feedback loops

### Trust & Compliance
- Bayesian Beta distribution trust scoring (per-agent, per-fact)
- Trust gates (block low-trust agents from writing/deleting)
- ABAC (Attribute-Based Access Control) with DB-persisted policies
- GDPR Article 15/17 compliance (full export + complete erasure)
- EU AI Act data sovereignty (Mode A: zero cloud, data stays local)
- Tamper-proof hash-chain audit trail (SHA-256 linked entries)
- Data provenance tracking (who created what, when, from where)

### Infrastructure
- 17-tab web dashboard (trust visualization, math health, recall lab)
- 17+ IDE integrations with pre-built configs
- Profile isolation (16+ independent memory spaces)
- V2 to V3 migration tool (zero data loss, rollback support)
- Auto-capture and auto-recall hooks for Claude Code

---

## Benchmarks

Evaluated on the [LoCoMo benchmark](https://arxiv.org/abs/2402.09714) (Long Conversation Memory):

### Mode A Ablation (conv-30, 81 questions, zero-LLM)

| Configuration | Micro Avg | Multi-Hop | Open Domain |
|:-------------|:---------:|:---------:|:-----------:|
| Full (all layers) | **62.3%** | **50%** | **78%** |
| Math layers off | 59.3% | 38% | 70% |
| Entity channel off | 56.8% | 38% | 73% |
| BM25 channel off | 53.2% | 23% | 71% |
| Cross-encoder off | 31.8% | 17% | — |

### Competitive Landscape

| System | Score | LLM Required | Open Source | EU AI Act |
|:-------|:-----:|:------------:|:-----------:|:---------:|
| EverMemOS | 92.3% | Yes | No | No |
| MemMachine | 91.7% | Yes | No | No |
| Hindsight | 89.6% | Yes | No | No |
| **SLM V3 Mode C** | **87.7%** | Optional | **Yes** | Partial |
| **SLM V3 Mode A** | **62.3%** | **No** | **Yes** | **Yes** |
| Mem0 ($24M) | 34.2% F1 | Yes | Partial | No |

*SLM V3 is the only system offering a fully local mode with mathematical guarantees and EU AI Act compliance.*

---

## Research Papers

### V3: Information-Geometric Foundations
> **SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**
> Varun Pratap Bhardwaj (2026)
> [Zenodo DOI: 10.5281/zenodo.19038659](https://zenodo.org/records/19038659)

### V2: Architecture & Engineering
> **SuperLocalMemory: A Structured Local Memory Architecture for Persistent AI Agent Context**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) | [Zenodo DOI: 10.5281/zenodo.18709670](https://zenodo.org/records/18709670)

---

## Project Structure

```
superlocalmemory/
├── src/superlocalmemory/     # Python package (17 sub-packages)
│   ├── core/                 # Engine, config, modes, profiles
│   ├── retrieval/            # 4-channel retrieval + fusion + reranking
│   ├── math/                 # Fisher-Rao, Sheaf, Langevin
│   ├── encoding/             # 11-step ingestion pipeline
│   ├── storage/              # SQLite with WAL, FTS5, migrations
│   ├── trust/                # Bayesian scoring, gates, provenance
│   ├── compliance/           # GDPR, EU AI Act, ABAC, audit chain
│   ├── learning/             # Adaptive ranking, behavioral patterns
│   ├── mcp/                  # MCP server (24 tools, 6 resources)
│   ├── cli/                  # CLI with setup wizard
│   └── server/               # Dashboard API + UI server
├── tests/                    # 1400+ tests
├── ui/                       # 17-tab web dashboard
├── ide/                      # IDE configs for 17+ tools
├── docs/                     # Documentation
└── pyproject.toml            # Modern Python packaging
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE).

## Attribution

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)

---

<p align="center">
  <sub>Built with mathematical rigor. Not in the race — here to help everyone build better AI memory systems.</sub>
</p>
